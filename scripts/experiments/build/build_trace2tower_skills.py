from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.high_paths import mine_high_paths
from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.renderer import (
    RendererStyle,
    render_high_card,
    render_mid_card,
)
from trace2tower.methods.trace2tower.skills import (
    LOW_SKILLS,
    HighSkillCard,
    MidSkillCard,
    build_mid_render_inputs,
)


def write_rendered_cards(
    output: Path,
    mid_cards: dict[str, MidSkillCard],
    high_cards: dict[str, HighSkillCard],
    usages: list[dict],
) -> None:
    write_json(
        output,
        {
            "mid_cards": [mid_cards[skill_id].to_record() for skill_id in sorted(mid_cards)],
            "high_cards": [high_cards[skill_id].to_record() for skill_id in sorted(high_cards)],
            "usage": usages,
        },
    )


def build_high_render_examples(
    records: list[dict],
    clusters: tuple[MidCluster, ...],
    path,
    *,
    limit: int = 8,
) -> tuple[dict, ...]:
    segment_to_mid = {
        segment_id: cluster.cluster_id
        for cluster in clusters
        for segment_id in cluster.member_segment_ids
    }
    records_by_id = {record["trajectory_id"]: record for record in records}
    examples = []
    seen_samples = set()
    for trajectory_id in path.supporting_trajectory_ids:
        record = records_by_id[trajectory_id]
        sample_id = record["sample_id"]
        if sample_id in seen_samples:
            continue
        groups = []
        for segment in sorted(record["segments"], key=lambda item: item["start_step"]):
            mid_id = segment_to_mid[segment["segment_id"]]
            if groups and groups[-1][0] == mid_id:
                groups[-1][1].extend(segment["raw_actions"])
            else:
                groups.append([mid_id, list(segment["raw_actions"])])
        ordered_mid_ids = tuple(group[0] for group in groups)
        start = next(
            index
            for index in range(len(ordered_mid_ids) - len(path.ordered_mid_ids) + 1)
            if ordered_mid_ids[index : index + len(path.ordered_mid_ids)] == path.ordered_mid_ids
        )
        path_groups = groups[start : start + len(path.ordered_mid_ids)]
        actions = [action for _, group_actions in path_groups for action in group_actions]
        goal = next(
            transition["goal"] for transition in record["transitions"] if transition.get("goal")
        )
        examples.append(
            {
                "goal": goal,
                "path_steps": [
                    {"mid_id": mid_id, "raw_actions": group_actions}
                    for mid_id, group_actions in path_groups
                ],
                "raw_actions": actions,
            }
        )
        seen_samples.add(sample_id)
        if len(examples) >= limit:
            break
    return tuple(examples)


async def main(options: argparse.Namespace) -> int:
    if options.render_high_limit < 0:
        raise ValueError("render High limit must be non-negative")
    if options.structure_only and options.render_all_mid:
        raise ValueError("structure-only build cannot render Mid cards")
    load_dotenv(options.env)
    config_record = load_yaml(options.config)
    config = Trace2TowerConfig.from_record(config_record)
    records = [
        json.loads(line) for line in options.input.read_text(encoding="utf-8").splitlines() if line
    ]
    cluster_records = json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    clusters = tuple(
        MidCluster(
            cluster_id=item["cluster_id"],
            member_segment_ids=tuple(item["member_segment_ids"]),
            centroid=tuple(item["centroid"]),
        )
        for item in cluster_records
    )
    mid_inputs = build_mid_render_inputs(records, clusters)
    high_paths = (
        ()
        if config.semantic_only
        else mine_high_paths(
            records,
            clusters,
            max_path_length=config.max_high_path_length,
            min_support_ratio=config.high_min_support_ratio,
            epsilon=config.high_path_epsilon,
            success_threshold=config.success_threshold,
        )
    )
    used_high_fallback = False
    if not high_paths and options.ensure_high_path and not config.semantic_only:
        high_paths = mine_high_paths(
            records,
            clusters,
            max_path_length=config.max_high_path_length,
            min_support_ratio=0.0,
            epsilon=config.high_path_epsilon,
            success_threshold=config.success_threshold,
        )[:1]
        used_high_fallback = bool(high_paths)
    invocation = {
        "benchmark": options.benchmark.value,
        "input": options.input.as_posix(),
        "clusters": options.clusters.as_posix(),
        "config": options.config.as_posix(),
        "output_dir": options.output_dir.as_posix(),
        "render_high_limit": options.render_high_limit,
        "render_all_mid": options.render_all_mid,
        "structure_only": options.structure_only,
        "success_threshold": config.success_threshold,
        "ensure_high_path": options.ensure_high_path,
        "renderer_style": options.renderer_style.value,
    }
    print(yaml.safe_dump({"method_config": config_record, "invocation": invocation}))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        options.output_dir / "low-skills.json",
        {"skills": [asdict(skill) for skill in LOW_SKILLS[options.benchmark]]},
    )
    write_json(
        options.output_dir / "mid-render-inputs.json",
        {"inputs": [item.to_record() for item in mid_inputs]},
    )
    write_json(
        options.output_dir / "high-paths.json",
        {"paths": [path.to_record() for path in high_paths]},
    )

    rendered_cards_path = options.output_dir / "rendered-cards.json"
    mid_cards: dict[str, MidSkillCard] = {}
    high_cards: dict[str, HighSkillCard] = {}
    usages = []
    if rendered_cards_path.exists():
        existing = json.loads(rendered_cards_path.read_text(encoding="utf-8"))
        mid_cards = {
            card.skill_id: card
            for card in (MidSkillCard.from_record(item) for item in existing["mid_cards"])
        }
        high_cards = {
            card.skill_id: card
            for card in (HighSkillCard.from_record(item) for item in existing["high_cards"])
        }
        usages = list(existing.get("usage", ()))
    elif options.reuse_mid_cards_from:
        existing = json.loads(options.reuse_mid_cards_from.read_text(encoding="utf-8"))
        mid_cards = {
            card.skill_id: card
            for card in (MidSkillCard.from_record(item) for item in existing["mid_cards"])
        }
        source_mid_ids = set(mid_cards)
        usages = [item for item in existing.get("usage", ()) if item["skill_id"] in source_mid_ids]

    inputs_by_id = {item.cluster_id: item for item in mid_inputs}
    paths_by_id = {path.path_id: path for path in high_paths}
    if set(mid_cards) - set(inputs_by_id) or set(high_cards) - set(paths_by_id):
        raise ValueError("rendered cards do not belong to the current tower structure")
    for skill_id, card in mid_cards.items():
        if card.member_segment_ids != inputs_by_id[skill_id].member_segment_ids:
            raise ValueError(f"rendered Mid card membership changed: {skill_id}")
    for skill_id, card in high_cards.items():
        if card.ordered_mid_ids != paths_by_id[skill_id].ordered_mid_ids:
            raise ValueError(f"rendered High card path changed: {skill_id}")

    reused_mid_count = len(mid_cards)
    reused_high_count = len(high_cards)
    selected_paths = (
        ()
        if options.structure_only
        else high_paths
        if options.render_high_limit == 0
        else high_paths[: options.render_high_limit]
    )
    required_mid_ids = tuple(
        inputs_by_id
        if options.render_all_mid
        else dict.fromkeys(mid_id for path in selected_paths for mid_id in path.ordered_mid_ids)
    )
    missing_mid_ids = tuple(skill_id for skill_id in required_mid_ids if skill_id not in mid_cards)
    missing_paths = tuple(path for path in selected_paths if path.path_id not in high_cards)
    if missing_mid_ids or missing_paths:
        common = load_yaml(options.config_root / "common.yaml")
        runtime = CommonLLMRuntime(
            max_concurrency=common["global_api_concurrency"],
            max_attempts=common["provider_max_attempts"],
            timeout_seconds=common["provider_timeout_seconds"],
            retry_base_seconds=common["retry_base_seconds"],
        )
        try:
            rendered_mids = await asyncio.gather(
                *(
                    render_mid_card(
                        runtime,
                        options.benchmark,
                        inputs_by_id[mid_id],
                        renderer_style=options.renderer_style,
                    )
                    for mid_id in missing_mid_ids
                )
            )
            for mid_id, (card, result) in zip(missing_mid_ids, rendered_mids):
                mid_cards[mid_id] = card
                usages.append(
                    {
                        "skill_id": mid_id,
                        **asdict(result.usage),
                        "latency_ms": result.latency_ms,
                    }
                )
                write_rendered_cards(rendered_cards_path, mid_cards, high_cards, usages)
            rendered_highs = await asyncio.gather(
                *(
                    render_high_card(
                        runtime,
                        path,
                        mid_cards,
                        build_high_render_examples(records, clusters, path),
                        renderer_style=options.renderer_style,
                    )
                    for path in missing_paths
                )
            )
            for path, (card, result) in zip(missing_paths, rendered_highs):
                high_cards[path.path_id] = card
                usages.append(
                    {
                        "skill_id": path.path_id,
                        **asdict(result.usage),
                        "latency_ms": result.latency_ms,
                    }
                )
                write_rendered_cards(rendered_cards_path, mid_cards, high_cards, usages)
        finally:
            await runtime.close()
    if mid_cards or high_cards:
        write_rendered_cards(rendered_cards_path, mid_cards, high_cards, usages)

    report = {
        "renderer_model": os.environ.get("RENDERER_MODEL"),
        **invocation,
        "trajectory_count": len(records),
        "mid_cluster_count": len(mid_inputs),
        "high_path_count": len(high_paths),
        "used_high_fallback": used_high_fallback,
        "rendered_mid_count": len(mid_cards),
        "rendered_high_count": len(high_cards),
        "reused_mid_count": reused_mid_count,
        "reused_high_count": reused_high_count,
        "new_mid_count": len(mid_cards) - reused_mid_count,
        "new_high_count": len(high_cards) - reused_high_count,
        "max_high_path_length": config.max_high_path_length,
        "high_min_support_ratio": config.high_min_support_ratio,
        "high_path_epsilon": config.high_path_epsilon,
    }
    write_json(options.output_dir / "report.json", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-high-limit", type=int, default=0)
    parser.add_argument("--render-all-mid", action="store_true")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--ensure-high-path", action="store_true")
    parser.add_argument("--reuse-mid-cards-from", type=Path)
    parser.add_argument(
        "--renderer-style",
        type=RendererStyle,
        choices=tuple(RendererStyle),
        default=RendererStyle.TRACE2TOWER,
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
