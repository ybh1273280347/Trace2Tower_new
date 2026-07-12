from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rollout_no_skill_train import load_yaml, write_json

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.high_paths import mine_high_paths
from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.renderer import render_high_card, render_mid_card
from trace2tower.methods.trace2tower.skills import LOW_SKILLS, build_mid_render_inputs


async def main(options: argparse.Namespace) -> int:
    if options.render_high_limit < 0:
        raise ValueError("render High limit must be non-negative")
    config_record = load_yaml(options.config)
    config = Trace2TowerConfig.from_record(config_record)
    records = [
        json.loads(line)
        for line in options.input.read_text(encoding="utf-8").splitlines()
        if line
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
    high_paths = mine_high_paths(
        records,
        clusters,
        max_path_length=config.max_high_path_length,
        min_support_ratio=config.high_min_support_ratio,
        epsilon=config.high_path_epsilon,
        success_threshold=options.success_threshold,
    )
    invocation = {
        "benchmark": options.benchmark.value,
        "input": options.input.as_posix(),
        "clusters": options.clusters.as_posix(),
        "config": options.config.as_posix(),
        "output_dir": options.output_dir.as_posix(),
        "render_high_limit": options.render_high_limit,
        "success_threshold": options.success_threshold,
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

    mid_cards = {}
    high_cards = []
    usages = []
    if options.render_high_limit:
        load_dotenv(options.env)
        common = load_yaml(options.config_root / "common.yaml")
        runtime = CommonLLMRuntime(
            max_concurrency=common["global_api_concurrency"],
            max_attempts=common["provider_max_attempts"],
            timeout_seconds=common["provider_timeout_seconds"],
            retry_base_seconds=common["retry_base_seconds"],
        )
        inputs_by_id = {item.cluster_id: item for item in mid_inputs}
        selected_paths = high_paths[: options.render_high_limit]
        required_mid_ids = tuple(
            dict.fromkeys(mid_id for path in selected_paths for mid_id in path.ordered_mid_ids)
        )
        try:
            for mid_id in required_mid_ids:
                card, result = await render_mid_card(
                    runtime, options.benchmark, inputs_by_id[mid_id]
                )
                mid_cards[mid_id] = card
                usages.append(
                    {
                        "skill_id": mid_id,
                        **asdict(result.usage),
                        "latency_ms": result.latency_ms,
                    }
                )
            for path in selected_paths:
                card, result = await render_high_card(runtime, path, mid_cards)
                high_cards.append(card)
                usages.append(
                    {
                        "skill_id": path.path_id,
                        **asdict(result.usage),
                        "latency_ms": result.latency_ms,
                    }
                )
        finally:
            await runtime.close()
        write_json(
            options.output_dir / "rendered-cards.json",
            {
                "mid_cards": [card.to_record() for card in mid_cards.values()],
                "high_cards": [card.to_record() for card in high_cards],
                "usage": usages,
            },
        )

    report = {
        **invocation,
        "trajectory_count": len(records),
        "mid_cluster_count": len(mid_inputs),
        "high_path_count": len(high_paths),
        "rendered_mid_count": len(mid_cards),
        "rendered_high_count": len(high_cards),
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
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
