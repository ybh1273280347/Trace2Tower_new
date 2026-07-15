from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

from dotenv import load_dotenv

from scripts.experiments.build.build_trace2tower_skills import (
    build_high_render_examples,
    write_rendered_cards,
)
from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.full_refinement import (
    StructuralSelection,
    apply_mid_updates,
    project_high_path,
)
from trace2tower.methods.trace2tower.high_paths import mine_high_paths
from trace2tower.methods.trace2tower.lineage import (
    build_mid_lineage,
    decompose_mid_lineage,
)
from trace2tower.methods.trace2tower.models import HighPath, MidCluster, SegmentInstance
from trace2tower.methods.trace2tower.refinement import LegalSplitProposal
from trace2tower.methods.trace2tower.renderer import render_high_card, render_mid_card
from trace2tower.methods.trace2tower.skills import (
    HighSkillCard,
    MidSkillCard,
    build_mid_render_inputs,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


async def main(options: argparse.Namespace) -> int:
    old_tower = TowerSnapshot.from_record(
        json.loads(options.tower.read_text(encoding="utf-8"))
    )
    records = [
        json.loads(line)
        for line in options.preprocessed.read_text(encoding="utf-8").splitlines()
        if line
    ]
    segments = {
        segment.segment_id: segment
        for record in records
        for segment in (
            SegmentInstance.from_record(item) for item in record["segments"]
        )
    }
    candidate_clusters = tuple(
        MidCluster.from_record(record)
        for record in json.loads(
            options.candidate_clusters.read_text(encoding="utf-8")
        )["clusters"]
    )
    structural_pareto = json.loads(
        options.structural_pareto.read_text(encoding="utf-8")
    )
    if structural_pareto["tower_snapshot_id"] != old_tower.snapshot_id:
        raise ValueError("structural Pareto report does not bind to the source Tower")
    selected_component_id = structural_pareto["selected_component_id"]
    if selected_component_id is None:
        raise ValueError("structural Pareto selected no refinement component")

    lineage = build_mid_lineage(old_tower.mid_clusters, candidate_clusters)
    decomposition = decompose_mid_lineage(lineage)
    component_index = int(selected_component_id.rsplit("_", 1)[1])
    source_ids, target_ids = decomposition.components[component_index]
    if len(source_ids) != 1 or len(target_ids) < 2:
        raise ValueError("the selected builder currently requires one Pareto Split")
    source_id = source_ids[0]
    local_lineage = replace(
        lineage,
        continuations=decomposition.continuations,
        splits=((source_id, target_ids),),
        merges=(),
        complex_repartitions=(),
    )
    selection = StructuralSelection(
        split=LegalSplitProposal(
            proposal_id=f"pareto-split:{source_id}:{','.join(target_ids)}",
            source_skill_id=source_id,
        ),
        merge=None,
        repartition=None,
        rejected_split_proposal_ids=(),
        rejected_repartition_proposal_ids=(),
        rejected_merge_proposal_ids=(),
        pareto_protected_merge_proposal_ids=(),
    )
    refined = apply_mid_updates(
        local_lineage,
        selection,
        old_tower.mid_clusters,
        candidate_clusters,
        segments,
    )
    max_high_path_length = (
        options.max_high_path_length
        if options.max_high_path_length is not None
        else old_tower.config.max_high_path_length
    )
    mined_paths = mine_high_paths(
        records,
        refined.clusters,
        max_path_length=max_high_path_length,
        min_support_ratio=old_tower.config.high_min_support_ratio,
        epsilon=old_tower.config.high_path_epsilon,
        success_threshold=old_tower.config.success_threshold,
    )
    mined_by_order = {path.ordered_mid_ids: path for path in mined_paths}
    projected_paths = []
    for old_path in old_tower.high_paths:
        ordered_mid_ids = project_high_path(
            old_path.ordered_mid_ids,
            refined.primary_replacement_by_old_id,
        )
        mined = mined_by_order.get(ordered_mid_ids)
        projected_paths.append(
            HighPath(
                path_id=old_path.path_id,
                ordered_mid_ids=ordered_mid_ids,
                positive_support=(mined or old_path).positive_support,
                negative_support=(mined or old_path).negative_support,
                contrastive_score=(mined or old_path).contrastive_score,
                supporting_trajectory_ids=(mined or old_path).supporting_trajectory_ids,
            )
        )
    existing_orders = {path.ordered_mid_ids for path in projected_paths}
    promotion_candidates = [
        path for path in mined_paths if path.ordered_mid_ids not in existing_orders
    ]
    promoted = max(
        promotion_candidates,
        key=lambda path: (
            path.contrastive_score,
            path.positive_support,
            path.path_id,
        ),
        default=None,
    )
    final_paths = tuple(
        sorted(
            (*projected_paths, *((promoted,) if promoted is not None else ())),
            key=lambda path: path.path_id,
        )
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    cards_path = options.output_dir / "rendered-cards.json"
    mid_cards: dict[str, MidSkillCard] = {}
    high_cards: dict[str, HighSkillCard] = {}
    usages = []
    if cards_path.exists():
        checkpoint = json.loads(cards_path.read_text(encoding="utf-8"))
        mid_cards = {
            item["skill_id"]: MidSkillCard.from_record(item)
            for item in checkpoint["mid_cards"]
        }
        high_cards = {
            item["skill_id"]: HighSkillCard.from_record(item)
            for item in checkpoint["high_cards"]
        }
        usages = list(checkpoint.get("usage", ()))

    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    inputs_by_id = {
        item.cluster_id: item
        for item in build_mid_render_inputs(records, refined.clusters)
    }
    old_mid_cards = {card.skill_id: card for card in old_tower.mid_cards}
    split_ids = refined.replacement_by_old_id[source_id]
    try:
        for cluster in refined.clusters:
            if cluster.cluster_id in mid_cards:
                continue
            if cluster.cluster_id in old_mid_cards:
                mid_cards[cluster.cluster_id] = replace(
                    old_mid_cards[cluster.cluster_id],
                    member_segment_ids=cluster.member_segment_ids,
                )
                continue
            card, result = await render_mid_card(
                runtime,
                old_tower.benchmark,
                inputs_by_id[cluster.cluster_id],
                tuple(
                    inputs_by_id[sibling_id]
                    for sibling_id in split_ids
                    if sibling_id != cluster.cluster_id
                ),
            )
            mid_cards[card.skill_id] = card
            usages.append({"skill_id": card.skill_id, **asdict(result.usage)})
            write_rendered_cards(cards_path, mid_cards, high_cards, usages)

        old_cards = {card.skill_id: card for card in old_tower.high_cards}
        for path in final_paths:
            if path.path_id in high_cards:
                continue
            if path.path_id in old_cards:
                high_cards[path.path_id] = replace(
                    old_cards[path.path_id],
                    ordered_mid_ids=path.ordered_mid_ids,
                )
                continue
            card, result = await render_high_card(
                runtime,
                path,
                mid_cards,
                build_high_render_examples(records, refined.clusters, path),
            )
            high_cards[card.skill_id] = card
            usages.append({"skill_id": card.skill_id, **asdict(result.usage)})
            write_rendered_cards(cards_path, mid_cards, high_cards, usages)
    finally:
        await runtime.close()

    ordered_mid_cards = tuple(mid_cards[skill_id] for skill_id in sorted(mid_cards))
    ordered_high_cards = tuple(high_cards[skill_id] for skill_id in sorted(high_cards))
    write_json(
        options.output_dir / "mid-clusters.json",
        {"clusters": [cluster.to_record() for cluster in refined.clusters]},
    )
    write_json(
        options.output_dir / "high-paths.json",
        {"paths": [path.to_record() for path in final_paths]},
    )
    write_rendered_cards(cards_path, mid_cards, high_cards, usages)
    write_json(
        options.output_dir / "report.json",
        {
            "source_tower_snapshot_id": old_tower.snapshot_id,
            "selected_component_id": selected_component_id,
            "selected_split_source_id": source_id,
            "selected_split_target_candidate_ids": target_ids,
            "refined_mid_count": len(refined.clusters),
            "mined_high_count": len(mined_paths),
            "max_high_path_length": max_high_path_length,
            "retained_high_count": len(projected_paths),
            "promoted_high_id": promoted.path_id if promoted else None,
            "final_high_count": len(final_paths),
            "rendered_skill_ids": [item["skill_id"] for item in usages],
        },
    )
    print(
        json.dumps(
            {
                "mid_count": len(ordered_mid_cards),
                "high_count": len(ordered_high_cards),
                "promoted_high_id": promoted.path_id if promoted else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--candidate-clusters", type=Path, required=True)
    parser.add_argument("--structural-pareto", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-high-path-length", type=int)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
