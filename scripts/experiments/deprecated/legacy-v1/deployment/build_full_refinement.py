from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.full_refinement import (
    apply_mid_updates,
    project_high_path,
    project_mid_objectives,
    select_structural_updates,
)
from trace2tower.methods.trace2tower.high_paths import mine_high_paths
from trace2tower.methods.trace2tower.lineage import (
    build_mid_lineage,
    legal_merge_proposals,
    legal_split_proposals,
)
from trace2tower.methods.trace2tower.models import HighPath, MidCluster, SegmentInstance
from trace2tower.methods.trace2tower.refinement import (
    LegalPromoteProposal,
    RankedSkillObjective,
    SkillLevel,
    rank_promotions,
    select_downweight,
    select_promote,
)
from trace2tower.methods.trace2tower.renderer import render_high_card, render_mid_card
from trace2tower.methods.trace2tower.retrieval import (
    SkillEmbeddingIndex,
    high_card_text,
    mid_card_text,
)
from trace2tower.methods.trace2tower.skills import (
    HighSkillCard,
    MidSkillCard,
    build_mid_render_inputs,
)
from trace2tower.methods.trace2tower.tower import (
    TowerSnapshot,
    TowerSourceHashes,
    TowerVersion,
    build_tower_snapshot,
    sha256_file,
)


async def build_index(
    runtime: CommonLLMRuntime,
    texts: dict[str, str],
    old_index: SkillEmbeddingIndex,
) -> SkillEmbeddingIndex:
    text_hashes = {
        skill_id: hashlib.sha256(text.encode()).hexdigest()
        for skill_id, text in texts.items()
    }
    reusable = {
        skill_id: vector
        for skill_id, vector, text_hash in zip(
            old_index.skill_ids,
            old_index.vectors,
            old_index.text_hashes,
            strict=True,
        )
        if text_hashes.get(skill_id) == text_hash
    }
    missing_ids = tuple(skill_id for skill_id in texts if skill_id not in reusable)
    result = await runtime.embed([texts[skill_id] for skill_id in missing_ids]) if missing_ids else None
    vectors = reusable | dict(
        zip(missing_ids, result.vectors if result else (), strict=True)
    )
    return SkillEmbeddingIndex(
        skill_ids=tuple(texts),
        vectors=tuple(vectors[skill_id] for skill_id in texts),
        text_hashes=tuple(text_hashes.values()),
    )


async def main(options: argparse.Namespace) -> int:
    old_tower = TowerSnapshot.from_record(
        json.loads(options.tower.read_text(encoding="utf-8"))
    )
    refinement_report = json.loads(
        options.refinement_report.read_text(encoding="utf-8")
    )
    if (
        refinement_report["tower_snapshot_id"] != old_tower.snapshot_id
        or refinement_report["ranking_status"] != "complete"
        or not refinement_report["audit"]["is_complete"]
    ):
        raise ValueError("refinement report does not bind complete v0 evidence")
    records = [
        json.loads(line)
        for line in options.preprocessed.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(
        record["benchmark"] != old_tower.benchmark
        or record["split"] != "train"
        for record in records
    ):
        raise ValueError("full refinement accepts only matching training trajectories")
    candidate_payload = json.loads(
        options.candidate_clusters.read_text(encoding="utf-8")
    )
    candidate_clusters = tuple(
        MidCluster.from_record(item) for item in candidate_payload["clusters"]
    )
    segments = {
        segment.segment_id: segment
        for record in records
        for segment in (
            SegmentInstance.from_record(item) for item in record["segments"]
        )
    }
    if len(segments) != sum(len(record["segments"]) for record in records):
        raise ValueError("refinement preprocessed data contains duplicate segment IDs")

    ranked = tuple(
        RankedSkillObjective.from_record(item)
        for item in refinement_report["ranked_skills"]
    )
    ranked_mid = {
        item.skill_id: item for item in ranked if item.skill_level is SkillLevel.MID
    }
    lineage = build_mid_lineage(old_tower.mid_clusters, candidate_clusters)
    selection = select_structural_updates(
        lineage,
        old_tower.mid_clusters,
        candidate_clusters,
        old_tower.high_paths,
        ranked_mid,
        max_high_path_length=old_tower.config.max_high_path_length,
        minimum_exposure_count=int(
            refinement_report["refinement_config"]["minimum_exposure_count"]
        ),
        excluded_proposal_ids=frozenset(options.exclude_structural_proposal),
    )
    refined_mid = apply_mid_updates(
        lineage,
        selection,
        old_tower.mid_clusters,
        candidate_clusters,
        segments,
    )
    projected_mid_objectives = project_mid_objectives(refined_mid, ranked_mid)

    mined_paths = mine_high_paths(
        records,
        refined_mid.clusters,
        max_path_length=old_tower.config.max_high_path_length,
        min_support_ratio=old_tower.config.high_min_support_ratio,
        epsilon=old_tower.config.high_path_epsilon,
        success_threshold=options.success_threshold,
    )
    mined_by_order = {path.ordered_mid_ids: path for path in mined_paths}
    projected_high_paths = []
    for old_path in old_tower.high_paths:
        ordered_mid_ids = project_high_path(
            old_path.ordered_mid_ids, refined_mid.primary_replacement_by_old_id
        )
        mined = mined_by_order.get(ordered_mid_ids)
        projected_high_paths.append(
            HighPath(
                path_id=old_path.path_id,
                ordered_mid_ids=ordered_mid_ids,
                positive_support=(mined or old_path).positive_support,
                negative_support=(mined or old_path).negative_support,
                contrastive_score=(mined or old_path).contrastive_score,
                supporting_trajectory_ids=(
                    mined or old_path
                ).supporting_trajectory_ids,
            )
        )
    existing_orders = {path.ordered_mid_ids for path in projected_high_paths}
    promote_candidates = tuple(
        LegalPromoteProposal(
            path_id=path.path_id,
            benchmark=old_tower.benchmark,
            refinement_round=int(refinement_report["refinement_round"]),
            child_mid_ids=path.ordered_mid_ids,
            contrastive_path_score=path.contrastive_score,
            positive_support=path.positive_support,
        )
        for path in mined_paths
        if path.ordered_mid_ids not in existing_orders
        and all(mid_id in projected_mid_objectives for mid_id in path.ordered_mid_ids)
    )
    ranked_promotions = rank_promotions(
        promote_candidates, projected_mid_objectives
    )
    selected_promotion = select_promote(ranked_promotions)
    final_high_paths = list(projected_high_paths)
    if selected_promotion:
        selected_path = next(
            path
            for path in mined_paths
            if path.path_id == selected_promotion.proposal.path_id
        )
        final_high_paths.append(selected_path)
    final_high_paths = tuple(sorted(final_high_paths, key=lambda path: path.path_id))

    options.output_dir.mkdir(parents=True, exist_ok=True)
    lineage_path = options.output_dir / "lineage.json"
    clusters_path = options.output_dir / "mid-clusters.json"
    paths_path = options.output_dir / "high-paths.json"
    cards_path = options.output_dir / "rendered-cards.json"
    index_path = options.output_dir / "index.json"
    mid_cards: dict[str, MidSkillCard] = {}
    high_cards: dict[str, HighSkillCard] = {}
    render_usage = []
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
        render_usage = list(checkpoint.get("usage", ()))

    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    old_mid_cards = {card.skill_id: card for card in old_tower.mid_cards}
    mid_inputs = {
        item.cluster_id: item
        for item in build_mid_render_inputs(records, refined_mid.clusters)
    }
    sibling_ids_by_mid: dict[str, tuple[str, ...]] = {}
    structural_groups = []
    if selection.split:
        structural_groups.append(
            refined_mid.replacement_by_old_id[selection.split.source_skill_id]
        )
    if selection.repartition:
        structural_groups.append(
            tuple(
                refined_mid.candidate_to_final_mid_id[candidate_id]
                for candidate_id in selection.repartition.candidate_cluster_ids
            )
        )
    for group in structural_groups:
        for mid_id in group:
            sibling_ids_by_mid[mid_id] = tuple(
                sibling_id for sibling_id in group if sibling_id != mid_id
            )
    try:
        for cluster in refined_mid.clusters:
            if cluster.cluster_id in mid_cards:
                if mid_cards[cluster.cluster_id].member_segment_ids != cluster.member_segment_ids:
                    raise ValueError("checkpointed Mid card belongs to another structure")
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
                mid_inputs[cluster.cluster_id],
                tuple(
                    mid_inputs[sibling_id]
                    for sibling_id in sibling_ids_by_mid.get(cluster.cluster_id, ())
                ),
            )
            mid_cards[cluster.cluster_id] = card
            render_usage.append({"skill_id": card.skill_id, **asdict(result.usage)})
            write_json(
                cards_path,
                {
                    "mid_cards": [
                        mid_cards[skill_id].to_record()
                        for skill_id in sorted(mid_cards)
                    ],
                    "high_cards": [
                        high_cards[skill_id].to_record()
                        for skill_id in sorted(high_cards)
                    ],
                    "usage": render_usage,
                },
            )

        old_high_cards = {card.skill_id: card for card in old_tower.high_cards}
        old_high_paths = {path.path_id: path for path in old_tower.high_paths}
        for path in final_high_paths:
            if path.path_id in high_cards:
                if high_cards[path.path_id].ordered_mid_ids != path.ordered_mid_ids:
                    raise ValueError("checkpointed High card belongs to another structure")
                continue
            if (
                path.path_id in old_high_cards
                and path.ordered_mid_ids
                == old_high_paths[path.path_id].ordered_mid_ids
            ):
                high_cards[path.path_id] = old_high_cards[path.path_id]
                continue
            card, result = await render_high_card(
                runtime,
                old_tower.benchmark,
                path,
                mid_cards,
            )
            high_cards[path.path_id] = card
            render_usage.append({"skill_id": card.skill_id, **asdict(result.usage)})
            write_json(
                cards_path,
                {
                    "mid_cards": [
                        mid_cards[skill_id].to_record()
                        for skill_id in sorted(mid_cards)
                    ],
                    "high_cards": [
                        high_cards[skill_id].to_record()
                        for skill_id in sorted(high_cards)
                    ],
                    "usage": render_usage,
                },
            )

        ordered_mid_cards = tuple(mid_cards[skill_id] for skill_id in sorted(mid_cards))
        ordered_high_cards = tuple(
            high_cards[skill_id] for skill_id in sorted(high_cards)
        )
        reusable_mid_index = old_tower.mid_index
        reusable_high_index = old_tower.high_index
        if index_path.exists():
            checkpoint_index = json.loads(index_path.read_text(encoding="utf-8"))
            reusable_mid_index = SkillEmbeddingIndex.from_record(
                checkpoint_index["mid_index"]
            )
            reusable_high_index = SkillEmbeddingIndex.from_record(
                checkpoint_index["high_index"]
            )
        mid_index = await build_index(
            runtime,
            {
                card.skill_id: mid_card_text(card) for card in ordered_mid_cards
            },
            reusable_mid_index,
        )
        high_index = await build_index(
            runtime,
            {
                card.skill_id: high_card_text(card) for card in ordered_high_cards
            },
            reusable_high_index,
        )
    finally:
        await runtime.close()

    write_json(lineage_path, lineage.to_record())
    write_json(
        clusters_path,
        {"clusters": [cluster.to_record() for cluster in refined_mid.clusters]},
    )
    write_json(paths_path, {"paths": [path.to_record() for path in final_high_paths]})
    write_json(
        cards_path,
        {
            "mid_cards": [card.to_record() for card in ordered_mid_cards],
            "high_cards": [card.to_record() for card in ordered_high_cards],
            "usage": render_usage,
        },
    )
    write_json(
        index_path,
        {
            "mid_index": mid_index.to_record(),
            "high_index": high_index.to_record(),
        },
    )
    snapshot = build_tower_snapshot(
        version=TowerVersion.V1,
        benchmark=old_tower.benchmark,
        config=old_tower.config,
        training_trajectory_ids=tuple(record["trajectory_id"] for record in records),
        source_hashes=TowerSourceHashes(
            preprocessed_trajectories=sha256_file(options.preprocessed),
            clusters=sha256_file(clusters_path),
            high_paths=sha256_file(paths_path),
            rendered_cards=sha256_file(cards_path),
            retrieval_index=sha256_file(index_path),
        ),
        low_skills=old_tower.low_skills,
        mid_clusters=refined_mid.clusters,
        high_paths=final_high_paths,
        mid_cards=ordered_mid_cards,
        high_cards=ordered_high_cards,
        mid_index=mid_index,
        high_index=high_index,
    )
    snapshot.require_complete()
    tower_path = options.output_dir / "tower-v1.json"
    write_json(tower_path, snapshot.to_record())

    minimum_exposure = int(
        refinement_report["refinement_config"]["minimum_exposure_count"]
    )
    downweight = []
    for level in (SkillLevel.MID, SkillLevel.HIGH):
        retained_ids = (
            {
                mid_id
                for mid_id, sources in refined_mid.source_old_ids_by_mid_id.items()
                if sources == (mid_id,)
            }
            if level is SkillLevel.MID
            else {
                path.path_id
                for path in final_high_paths
                if path.path_id in {item.path_id for item in old_tower.high_paths}
            }
        )
        eligible = {
            item.skill_id
            for item in ranked
            if item.skill_level is level
            and item.skill_id in retained_ids
            and item.exposure_count >= minimum_exposure
            and item.pareto_front_rank > 1
        }
        scoped = [item for item in ranked if item.skill_level is level]
        if eligible:
            downweight.append(select_downweight(scoped, eligible).to_record())
    runtime_states_path = options.output_dir / "runtime-states.json"
    write_json(
        runtime_states_path,
        {
            "tower_snapshot_id": snapshot.snapshot_id,
            "ranking_status": "complete",
            "status_tie_epsilon": refinement_report["refinement_config"][
                "status_tie_epsilon"
            ],
            "downweight": downweight,
        },
    )
    method_config = load_yaml(
        options.config_root / "trace2tower_full_execution.yaml"
    )
    method_config["lifecycle_report"] = runtime_states_path.as_posix()
    method_config_path = options.output_dir / "method-config.yaml"
    method_config_path.write_text(
        yaml.safe_dump(method_config, sort_keys=False), encoding="utf-8"
    )

    report = {
        "source_tower_snapshot_id": old_tower.snapshot_id,
        "tower_snapshot_id": snapshot.snapshot_id,
        "version": snapshot.version.value,
        "refinement_report": options.refinement_report.as_posix(),
        "refinement_report_sha256": sha256_file(options.refinement_report),
        "trajectory_count": len(records),
        "old_mid_count": len(old_tower.mid_clusters),
        "candidate_mid_count": len(candidate_clusters),
        "refined_mid_count": len(refined_mid.clusters),
        "old_high_count": len(old_tower.high_paths),
        "mined_high_count": len(mined_paths),
        "refined_high_count": len(final_high_paths),
        "proposals": {
            "manually_rejected_as_inexpressible": options.exclude_structural_proposal,
            "split": {
                "legal_count": len(legal_split_proposals(lineage)),
                "selected": asdict(selection.split) if selection.split else None,
                "rejected_structural": selection.rejected_split_proposal_ids,
            },
            "merge": {
                "legal_count": len(
                    legal_merge_proposals(
                        lineage, old_tower.mid_clusters, candidate_clusters
                    )
                ),
                "selected": asdict(selection.merge) if selection.merge else None,
                "rejected_structural_or_conflict": selection.rejected_merge_proposal_ids,
                "pareto_protected": selection.pareto_protected_merge_proposal_ids,
            },
            "coordinated_split_merge": {
                "legal_count": len(lineage.complex_repartitions),
                "selected": (
                    selection.repartition.to_record()
                    if selection.repartition
                    else None
                ),
                "rejected_structural": selection.rejected_repartition_proposal_ids,
            },
            "promote": {
                "legal_count": len(promote_candidates),
                "selected": (
                    selected_promotion.proposal.path_id
                    if selected_promotion
                    else None
                ),
            },
            "downweight": downweight,
        },
        "projected_mid_objective_sources": refined_mid.source_old_ids_by_mid_id,
        "rendered_skill_ids": [item["skill_id"] for item in render_usage],
        "lineage": lineage_path.as_posix(),
        "runtime_states": runtime_states_path.as_posix(),
        "method_config": method_config_path.as_posix(),
        "tower": tower_path.as_posix(),
    }
    write_json(options.output_dir / "build-report.json", report)
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--candidate-clusters", type=Path, required=True)
    parser.add_argument("--refinement-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument(
        "--exclude-structural-proposal",
        action="append",
        default=[],
        help="Proposal ID rejected by the recorded training-only renderability review.",
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
