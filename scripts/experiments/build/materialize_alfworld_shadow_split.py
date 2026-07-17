from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.components.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.artifacts.tower import (
    TowerSourceHashes,
    TowerSnapshot,
    TowerVersion,
    build_tower_snapshot,
    sha256_file,
)
from trace2tower.methods.trace2tower.core.models import MidCluster
from trace2tower.methods.trace2tower.induction.skills import build_mid_render_inputs
from trace2tower.methods.trace2tower.inference.formatting import mid_card_text
from trace2tower.methods.trace2tower.rendering.renderer import render_mid_card


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    base = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    structural = json.loads(options.structural_report.read_text(encoding="utf-8"))
    selected = tuple(item for item in structural["selected"] if item["action"] == "split")
    if len(selected) != 1:
        raise ValueError("shadow Split materialization requires exactly one selected Split")
    proposal = selected[0]
    if len(proposal["old_mid_ids"]) != 1 or len(proposal["new_mid_ids"]) < 2:
        raise ValueError("selected Split has an invalid lineage shape")

    source_id = str(proposal["old_mid_ids"][0])
    source = next(cluster for cluster in base.mid_clusters if cluster.cluster_id == source_id)
    candidate_records = {
        item["cluster_id"]: item
        for item in json.loads(options.candidate_clusters.read_text(encoding="utf-8"))["clusters"]
    }
    candidate_ids = tuple(str(item) for item in proposal["new_mid_ids"])
    if not set(candidate_ids) <= set(candidate_records):
        raise ValueError("selected Split references unknown candidate Mid clusters")

    source_members = set(source.member_segment_ids)
    candidate_members = {
        candidate_id: set(candidate_records[candidate_id]["member_segment_ids"])
        for candidate_id in candidate_ids
    }
    wanted_ids = source_members | set().union(*candidate_members.values())
    records, vectors, all_trajectory_ids = _load_records(options.preprocessed, wanted_ids)
    historical_trajectory_ids = set(base.training_trajectory_ids)

    child_members = {}
    for candidate_id in candidate_ids:
        child_members[candidate_id] = {
            segment_id
            for segment_id in candidate_members[candidate_id]
            if segment_id in source_members
            or segment_id.rsplit(":segment:", 1)[0] not in historical_trajectory_ids
        }
    residual_ids = source_members - set().union(*child_members.values())
    candidate_centroids = {
        candidate_id: np.asarray(candidate_records[candidate_id]["centroid"], dtype=np.float32)
        for candidate_id in candidate_ids
    }
    for segment_id in sorted(residual_ids):
        child_id = max(
            candidate_ids,
            key=lambda candidate_id: (
                _cosine(vectors[segment_id], candidate_centroids[candidate_id]),
                candidate_id,
            ),
        )
        child_members[child_id].add(segment_id)
    if set().union(*child_members.values()) & source_members != source_members:
        raise ValueError("shadow Split does not cover all source historical members")
    if set.intersection(*(set(members) for members in child_members.values())):
        raise ValueError("shadow Split children overlap")

    children = []
    candidate_to_child = {}
    for candidate_id in candidate_ids:
        members = tuple(sorted(child_members[candidate_id]))
        digest = hashlib.sha256(
            (source_id + "\0" + "\0".join(members)).encode("utf-8")
        ).hexdigest()[:12]
        child_id = f"mid_split_{digest}"
        candidate_to_child[candidate_id] = child_id
        centroid = np.asarray([vectors[segment_id] for segment_id in members]).mean(axis=0)
        children.append(MidCluster(child_id, members, tuple(float(value) for value in centroid)))
    children = tuple(children)

    child_cards, usage, embedding = await _render_children(
        options, base, records, children
    )
    return _write_candidate(
        options,
        base,
        children,
        child_cards,
        usage,
        embedding,
        all_trajectory_ids,
        source_id,
        candidate_to_child,
        len(source_members),
        len(residual_ids),
    )


async def _render_children(options, base, records, children):
    child_segment_ids = {
        segment_id for child in children for segment_id in child.member_segment_ids
    }
    render_records = [
        {
            **record,
            "segments": [
                segment
                for segment in record["segments"]
                if segment["segment_id"] in child_segment_ids
            ],
        }
        for record in records
        if any(segment["segment_id"] in child_segment_ids for segment in record["segments"])
    ]
    render_inputs = build_mid_render_inputs(render_records, children)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=min(options.concurrency, common["global_api_concurrency"]),
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    try:
        rendered = await asyncio.gather(
            *(
                render_mid_card(
                    runtime,
                    base.benchmark,
                    render_input,
                    tuple(sibling for sibling in render_inputs if sibling is not render_input),
                )
                for render_input in render_inputs
            )
        )
        child_cards = tuple(card for card, _ in rendered)
        embedding = await runtime.embed([mid_card_text(card) for card in child_cards])
    finally:
        await runtime.close()
    return child_cards, rendered, embedding


def _write_candidate(
    options,
    base,
    children,
    child_cards,
    usage,
    embedding,
    all_trajectory_ids,
    source_id,
    candidate_to_child,
    source_member_count,
    residual_count,
) -> int:
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = output_dir / "clusters.json"
    high_paths_path = output_dir / "high-paths.json"
    cards_path = output_dir / "rendered-cards.json"
    index_path = output_dir / "retrieval-index.json"
    tower_path = output_dir / "tower.json"

    mid_clusters = (*base.mid_clusters, *children)
    mid_cards = (*base.mid_cards, *child_cards)
    mid_index = SkillEmbeddingIndex(
        (*base.mid_index.skill_ids, *(card.skill_id for card in child_cards)),
        (*base.mid_index.vectors, *embedding.vectors),
        (
            *base.mid_index.text_hashes,
            *(
                hashlib.sha256(mid_card_text(card).encode("utf-8")).hexdigest()
                for card in child_cards
            ),
        ),
    )
    write_json(clusters_path, {"clusters": [cluster.to_record() for cluster in mid_clusters]})
    write_json(high_paths_path, {"paths": [path.to_record() for path in base.high_paths]})
    write_json(
        cards_path,
        {
            "mid_cards": [card.to_record() for card in mid_cards],
            "high_cards": [card.to_record() for card in base.high_cards],
            "usage": [
                {"skill_id": card.skill_id, **asdict(result.usage), "latency_ms": result.latency_ms}
                for card, result in usage
            ],
        },
    )
    write_json(
        index_path,
        {
            "mid_index": mid_index.to_record(),
            "high_index": base.high_index.to_record(),
        },
    )
    snapshot = build_tower_snapshot(
        version=TowerVersion.V1,
        benchmark=base.benchmark,
        config=base.config,
        training_trajectory_ids=tuple(all_trajectory_ids),
        source_hashes=TowerSourceHashes(
            preprocessed_trajectories=sha256_file(options.preprocessed),
            clusters=sha256_file(clusters_path),
            high_paths=sha256_file(high_paths_path),
            rendered_cards=sha256_file(cards_path),
            retrieval_index=sha256_file(index_path),
        ),
        low_skills=base.low_skills,
        mid_clusters=mid_clusters,
        high_paths=base.high_paths,
        mid_cards=mid_cards,
        high_cards=base.high_cards,
        mid_index=mid_index,
        high_index=base.high_index,
    )
    snapshot.require_complete()
    write_json(tower_path, snapshot.to_record())
    report = {
        "protocol_id": "alfworld-deployment-optimization-v1-shadow-split",
        "candidate_type": "shadow_split",
        "snapshot_id": snapshot.snapshot_id,
        "base_snapshot_id": base.snapshot_id,
        "source_mid_id": source_id,
        "source_retained_for_high_compatibility": True,
        "candidate_to_child_mid": candidate_to_child,
        "historical_source_member_count": source_member_count,
        "residual_nearest_centroid_assignments": residual_count,
        "child_member_counts": {
            child.cluster_id: len(child.member_segment_ids) for child in children
        },
        "training_trajectory_count": len(all_trajectory_ids),
        "mid_count": len(mid_cards),
        "high_count": len(base.high_cards),
        "tower_path": tower_path.as_posix(),
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def _load_records(
    path: Path,
    wanted_ids: set[str],
) -> tuple[list[dict], dict[str, np.ndarray], tuple[str, ...]]:
    records = []
    vectors = {}
    trajectory_ids = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            trajectory_ids.append(str(record["trajectory_id"]))
            for segment in record["segments"]:
                segment_id = str(segment["segment_id"])
                if segment_id in wanted_ids:
                    vectors[segment_id] = np.asarray(segment["embedding"], dtype=np.float32)
                segment["embedding"] = ()
            records.append(record)
    if set(vectors) != wanted_ids:
        raise ValueError("preprocessed input does not cover Split evidence")
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("refinement input contains duplicate trajectory IDs")
    return records, vectors, tuple(trajectory_ids)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else -1.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--candidate-clusters", type=Path, required=True)
    parser.add_argument("--structural-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
