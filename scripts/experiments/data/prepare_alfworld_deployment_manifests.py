from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.core.manifests import ManifestEntry, read_manifest, write_manifest

PARTITION_NAMES = ("deployment_feedback", "deployment_gate", "deployment_holdout")
PILOT_NAME = "deployment_feedback_pilot"


def partition_train_tasks(
    candidates: list[tuple[ManifestEntry, str]],
    counts: tuple[int, int, int],
    *,
    seed: int,
) -> dict[str, list[tuple[ManifestEntry, str]]]:
    if any(count <= 0 for count in counts) or sum(counts) != len(candidates):
        raise ValueError("partition counts must be positive and cover every candidate")
    shuffled = sorted(candidates, key=lambda item: item[0].sample_id)
    randomizer = random.Random(seed)
    randomizer.shuffle(shuffled)
    partitions = {}
    offset = 0
    for partition_name, target_count in zip(PARTITION_NAMES, counts):
        selected = shuffled[offset : offset + target_count]
        partitions[partition_name] = sorted(selected, key=lambda item: item[0].sample_id)
        offset += target_count
    if [len(partitions[name]) for name in PARTITION_NAMES] != list(counts):
        raise RuntimeError("random partitioning produced incorrect counts")
    return partitions


def random_subset(
    candidates: list[tuple[ManifestEntry, str]],
    count: int,
    *,
    seed: int,
) -> list[tuple[ManifestEntry, str]]:
    if not 0 < count <= len(candidates):
        raise ValueError("subset count must be positive and no larger than its source")
    shuffled = sorted(candidates, key=lambda item: item[0].sample_id)
    randomizer = random.Random(seed)
    randomizer.shuffle(shuffled)
    return sorted(shuffled[:count], key=lambda item: item[0].sample_id)


def main(options: argparse.Namespace) -> int:
    source_entries = read_manifest(options.source_manifest)
    construction_ids = {
        json.loads(line)["sample_id"]
        for line in options.construction_pool.read_text(encoding="utf-8").splitlines()
        if line
    }
    family_by_id = {
        record["sample_id"]: record["task_family"]
        for line in options.family_manifest.read_text(encoding="utf-8").splitlines()
        if line
        for record in (json.loads(line),)
    }
    source_by_id = {entry.sample_id: entry for entry in source_entries}
    if len(source_by_id) != len(source_entries):
        raise ValueError("source manifest must contain one row per task")
    if not construction_ids <= set(source_by_id):
        raise ValueError("construction pool contains tasks outside the source manifest")
    candidate_ids = sorted(set(source_by_id) - construction_ids)
    missing_families = set(candidate_ids) - set(family_by_id)
    if missing_families:
        raise ValueError(
            f"deployment candidates lack task families: {sorted(missing_families)[:3]}"
        )
    candidates = [(source_by_id[sample_id], family_by_id[sample_id]) for sample_id in candidate_ids]
    counts = (options.feedback_count, options.gate_count, options.holdout_count)
    partitions = partition_train_tasks(candidates, counts, seed=options.seed)
    pilot = random_subset(
        partitions["deployment_feedback"],
        options.pilot_count,
        seed=options.seed + 1,
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    family_counts = {}
    partition_ids = {}
    for partition_name in PARTITION_NAMES:
        selected = partitions[partition_name]
        manifest_path = options.output_dir / f"{partition_name}.jsonl"
        write_manifest(manifest_path, (entry for entry, _ in selected))
        manifest_paths[partition_name] = manifest_path
        partition_ids[partition_name] = {entry.sample_id for entry, _ in selected}
        counts_by_family = defaultdict(int)
        for _, family in selected:
            counts_by_family[family] += 1
        family_counts[partition_name] = dict(sorted(counts_by_family.items()))

    pilot_path = options.output_dir / f"{PILOT_NAME}.jsonl"
    write_manifest(pilot_path, (entry for entry, _ in pilot))
    manifest_paths[PILOT_NAME] = pilot_path
    family_counts[PILOT_NAME] = dict(
        sorted(
            (family, sum(candidate_family == family for _, candidate_family in pilot))
            for family in {family for _, family in pilot}
        )
    )

    intersections = {
        "feedback_gate": len(
            partition_ids["deployment_feedback"] & partition_ids["deployment_gate"]
        ),
        "feedback_holdout": len(
            partition_ids["deployment_feedback"] & partition_ids["deployment_holdout"]
        ),
        "gate_holdout": len(partition_ids["deployment_gate"] & partition_ids["deployment_holdout"]),
        "construction_candidates": len(construction_ids & set().union(*partition_ids.values())),
    }
    if any(intersections.values()):
        raise RuntimeError(f"deployment manifest partitions overlap: {intersections}")
    audit = {
        "protocol_id": "alfworld-deployment-optimization-v1",
        "benchmark": "alfworld",
        "source_split": "train",
        "selection_seed": options.seed,
        "construction_task_count": len(construction_ids),
        "candidate_task_count": len(candidates),
        "repeat_ids": [0],
        "pilot_is_feedback_subset": {entry.sample_id for entry, _ in pilot}
        <= partition_ids["deployment_feedback"],
        "partitions": {
            partition_name: {
                "path": manifest_paths[partition_name].as_posix(),
                "sha256": hashlib.sha256(manifest_paths[partition_name].read_bytes()).hexdigest(),
                "task_count": (
                    len(pilot) if partition_name == PILOT_NAME else len(partitions[partition_name])
                ),
                "task_family_counts": family_counts[partition_name],
            }
            for partition_name in (*PARTITION_NAMES, PILOT_NAME)
        },
        "intersections": intersections,
        "source_manifest": options.source_manifest.as_posix(),
        "source_manifest_sha256": hashlib.sha256(options.source_manifest.read_bytes()).hexdigest(),
        "construction_pool": options.construction_pool.as_posix(),
        "construction_pool_sha256": hashlib.sha256(
            options.construction_pool.read_bytes()
        ).hexdigest(),
    }
    write_json(options.output_dir / "manifest-audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/manifests/alfworld_train_p1000.jsonl"),
    )
    parser.add_argument(
        "--construction-pool",
        type=Path,
        default=Path("artifacts/trajectories/alfworld/alfworld-pool-v1-pro-expanded.jsonl"),
    )
    parser.add_argument(
        "--family-manifest",
        type=Path,
        default=Path("artifacts/manifests/alfworld_train.jsonl"),
    )
    parser.add_argument("--feedback-count", type=int, default=450)
    parser.add_argument("--pilot-count", type=int, default=60)
    parser.add_argument("--gate-count", type=int, default=120)
    parser.add_argument("--holdout-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/trace2tower/alfworld/deployment-optimization-v1/manifests"),
    )
    raise SystemExit(main(parser.parse_args()))
