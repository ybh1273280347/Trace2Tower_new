from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json

from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.models import HighPath, MidCluster
from trace2tower.methods.trace2tower.retrieval import SkillEmbeddingIndex
from trace2tower.methods.trace2tower.skills import LOW_SKILLS, HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.tower import (
    TowerSourceHashes,
    TowerVersion,
    build_tower_snapshot,
    sha256_file,
)


def main(options: argparse.Namespace) -> int:
    records = [
        json.loads(line)
        for line in options.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(record.get("split") != ExperimentSplit.TRAIN for record in records):
        raise ValueError("Tower snapshots may only use explicit training trajectories")
    trajectory_methods = {record.get("trajectory_method") for record in records}
    if options.version is TowerVersion.V0 and trajectory_methods != {"no_skill"}:
        raise ValueError("Tower v0 requires shared No-Skill trajectories")
    if options.version is TowerVersion.V1 and not trajectory_methods <= {
        "no_skill",
        "trace2tower",
    }:
        raise ValueError("Tower v1 accepts only No-Skill and prior-Tower feedback")
    if any(record["benchmark"] != options.benchmark for record in records):
        raise ValueError("preprocessed benchmark does not match snapshot benchmark")
    cluster_payload = json.loads(options.clusters.read_text(encoding="utf-8"))
    path_payload = json.loads(options.high_paths.read_text(encoding="utf-8"))
    card_payload = json.loads(options.cards.read_text(encoding="utf-8"))
    index_payload = json.loads(options.index.read_text(encoding="utf-8"))
    snapshot = build_tower_snapshot(
        version=options.version,
        benchmark=options.benchmark,
        config=Trace2TowerConfig.from_record(load_yaml(options.config)),
        training_trajectory_ids=tuple(record["trajectory_id"] for record in records),
        source_hashes=TowerSourceHashes(
            preprocessed_trajectories=sha256_file(options.input),
            clusters=sha256_file(options.clusters),
            high_paths=sha256_file(options.high_paths),
            rendered_cards=sha256_file(options.cards),
            retrieval_index=sha256_file(options.index),
        ),
        low_skills=LOW_SKILLS[options.benchmark],
        mid_clusters=tuple(
            MidCluster.from_record(item) for item in cluster_payload["clusters"]
        ),
        high_paths=tuple(
            HighPath.from_record(item) for item in path_payload["paths"]
        ),
        mid_cards=tuple(
            MidSkillCard.from_record(item) for item in card_payload["mid_cards"]
        ),
        high_cards=tuple(
            HighSkillCard.from_record(item) for item in card_payload["high_cards"]
        ),
        mid_index=SkillEmbeddingIndex.from_record(index_payload["mid_index"]),
        high_index=SkillEmbeddingIndex.from_record(index_payload["high_index"]),
    )
    snapshot.require_complete()
    write_json(options.output, snapshot.to_record())
    report = {
        "snapshot_id": snapshot.snapshot_id,
        "benchmark": snapshot.benchmark.value,
        "version": snapshot.version.value,
        "training_trajectory_count": len(snapshot.training_trajectory_ids),
        "mid_count": len(snapshot.mid_cards),
        "high_count": len(snapshot.high_cards),
        "mid_coverage_complete": snapshot.mid_coverage_complete,
        "high_coverage_complete": snapshot.high_coverage_complete,
    }
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument(
        "--version", type=TowerVersion, choices=tuple(TowerVersion), default=TowerVersion.V0
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--high-paths", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
