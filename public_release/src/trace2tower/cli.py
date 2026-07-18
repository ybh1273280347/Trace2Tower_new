from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from trace2tower.retrieval.index import SkillEmbeddingIndex
from trace2tower.core.manifests import Benchmark
from trace2tower.artifacts.tower import (
    TowerSourceHashes,
    TowerVersion,
    build_tower_snapshot,
    sha256_file,
)
from trace2tower.config.tower import Trace2TowerConfig
from trace2tower.core.tower_models import HighCommunity, HighPath, MidCluster
from trace2tower.skills.induction import LOW_SKILLS, HighSkillCard, MidSkillCard


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="trace2tower",
        description="Build validated Tower snapshots and export reusable skill cards.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build-tower")
    _add_build_arguments(build_parser)
    build_parser.set_defaults(handler=_build_tower)
    export_parser = commands.add_parser("extract-skills")
    export_parser.add_argument("--tower", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.set_defaults(handler=_extract_skills)
    options = parser.parse_args()
    return options.handler(options)


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--version", type=TowerVersion, choices=tuple(TowerVersion), default=TowerVersion.V0)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--high-paths", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _build_tower(options: argparse.Namespace) -> int:
    provenance = _load_provenance(options.provenance)
    if any(item["benchmark"] != options.benchmark.value for item in provenance):
        raise ValueError("provenance mixes benchmarks")
    trajectory_ids = tuple(item["trajectory_id"] for item in provenance)
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("provenance contains duplicate trajectory IDs")
    clusters = json.loads(options.clusters.read_text(encoding="utf-8"))
    high_paths = json.loads(options.high_paths.read_text(encoding="utf-8"))
    cards = json.loads(options.cards.read_text(encoding="utf-8"))
    indexes = json.loads(options.index.read_text(encoding="utf-8"))
    snapshot = build_tower_snapshot(
        version=options.version,
        benchmark=options.benchmark,
        config=Trace2TowerConfig.from_record(_load_yaml(options.config)),
        training_trajectory_ids=trajectory_ids,
        source_hashes=TowerSourceHashes(
            preprocessed_trajectories=sha256_file(options.provenance),
            clusters=sha256_file(options.clusters),
            high_paths=sha256_file(options.high_paths),
            rendered_cards=sha256_file(options.cards),
            retrieval_index=sha256_file(options.index),
        ),
        low_skills=LOW_SKILLS[options.benchmark],
        mid_clusters=tuple(MidCluster.from_record(item) for item in clusters["clusters"]),
        high_paths=tuple(HighPath.from_record(item) for item in high_paths["paths"]),
        mid_cards=tuple(MidSkillCard.from_record(item) for item in cards["mid_cards"]),
        high_cards=tuple(HighSkillCard.from_record(item) for item in cards["high_cards"]),
        mid_index=SkillEmbeddingIndex.from_record(indexes["mid_index"]),
        high_index=SkillEmbeddingIndex.from_record(indexes["high_index"]),
        high_communities=tuple(
            HighCommunity.from_record(item) for item in high_paths.get("communities", ())
        ),
    )
    snapshot.require_complete()
    _write_json(options.output, snapshot.to_record())
    print(json.dumps({"snapshot_id": snapshot.snapshot_id, "output": str(options.output)}, indent=2))
    return 0


def _extract_skills(options: argparse.Namespace) -> int:
    record = json.loads(options.tower.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "trace2tower.skill_snapshot.v1",
        "benchmark": record["benchmark"],
        "tower_snapshot_id": record["snapshot_id"],
        "low_skills": record.get("low_skills", ()),
        "mid_skills": [_public_mid_card(card) for card in record.get("mid_cards", ())],
        "high_skills": [_public_high_card(card) for card in record.get("high_cards", ())],
    }
    _write_json(options.output, payload)
    print(json.dumps({"benchmark": payload["benchmark"], "output": str(options.output)}, indent=2))
    return 0


def _public_mid_card(card: dict) -> dict:
    return {
        key: card[key]
        for key in ("skill_id", "name", "description", "procedure", "constraints", "grounding_actions")
        if key in card
    }


def _public_high_card(card: dict) -> dict:
    return {
        key: card[key]
        for key in (
            "skill_id",
            "ordered_mid_ids",
            "member_mid_ids",
            "name",
            "description",
            "procedure",
            "constraints",
            "retrieval_condition",
        )
        if key in card
    }


def _load_provenance(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
