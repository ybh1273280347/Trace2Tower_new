from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.methods.flat_skill_summary.models import (
    FlatSkillCard,
    build_flat_library,
)
from trace2tower.results import MethodName
from trace2tower.semantic_index import SkillEmbeddingIndex


def matrix_module():
    scripts_path = str(Path("scripts/experiments").resolve())
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    import run_matrix

    return run_matrix


def flat_library(benchmark: Benchmark):
    card = FlatSkillCard(
        "flat_one",
        f"{benchmark}:train:no_skill:sample:0",
        "Skill",
        "Use when relevant.",
        ("Act.",),
        ("Check.",),
    )
    return build_flat_library(
        benchmark,
        "a" * 64,
        (card,),
        SkillEmbeddingIndex((card.skill_id,), ((1.0, 0.0),), ("b" * 64,)),
    )


def test_shard_parser_is_deterministic_and_validated() -> None:
    parse_shard_ids = matrix_module().parse_shard_ids
    assert parse_shard_ids("2,0,2", 3) == (0, 2)
    assert parse_shard_ids("all", 3) == (0, 1, 2)
    with pytest.raises(ValueError, match="shard IDs"):
        parse_shard_ids("3", 3)


def test_method_artifact_binds_content_id_hash_and_benchmark(tmp_path: Path) -> None:
    load_method_artifact = matrix_module().load_method_artifact
    library = flat_library(Benchmark.WEBSHOP)
    path = tmp_path / "library.json"
    path.write_text(json.dumps(library.to_record()), encoding="utf-8")
    artifact = load_method_artifact(
        Benchmark.WEBSHOP,
        MethodName.FLAT_SKILL_SUMMARY,
        path,
    )
    assert artifact.artifact_id == library.library_id
    assert len(artifact.sha256) == 64
    with pytest.raises(ValueError, match="benchmark"):
        load_method_artifact(
            Benchmark.ALFWORLD,
            MethodName.FLAT_SKILL_SUMMARY,
            path,
        )


def test_sample_selection_requires_every_requested_manifest_id() -> None:
    select_entries = matrix_module().select_entries
    entries = [
        ManifestEntry(
            Benchmark.WEBSHOP,
            ExperimentSplit.TRAIN,
            f"webshop:{index}",
            index,
            "goals",
            0,
        )
        for index in range(2)
    ]
    assert [entry.sample_id for entry in select_entries(entries, ("webshop:1",))] == [
        "webshop:1"
    ]
    with pytest.raises(ValueError, match="absent"):
        select_entries(entries, ("webshop:3",))
