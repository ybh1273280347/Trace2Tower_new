from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.run import run_matrix
from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.methods.global_e2e.models import (
    GlobalE2ESkillCard,
    build_global_e2e_library,
)
from trace2tower.results import MethodName
from trace2tower.semantic_index import SkillEmbeddingIndex


def matrix_module():
    return run_matrix


def global_e2e_library(benchmark: Benchmark):
    trajectory_id = f"{benchmark}:train:no_skill:sample:0"
    card = GlobalE2ESkillCard(
        "global_one",
        (trajectory_id,),
        "Skill",
        "Use when relevant.",
        ("Act.",),
        ("Check.",),
    )
    return build_global_e2e_library(
        benchmark,
        "a" * 64,
        "b" * 64,
        (trajectory_id,),
        (card,),
        SkillEmbeddingIndex((card.skill_id,), ((1.0, 0.0),), ("c" * 64,)),
    )


def test_shard_parser_is_deterministic_and_validated() -> None:
    parse_shard_ids = matrix_module().parse_shard_ids
    assert parse_shard_ids("2,0,2", 3) == (0, 2)
    assert parse_shard_ids("all", 3) == (0, 1, 2)
    with pytest.raises(ValueError, match="shard IDs"):
        parse_shard_ids("3", 3)


def test_benchmark_path_assignments_are_typed_and_unique() -> None:
    parse_paths = matrix_module().parse_benchmark_paths
    assert parse_paths(["webshop=manifest.jsonl"], "manifest") == {
        Benchmark.WEBSHOP: Path("manifest.jsonl")
    }
    with pytest.raises(ValueError, match="duplicate manifest"):
        parse_paths(
            ["webshop=first.jsonl", "webshop=second.jsonl"],
            "manifest",
        )


def test_method_artifact_binds_content_id_hash_and_benchmark(tmp_path: Path) -> None:
    load_method_artifact = matrix_module().load_method_artifact
    library = global_e2e_library(Benchmark.WEBSHOP)
    path = tmp_path / "library.json"
    path.write_text(json.dumps(library.to_record()), encoding="utf-8")
    artifact = load_method_artifact(
        Benchmark.WEBSHOP,
        MethodName.GLOBAL_E2E_GPT,
        path,
    )
    assert artifact.artifact_id == library.library_id
    assert len(artifact.sha256) == 64
    with pytest.raises(ValueError, match="benchmark"):
        load_method_artifact(
            Benchmark.ALFWORLD,
            MethodName.GLOBAL_E2E_GPT,
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


def test_sample_selection_expands_explicit_repeat_ids() -> None:
    entries = [
        ManifestEntry(
            Benchmark.WEBSHOP,
            ExperimentSplit.TRAIN,
            "webshop:1",
            1,
            "goals",
            0,
        )
    ]
    selected = matrix_module().select_entries(
        entries,
        ("webshop:1",),
        (0, 1, 2),
    )
    assert [entry.repeat_id for entry in selected] == [0, 1, 2]
