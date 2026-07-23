from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.run import run_matrix
from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.core.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.core.results import MethodName
from trace2tower.methods.skillx.models import build_execution_library
from trace2tower.methods.skillx.native_inference import SKILLX_COMMIT


def matrix_module():
    return run_matrix


def skillx_library(benchmark: Benchmark):
    return build_execution_library(
        benchmark,
        "a" * 64,
        SKILLX_COMMIT,
        (),
        (),
        SkillEmbeddingIndex((), ()),
        SkillEmbeddingIndex((), ()),
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
    library = skillx_library(Benchmark.WEBSHOP)
    path = tmp_path / "library.json"
    path.write_text(json.dumps(library.to_record()), encoding="utf-8")
    artifact = load_method_artifact(
        Benchmark.WEBSHOP,
        MethodName.SKILLX,
        path,
    )
    assert artifact.artifact_id == library.library_id
    assert len(artifact.sha256) == 64
    with pytest.raises(ValueError, match="benchmark"):
        load_method_artifact(
            Benchmark.ALFWORLD,
            MethodName.SKILLX,
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
    assert [entry.sample_id for entry in select_entries(entries, ("webshop:1",))] == ["webshop:1"]
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


def test_alfworld_server_override_is_included_in_resolved_config() -> None:
    configs = {
        Benchmark.ALFWORLD: {"server_url": "http://127.0.0.1:18080"},
        Benchmark.WEBSHOP: {"max_steps": 15},
    }

    resolved = matrix_module().apply_benchmark_overrides(
        configs,
        alfworld_server_url="http://127.0.0.1:18081",
    )

    assert resolved[Benchmark.ALFWORLD]["server_url"] == "http://127.0.0.1:18081"
    assert configs[Benchmark.ALFWORLD]["server_url"] == "http://127.0.0.1:18080"
    assert resolved[Benchmark.WEBSHOP] == configs[Benchmark.WEBSHOP]


def test_concurrency_overrides_are_included_in_resolved_config() -> None:
    common = {"episode_concurrency": 50, "global_api_concurrency": 50}

    resolved = matrix_module().apply_common_overrides(
        common,
        episode_concurrency=12,
        api_concurrency=40,
    )

    assert resolved["episode_concurrency"] == 12
    assert resolved["global_api_concurrency"] == 40
    assert common == {"episode_concurrency": 50, "global_api_concurrency": 50}
