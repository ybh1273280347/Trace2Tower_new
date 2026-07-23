from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.data.audit_webshop_training_pools import audit_pool
from trace2tower.benchmarks.models import ClickableKind
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory, StepRecord, write_trajectory_jsonl


def trajectory(sample_index: int, repeat_id: int) -> EpisodeTrajectory:
    score = 1.0 if repeat_id == 0 else 0.5
    return EpisodeTrajectory(
        run_id="source-run",
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TRAIN,
        method=MethodName.NO_SKILL,
        sample_id=f"webshop:{sample_index}",
        repeat_id=repeat_id,
        task_goal="find a product",
        steps=(
            StepRecord(
                step_index=0,
                observation="search",
                action_name="search_action",
                action_arguments={"keywords": "product"},
                next_observation="done",
                reward=score,
                done=True,
                valid_action=True,
                admissible_actions=(),
                clickable_types={"Buy Now": ClickableKind.BUTTON},
            ),
        ),
        primary_score=score,
        finish_reason=FinishReason.COMPLETED,
    )


def write_source_run(runs_root: Path, *, error_attempt_count: int = 0) -> None:
    run_root = runs_root / "source-run"
    run_root.mkdir(parents=True)
    (run_root / "resolved-config.yaml").write_text(
        "agent_model: deepseek-v4-flash\n",
        encoding="utf-8",
    )
    metadata = {
        "run_id": "source-run",
        "benchmarks": ["webshop"],
        "split": "train",
        "method": "no_skill",
        "agent_model": "deepseek-v4-flash",
        "repeat_ids": [0, 1, 2, 3],
        "shards": [
            {
                "official_result_count": 8,
                "trajectory_count": 8,
                "error_attempt_count": error_attempt_count,
                "invocation_summary": {"failed": 0},
            }
        ],
    }
    (run_root / "matrix-metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_pool_audit_checks_cartesian_coverage_and_source_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_source_run(runs_root)
    pool_path = tmp_path / "pool.jsonl"
    write_trajectory_jsonl(
        (
            trajectory(sample_index, repeat_id)
            for sample_index in (1000, 1001)
            for repeat_id in range(4)
        ),
        pool_path,
    )

    report, records = audit_pool("test", pool_path, 2, runs_root)

    assert report["task_count"] == 2
    assert report["trajectory_count"] == 8
    assert report["reward"]["full_success_count"] == 2
    assert report["reward"]["positive_partial_count"] == 6
    assert report["source_runs"][0]["error_attempt_count"] == 0
    assert len(records) == 8


def test_pool_audit_rejects_source_errors(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_source_run(runs_root, error_attempt_count=1)
    pool_path = tmp_path / "pool.jsonl"
    write_trajectory_jsonl(
        (
            trajectory(sample_index, repeat_id)
            for sample_index in (1000, 1001)
            for repeat_id in range(4)
        ),
        pool_path,
    )

    with pytest.raises(ValueError, match="unresolved errors"):
        audit_pool("test", pool_path, 2, runs_root)
