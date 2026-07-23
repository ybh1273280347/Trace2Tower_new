from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.experiments.data.materialize_multirepeat_pool import (
    validate_multirepeat_pool,
)
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory


def trajectory(sample_id: str, repeat_id: int, run_id: str = "source") -> EpisodeTrajectory:
    return EpisodeTrajectory(
        run_id=run_id,
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TRAIN,
        method=MethodName.NO_SKILL,
        sample_id=sample_id,
        repeat_id=repeat_id,
        task_goal=f"goal for {sample_id}",
        steps=(),
        primary_score=1.0,
        finish_reason=FinishReason.COMPLETED,
    )


def test_multirepeat_pool_requires_exact_cartesian_coverage() -> None:
    source = (trajectory("a", 0), trajectory("b", 0))
    matrix = tuple(
        replace(trajectory(sample_id, repeat_id), run_id="matrix")
        for sample_id in ("a", "b")
        for repeat_id in (0, 1, 2, 3)
    )
    assert validate_multirepeat_pool(
        source,
        matrix,
        benchmark=Benchmark.WEBSHOP,
        run_id="matrix",
        repeat_ids=(0, 1, 2, 3),
    ) == ("a", "b")

    with pytest.raises(ValueError, match="coverage differs"):
        validate_multirepeat_pool(
            source,
            matrix[:-1],
            benchmark=Benchmark.WEBSHOP,
            run_id="matrix",
            repeat_ids=(0, 1, 2, 3),
        )
