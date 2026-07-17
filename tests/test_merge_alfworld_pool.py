from __future__ import annotations

from dataclasses import replace

from scripts.experiments.data.merge_alfworld_pool import deduplicate_trajectories
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory


def trajectory(
    sample_id: str,
    repeat_id: int,
    *,
    score: float = 1.0,
) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        run_id="source",
        benchmark=Benchmark.ALFWORLD,
        split=ExperimentSplit.TRAIN,
        method=MethodName.NO_SKILL,
        sample_id=sample_id,
        repeat_id=repeat_id,
        task_goal=f"goal for {sample_id}",
        steps=(),
        primary_score=score,
        finish_reason=FinishReason.COMPLETED,
    )


def test_deduplicate_trajectories_keeps_first_key_and_audits_disagreements() -> None:
    first = trajectory("a", 0)
    duplicate = replace(first, primary_score=0.0)
    other = trajectory("b", 1)

    selected, audit = deduplicate_trajectories((first, duplicate, other))

    assert selected == (first, other)
    assert audit == {
        "raw_trajectory_count": 3,
        "selected_trajectory_count": 2,
        "duplicate_key_count": 1,
        "discarded_trajectory_count": 1,
        "score_disagreement_count": 1,
        "step_disagreement_count": 0,
    }
