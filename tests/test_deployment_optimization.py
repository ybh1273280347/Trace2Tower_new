from __future__ import annotations

from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import EpisodeResult, FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory, write_trajectory_jsonl
from trace2tower.methods.trace2tower.core.models import MidCluster
from trace2tower.methods.trace2tower.deployment_optimization.feedback import (
    bootstrap_pareto,
    bundle_metrics,
    pair_feedback,
    read_no_skill_trajectories,
)
from trace2tower.methods.trace2tower.deployment_optimization.lineage import build_mid_lineage
from trace2tower.methods.trace2tower.deployment_optimization.models import LineageKind
from trace2tower.methods.trace2tower.deployment_optimization.pareto import rank_fronts


def result(
    method: MethodName,
    sample_id: str,
    repeat_id: int,
    *,
    success: bool,
    steps: int,
    skill_ids: tuple[str, ...] = (),
) -> EpisodeResult:
    return EpisodeResult(
        run_id=f"run-{method}",
        benchmark=Benchmark.ALFWORLD,
        split=ExperimentSplit.TRAIN,
        method=method,
        sample_id=sample_id,
        repeat_id=repeat_id,
        shard_id=0,
        primary_score=float(success),
        success=success,
        steps=steps,
        invalid_actions=0,
        finish_reason=FinishReason.COMPLETED if success else FinishReason.TASK_LIMIT_REACHED,
        input_tokens=0,
        output_tokens=0,
        billable_tokens=0,
        latency_ms=0,
        skill_ids=skill_ids,
        context_skill_ids=skill_ids,
        skill_context_chars=0,
    )


def cluster(cluster_id: str, *member_segment_ids: str) -> MidCluster:
    return MidCluster(cluster_id, member_segment_ids, ())


def test_feedback_uses_primary_high_and_does_not_reward_faster_failure() -> None:
    no_skill = (
        result(MethodName.NO_SKILL, "task-a", 0, success=True, steps=10),
        result(MethodName.NO_SKILL, "task-b", 0, success=False, steps=10),
    )
    tower = (
        result(
            MethodName.TRACE2TOWER,
            "task-a",
            0,
            success=False,
            steps=5,
            skill_ids=("high_primary", "high_evidence", "mid_0001"),
        ),
        result(
            MethodName.TRACE2TOWER,
            "task-b",
            0,
            success=True,
            steps=6,
            skill_ids=("high_primary", "high_other", "mid_0002"),
        ),
    )

    pairs = pair_feedback(no_skill, tower)
    metrics = bundle_metrics(pairs)

    assert {pair.primary_high_id for pair in pairs} == {"high_primary"}
    assert metrics[0].exposure_count == 2
    assert metrics[0].objectives.performance_level == 0.5
    assert metrics[0].objectives.paired_success_gain == 0.0
    assert metrics[0].objectives.guarded_step_saving == 0.2


def test_bootstrap_pareto_is_task_clustered_and_deterministic() -> None:
    no_skill = tuple(
        result(MethodName.NO_SKILL, f"task-{index}", repeat, success=False, steps=10)
        for index in range(4)
        for repeat in (0, 1)
    )
    tower = tuple(
        result(
            MethodName.TRACE2TOWER,
            f"task-{index}",
            repeat,
            success=index < 2,
            steps=6 if index < 2 else 12,
            skill_ids=(("high_good",) if index < 2 else ("high_bad",)),
        )
        for index in range(4)
        for repeat in (0, 1)
    )
    pairs = pair_feedback(no_skill, tower)

    first = bootstrap_pareto(pairs, samples=100, seed=7)
    second = bootstrap_pareto(pairs, samples=100, seed=7)

    assert first == second
    estimates = {item.metrics.primary_high_id: item for item in first}
    assert estimates["high_good"].pareto_front_rank == 1
    assert estimates["high_bad"].pareto_front_rank == 2
    assert estimates["high_good"].front_1_probability > 0.9


def test_no_skill_pool_is_filtered_to_manifest_and_repeat_zero(tmp_path) -> None:
    trajectories = tuple(
        EpisodeTrajectory(
            run_id="pool",
            benchmark=Benchmark.ALFWORLD,
            split=ExperimentSplit.TRAIN,
            method=MethodName.NO_SKILL,
            sample_id=sample_id,
            repeat_id=repeat_id,
            task_goal="goal",
            steps=(),
            primary_score=0.0,
            finish_reason=FinishReason.TASK_LIMIT_REACHED,
        )
        for sample_id in ("task-a", "task-b")
        for repeat_id in (0, 1)
    )
    pool_path = tmp_path / "pool.jsonl"
    write_trajectory_jsonl(trajectories, pool_path)

    selected = read_no_skill_trajectories(
        pool_path,
        sample_ids=frozenset({"task-a"}),
        repeat_ids=frozenset({0}),
    )

    assert [(episode.sample_id, episode.repeat_id) for episode in selected] == [("task-a", 0)]


def test_pareto_ranking_preserves_non_dominated_tradeoffs() -> None:
    ranks = rank_fronts(
        {
            "reliable": (1.0, 0.1, 0.0),
            "efficient": (0.9, 0.1, 0.5),
            "dominated": (0.8, 0.0, -0.1),
        }
    )

    assert ranks == {"reliable": 1, "efficient": 1, "dominated": 2}


def test_mid_lineage_classifies_split_merge_and_new_structure() -> None:
    old = (
        cluster("old_split", "a", "b"),
        cluster("old_merge_a", "c"),
        cluster("old_merge_b", "d"),
        cluster("old_gone", "e"),
    )
    new = (
        cluster("new_split_a", "a"),
        cluster("new_split_b", "b"),
        cluster("new_merge", "c", "d"),
        cluster("new_only", "f"),
    )

    lineage = build_mid_lineage(old, new)
    by_kind = {component.kind: component for component in lineage}

    assert by_kind[LineageKind.SPLIT].old_mid_ids == ("old_split",)
    assert by_kind[LineageKind.SPLIT].new_mid_ids == ("new_split_a", "new_split_b")
    assert by_kind[LineageKind.MERGE].old_mid_ids == ("old_merge_a", "old_merge_b")
    assert by_kind[LineageKind.MERGE].new_mid_ids == ("new_merge",)
    assert by_kind[LineageKind.NEW_MID].new_mid_ids == ("new_only",)
    assert by_kind[LineageKind.DISAPPEARED_MID].old_mid_ids == ("old_gone",)
