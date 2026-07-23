from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import EpisodeResult, MethodName
from trace2tower.core.trajectory import TrajectoryReader
from trace2tower.methods.trace2tower.deployment_optimization.models import (
    BundleMetrics,
    BundleParetoEstimate,
    DeploymentObjectives,
    FeedbackSummary,
)
from trace2tower.methods.trace2tower.deployment_optimization.pareto import rank_fronts


@dataclass(frozen=True, slots=True)
class FeedbackPair:
    sample_id: str
    repeat_id: int
    primary_high_id: str
    context_skill_ids: tuple[str, ...]
    no_skill_success: bool
    tower_success: bool
    no_skill_steps: int
    tower_steps: int


@dataclass(frozen=True, slots=True)
class FeedbackEpisode:
    benchmark: Benchmark
    split: ExperimentSplit
    method: MethodName
    sample_id: str
    repeat_id: int
    success: bool
    steps: int
    skill_ids: tuple[str, ...] = ()
    context_skill_ids: tuple[str, ...] = ()


def read_results(root: Path) -> tuple[FeedbackEpisode, ...]:
    results = tuple(
        _from_result(EpisodeResult.from_record(json.loads(line)))
        for path in sorted(root.rglob("results.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    keys = [(result.sample_id, result.repeat_id) for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate episode results under {root}")
    return results


def read_no_skill_trajectories(
    path: Path,
    *,
    sample_ids: frozenset[str],
    repeat_ids: frozenset[int],
) -> tuple[FeedbackEpisode, ...]:
    if not sample_ids or not repeat_ids:
        raise ValueError("No-Skill trajectory selection cannot be empty")
    trajectories = tuple(
        trajectory
        for trajectory in TrajectoryReader.read_jsonl(path)
        if trajectory.sample_id in sample_ids and trajectory.repeat_id in repeat_ids
    )
    episodes = tuple(
        FeedbackEpisode(
            benchmark=trajectory.benchmark,
            split=trajectory.split,
            method=trajectory.method,
            sample_id=trajectory.sample_id,
            repeat_id=trajectory.repeat_id,
            success=bool(trajectory.primary_score),
            steps=len(trajectory.steps),
        )
        for trajectory in trajectories
    )
    keys = [(episode.sample_id, episode.repeat_id) for episode in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate No-Skill trajectories in {path}")
    expected_keys = {(sample_id, repeat_id) for sample_id in sample_ids for repeat_id in repeat_ids}
    if set(keys) != expected_keys:
        raise ValueError(
            "No-Skill trajectory pool does not cover the selected manifest: "
            f"expected={len(expected_keys)}, observed={len(keys)}"
        )
    return episodes


def pair_feedback(
    no_skill_results: Iterable[FeedbackEpisode],
    tower_results: Iterable[FeedbackEpisode],
) -> tuple[FeedbackPair, ...]:
    no_skill = _index_results(no_skill_results, MethodName.NO_SKILL)
    tower = _index_results(tower_results, MethodName.TRACE2TOWER)
    if set(no_skill) != set(tower):
        raise ValueError(
            f"feedback result keys differ: no_skill={len(no_skill)}, tower={len(tower)}"
        )

    pairs = []
    for key in sorted(no_skill):
        baseline = no_skill[key]
        candidate = tower[key]
        context_skill_ids = candidate.context_skill_ids or candidate.skill_ids
        high_ids = tuple(skill_id for skill_id in context_skill_ids if skill_id.startswith("high_"))
        if not high_ids:
            raise ValueError(f"Tower feedback has no identifiable High skill: {key}")
        pairs.append(
            FeedbackPair(
                sample_id=candidate.sample_id,
                repeat_id=candidate.repeat_id,
                primary_high_id=high_ids[0],
                context_skill_ids=context_skill_ids,
                no_skill_success=bool(baseline.success),
                tower_success=bool(candidate.success),
                no_skill_steps=baseline.steps,
                tower_steps=candidate.steps,
            )
        )
    return tuple(pairs)


def bundle_metrics(pairs: Iterable[FeedbackPair]) -> tuple[BundleMetrics, ...]:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair.primary_high_id].append(pair)
    return tuple(
        _measure_bundle(primary_high_id, grouped[primary_high_id])
        for primary_high_id in sorted(grouped)
    )


def feedback_summary(pairs: Iterable[FeedbackPair]) -> FeedbackSummary:
    selected = tuple(pairs)
    if not selected:
        raise ValueError("feedback contains no pairs")
    no_skill_success = np.asarray([pair.no_skill_success for pair in selected], dtype=np.float64)
    tower_success = np.asarray([pair.tower_success for pair in selected], dtype=np.float64)
    no_skill_steps = np.asarray([pair.no_skill_steps for pair in selected], dtype=np.float64)
    tower_steps = np.asarray([pair.tower_steps for pair in selected], dtype=np.float64)
    return FeedbackSummary(
        task_count=len(selected),
        no_skill_success_rate=float(no_skill_success.mean()),
        tower_success_rate=float(tower_success.mean()),
        paired_success_gain=float((tower_success - no_skill_success).mean()),
        paired_wins=sum(pair.tower_success > pair.no_skill_success for pair in selected),
        paired_losses=sum(pair.tower_success < pair.no_skill_success for pair in selected),
        paired_ties=sum(pair.tower_success == pair.no_skill_success for pair in selected),
        no_skill_mean_steps=float(no_skill_steps.mean()),
        tower_mean_steps=float(tower_steps.mean()),
        guarded_step_saving=float(np.mean([_guarded_step(pair) for pair in selected])),
    )


def bootstrap_pareto(
    pairs: Iterable[FeedbackPair],
    *,
    samples: int,
    seed: int,
    min_exposure: int = 1,
) -> tuple[BundleParetoEstimate, ...]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if min_exposure <= 0:
        raise ValueError("minimum exposure must be positive")
    selected = tuple(pairs)
    tasks = sorted({pair.sample_id for pair in selected})
    if not tasks:
        raise ValueError("feedback contains no tasks")
    pairs_by_task = {
        task: tuple(pair for pair in selected if pair.sample_id == task) for task in tasks
    }
    metrics = tuple(
        metric for metric in bundle_metrics(selected) if metric.exposure_count >= min_exposure
    )
    if not metrics:
        return ()
    eligible_ids = {metric.primary_high_id for metric in metrics}
    ranks = rank_fronts({metric.primary_high_id: metric.objectives.values for metric in metrics})
    front_counts = defaultdict(int)
    present_counts = defaultdict(int)
    random = np.random.default_rng(seed)

    for _ in range(samples):
        sampled_pairs = tuple(
            pair
            for task_index in random.integers(0, len(tasks), size=len(tasks))
            for pair in pairs_by_task[tasks[int(task_index)]]
        )
        sampled_metrics = tuple(
            metric
            for metric in bundle_metrics(sampled_pairs)
            if metric.primary_high_id in eligible_ids
        )
        sampled_ranks = rank_fronts(
            {metric.primary_high_id: metric.objectives.values for metric in sampled_metrics}
        )
        for primary_high_id, rank in sampled_ranks.items():
            present_counts[primary_high_id] += 1
            if rank == 1:
                front_counts[primary_high_id] += 1

    return tuple(
        BundleParetoEstimate(
            metrics=metric,
            pareto_front_rank=ranks[metric.primary_high_id],
            front_1_probability=(
                front_counts[metric.primary_high_id] / present_counts[metric.primary_high_id]
                if present_counts[metric.primary_high_id]
                else 0.0
            ),
            dominated_probability=(
                1 - front_counts[metric.primary_high_id] / present_counts[metric.primary_high_id]
                if present_counts[metric.primary_high_id]
                else 0.0
            ),
        )
        for metric in metrics
    )


def _index_results(
    results: Iterable[FeedbackEpisode], expected_method: MethodName
) -> dict[tuple[str, int], FeedbackEpisode]:
    indexed = {}
    for result in results:
        if (
            result.benchmark is not Benchmark.ALFWORLD
            or result.split is not ExperimentSplit.TRAIN
            or result.method is not expected_method
        ):
            raise ValueError(f"invalid {expected_method} feedback result: {result.episode_key}")
        key = (result.sample_id, result.repeat_id)
        if key in indexed:
            raise ValueError(f"duplicate feedback result: {key}")
        indexed[key] = result
    return indexed


def _from_result(result: EpisodeResult) -> FeedbackEpisode:
    if result.success is None:
        raise ValueError(f"feedback result has no binary success: {result.episode_key}")
    return FeedbackEpisode(
        benchmark=result.benchmark,
        split=result.split,
        method=result.method,
        sample_id=result.sample_id,
        repeat_id=result.repeat_id,
        success=result.success,
        steps=result.steps,
        skill_ids=result.skill_ids,
        context_skill_ids=result.context_skill_ids,
    )


def _measure_bundle(primary_high_id: str, pairs: list[FeedbackPair]) -> BundleMetrics:
    success = np.asarray([pair.tower_success for pair in pairs], dtype=np.float64)
    paired_gain = np.asarray(
        [int(pair.tower_success) - int(pair.no_skill_success) for pair in pairs],
        dtype=np.float64,
    )
    return BundleMetrics(
        primary_high_id=primary_high_id,
        exposure_count=len(pairs),
        objectives=DeploymentObjectives(
            performance_level=float(success.mean()),
            paired_success_gain=float(paired_gain.mean()),
            guarded_step_saving=float(np.mean([_guarded_step(pair) for pair in pairs])),
        ),
    )


def _guarded_step(pair: FeedbackPair) -> float:
    raw_step = (pair.no_skill_steps - pair.tower_steps) / max(pair.no_skill_steps, 1)
    return min(raw_step, 0.0) if pair.tower_success < pair.no_skill_success else raw_step
