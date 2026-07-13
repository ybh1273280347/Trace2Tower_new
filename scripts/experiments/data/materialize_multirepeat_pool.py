from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import yaml
from scripts.experiments.run.rollout_no_skill_train import write_json

from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.results import MethodName
from trace2tower.trajectory import (
    EpisodeTrajectory,
    TrajectoryReader,
    write_trajectory_jsonl,
)


def validate_multirepeat_pool(
    source: tuple[EpisodeTrajectory, ...],
    trajectories: tuple[EpisodeTrajectory, ...],
    *,
    benchmark: Benchmark,
    run_id: str,
    repeat_ids: tuple[int, ...],
) -> tuple[str, ...]:
    if not repeat_ids or any(repeat_id < 0 for repeat_id in repeat_ids):
        raise ValueError("repeat IDs must be non-negative and non-empty")
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("repeat IDs must be unique")
    if not source:
        raise ValueError("sample source pool is empty")
    if any(
        trajectory.benchmark is not benchmark
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.NO_SKILL
        for trajectory in source
    ):
        raise ValueError("sample source must contain No-Skill training trajectories")

    source_goals = {}
    for trajectory in source:
        previous = source_goals.setdefault(trajectory.sample_id, trajectory.task_goal)
        if previous != trajectory.task_goal:
            raise ValueError("sample source contains conflicting task goals")
    sample_ids = tuple(sorted(source_goals))
    expected = {
        (sample_id, repeat_id)
        for sample_id in sample_ids
        for repeat_id in repeat_ids
    }
    actual = {(trajectory.sample_id, trajectory.repeat_id) for trajectory in trajectories}
    if len(actual) != len(trajectories):
        raise ValueError("matrix trajectories contain duplicate episode keys")
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"matrix trajectory coverage differs: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    if any(
        trajectory.run_id != run_id
        or trajectory.benchmark is not benchmark
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.NO_SKILL
        or trajectory.task_goal != source_goals[trajectory.sample_id]
        for trajectory in trajectories
    ):
        raise ValueError("matrix trajectories violate the execution or task contract")
    return sample_ids


def read_matrix_trajectories(run_root: Path) -> tuple[EpisodeTrajectory, ...]:
    episode_dirs = sorted(run_root.glob("shard-*/trajectories"))
    return tuple(
        trajectory
        for episode_dir in episode_dirs
        for trajectory in TrajectoryReader.read_episode_files(episode_dir)
    )


def main(options: argparse.Namespace) -> int:
    benchmark = Benchmark(options.benchmark)
    repeat_ids = tuple(sorted(options.repeat_id))
    source = TrajectoryReader.read_jsonl(options.sample_source_pool)
    run_root = (
        options.runs_dir
        / options.run_id
        / benchmark
        / ExperimentSplit.TRAIN
        / MethodName.NO_SKILL
    )
    trajectories = read_matrix_trajectories(run_root)
    sample_ids = validate_multirepeat_pool(
        source,
        trajectories,
        benchmark=benchmark,
        run_id=options.run_id,
        repeat_ids=repeat_ids,
    )
    resolved_config_path = options.runs_dir / options.run_id / "resolved-config.yaml"
    resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    if tuple(resolved_config.get("selection", {}).get("repeat_ids", ())) != repeat_ids:
        raise ValueError("resolved config does not prove the requested repeat IDs")
    if resolved_config.get("method", {}).get("method") != MethodName.NO_SKILL:
        raise ValueError("resolved config is not a No-Skill run")

    count = write_trajectory_jsonl(trajectories, options.output)
    success_counts = Counter(
        trajectory.repeat_id
        for trajectory in trajectories
        if trajectory.primary_score >= options.success_threshold
    )
    report = {
        "run_id": options.run_id,
        "benchmark": benchmark.value,
        "method": MethodName.NO_SKILL.value,
        "agent_model": resolved_config["agent_model"],
        "sample_source_pool": options.sample_source_pool.as_posix(),
        "sample_source_pool_sha256": hashlib.sha256(
            options.sample_source_pool.read_bytes()
        ).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            resolved_config_path.read_bytes()
        ).hexdigest(),
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
        "repeat_ids": list(repeat_ids),
        "expected_trajectory_count": len(sample_ids) * len(repeat_ids),
        "trajectory_count": count,
        "full_success_count": sum(success_counts.values()),
        "full_success_count_by_repeat": {
            str(repeat_id): success_counts[repeat_id] for repeat_id in repeat_ids
        },
        "output": options.output.as_posix(),
        "output_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
    }
    report_path = options.output.with_suffix(".audit.json")
    write_json(report_path, report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-source-pool", type=Path, required=True)
    parser.add_argument("--repeat-id", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--success-threshold", type=float, default=0.999)
    raise SystemExit(main(parser.parse_args()))
