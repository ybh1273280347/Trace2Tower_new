from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2tower.trajectory import (
    EpisodeTrajectory,
    TrajectoryReader,
    write_trajectory_jsonl,
)


def read_run(root: Path) -> tuple:
    return tuple(
        trajectory
        for episode_dir in sorted(root.glob("alfworld/train/no_skill/shard-*/trajectories"))
        for trajectory in TrajectoryReader.read_episode_files(episode_dir)
    )


def deduplicate_trajectories(
    trajectories: tuple[EpisodeTrajectory, ...],
) -> tuple[tuple[EpisodeTrajectory, ...], dict]:
    selected = {}
    duplicate_keys = set()
    score_disagreements = 0
    step_disagreements = 0
    for trajectory in trajectories:
        key = (trajectory.sample_id, trajectory.repeat_id)
        existing = selected.get(key)
        if existing is None:
            selected[key] = trajectory
            continue
        duplicate_keys.add(key)
        score_disagreements += trajectory.primary_score != existing.primary_score
        step_disagreements += len(trajectory.steps) != len(existing.steps)
    return tuple(selected.values()), {
        "raw_trajectory_count": len(trajectories),
        "selected_trajectory_count": len(selected),
        "duplicate_key_count": len(duplicate_keys),
        "discarded_trajectory_count": len(trajectories) - len(selected),
        "score_disagreement_count": score_disagreements,
        "step_disagreement_count": step_disagreements,
    }


def main(options: argparse.Namespace) -> int:
    base = TrajectoryReader.read_jsonl(options.base_pool)
    additions, duplicate_audit = deduplicate_trajectories(
        read_run(options.additions_run)
    )
    trajectories = tuple(base) + tuple(additions)
    keys = [(trajectory.sample_id, trajectory.repeat_id) for trajectory in trajectories]
    if len(keys) != len(set(keys)):
        raise ValueError("merged pool contains duplicate sample/repeat keys")
    expected_ids = {
        json.loads(line)["sample_id"]
        for line in options.full_manifest.read_text(encoding="utf-8").splitlines()
        if line
    }
    expected_repeats = set(options.repeat_id)
    observed = set(keys)
    expected = {
        (sample_id, repeat_id)
        for sample_id in expected_ids
        for repeat_id in expected_repeats
    }
    if observed != expected:
        raise ValueError(
            f"pool coverage mismatch: missing={len(expected - observed)} "
            f"unexpected={len(observed - expected)}"
        )
    if any(
        trajectory.benchmark.value != "alfworld"
        or trajectory.split.value != "train"
        or trajectory.method.value != "no_skill"
        for trajectory in trajectories
    ):
        raise ValueError("pool contains a trajectory from the wrong protocol")
    count = write_trajectory_jsonl(trajectories, options.output)
    successful = sum(trajectory.primary_score == 1 for trajectory in trajectories)
    report = {
        "base_pool": options.base_pool.as_posix(),
        "additions_run": options.additions_run.as_posix(),
        "additions_duplicate_audit": duplicate_audit,
        "full_manifest": options.full_manifest.as_posix(),
        "repeat_ids": sorted(expected_repeats),
        "task_count": len(expected_ids),
        "trajectory_count": count,
        "success_count": successful,
        "success_rate": successful / count,
        "output": options.output.as_posix(),
    }
    options.output.with_suffix(".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--additions-run", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--repeat-id", type=int, action="append", default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
