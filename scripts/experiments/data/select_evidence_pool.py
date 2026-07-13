from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import yaml

from trace2tower.trajectory import TrajectoryReader, write_trajectory_jsonl

from scripts.experiments.run.rollout_no_skill_train import write_json


def main(options: argparse.Namespace) -> int:
    trajectories = TrajectoryReader.read_jsonl(options.input)
    successful_samples = {
        trajectory.sample_id
        for trajectory in trajectories
        if trajectory.primary_score >= options.success_threshold
    }
    selected = []
    reasons = {}
    excluded = []
    for trajectory in trajectories:
        if trajectory.primary_score >= options.success_threshold:
            reason = "full_success"
        elif options.policy == "success-only":
            excluded.append(trajectory.trajectory_id)
            continue
        elif trajectory.primary_score > 0:
            reason = "partial_reward"
        elif trajectory.sample_id in successful_samples and trajectory.steps:
            reason = "same_task_contrast"
        else:
            excluded.append(trajectory.trajectory_id)
            continue
        selected.append(trajectory)
        reasons[trajectory.trajectory_id] = reason

    count = write_trajectory_jsonl(selected, options.output)
    reason_counts = Counter(reasons.values())
    report = {
        "policy": options.policy,
        "success_threshold": options.success_threshold,
        "input": options.input.as_posix(),
        "input_sha256": hashlib.sha256(options.input.read_bytes()).hexdigest(),
        "input_trajectory_count": len(trajectories),
        "successful_sample_count": len(successful_samples),
        "selected_trajectory_count": count,
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "selection_reasons": dict(sorted(reasons.items())),
        "excluded_trajectory_count": len(excluded),
        "excluded_trajectory_ids": sorted(excluded),
        "output": options.output.as_posix(),
        "output_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
    }
    write_json(options.output.with_suffix(".audit.json"), report)
    print(yaml.safe_dump({key: value for key, value in report.items() if key not in {"selection_reasons", "excluded_trajectory_ids"}}, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("success-only", "success-prioritized-contrastive-v1"),
        default="success-prioritized-contrastive-v1",
    )
    parser.add_argument("--success-threshold", type=float, default=0.999)
    raise SystemExit(main(parser.parse_args()))
