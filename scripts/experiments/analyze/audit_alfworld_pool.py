from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as parquet

from trace2tower.trajectory import TrajectoryReader, write_trajectory_jsonl


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    expected_ids = set(protocol["trajectory_pool"]["sample_ids"])
    expected_repeats = set(protocol["trajectory_pool"]["repeat_ids"])
    expected_keys = {
        (sample_id, repeat_id)
        for sample_id in expected_ids
        for repeat_id in expected_repeats
    }
    expansion_records = [
        json.loads(path.read_text(encoding="utf-8")) for path in options.expansion
    ]
    for expansion in expansion_records:
        expected_ids.update(expansion["sample_ids"])
        expected_keys.update(
            (sample_id, repeat_id)
            for sample_id in expansion["sample_ids"]
            for repeat_id in expansion["repeat_ids"]
        )
    family_by_id = {
        f"alfworld:train:{item['task_id']}": item["task_family"]
        for item in parquet.read_table(
            options.dataset_root / "train.parquet", columns=["extra_info"]
        ).column(0).combine_chunks().to_pylist()
    }
    run_roots = (options.run_root, *options.additional_run_root)
    trajectories = tuple(
        trajectory
        for run_root in run_roots
        for episode_dir in sorted(run_root.glob("shard-*/trajectories"))
        for trajectory in TrajectoryReader.read_episode_files(episode_dir)
    )
    keys = {(trajectory.sample_id, trajectory.repeat_id) for trajectory in trajectories}
    if keys != expected_keys or len(keys) != len(trajectories):
        raise ValueError(
            f"trajectory coverage mismatch: missing={len(expected_keys - keys)}, "
            f"unexpected={len(keys - expected_keys)}"
        )
    successful = [trajectory for trajectory in trajectories if trajectory.primary_score == 1]
    successful_tasks = defaultdict(set)
    for trajectory in successful:
        successful_tasks[family_by_id[trajectory.sample_id]].add(trajectory.sample_id)
    distinct_counts = {
        family: len(sample_ids) for family, sample_ids in sorted(successful_tasks.items())
    }
    required_per_family = int(
        protocol["trajectory_pool"]["minimum_successful_distinct_tasks_per_family"]
    )
    required_successes = int(protocol["trajectory_pool"]["minimum_successful_trajectory_count"])
    deficient = {
        family: count
        for family, count in Counter(
            family_by_id[sample_id] for sample_id in expected_ids
        ).items()
        if distinct_counts.get(family, 0) < required_per_family
    }
    passed = len(successful) >= required_successes and not deficient
    count = write_trajectory_jsonl(trajectories, options.output)
    report = {
        "run_roots": [path.as_posix() for path in run_roots],
        "expansions": [path.as_posix() for path in options.expansion],
        "trajectory_count": count,
        "success_count": len(successful),
        "success_rate": len(successful) / len(trajectories),
        "successful_distinct_tasks_by_family": distinct_counts,
        "required_success_count": required_successes,
        "required_successful_distinct_tasks_per_family": required_per_family,
        "deficient_families": deficient,
        "pool_stop_condition_met": passed,
        "output": options.output.as_posix(),
    }
    options.output.with_suffix(".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--additional-run-root", type=Path, action="append", default=[])
    parser.add_argument("--expansion", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiments/alfworld_protocol_v1.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    raise SystemExit(main(parser.parse_args()))
