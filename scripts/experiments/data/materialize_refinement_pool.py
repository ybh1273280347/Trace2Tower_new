from __future__ import annotations

import argparse
import hashlib
import json
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


def read_run_trajectories(run_root: Path) -> tuple[EpisodeTrajectory, ...]:
    return tuple(
        trajectory
        for trajectory_dir in sorted(run_root.glob("shard-*/trajectories"))
        for trajectory in TrajectoryReader.read_episode_files(trajectory_dir)
    )


def main(options: argparse.Namespace) -> int:
    benchmark = Benchmark(options.benchmark)
    initial = TrajectoryReader.read_jsonl(options.initial_pool)
    conditioned = read_run_trajectories(options.skill_run_root)
    report = json.loads(options.refinement_report.read_text(encoding="utf-8"))

    if not initial or any(
        trajectory.benchmark is not benchmark
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.NO_SKILL
        for trajectory in initial
    ):
        raise ValueError("initial refinement pool must contain train No-Skill trajectories")
    if not conditioned or any(
        trajectory.benchmark is not benchmark
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.TRACE2TOWER_STATIC
        for trajectory in conditioned
    ):
        raise ValueError(
            "conditioned refinement pool must contain train Trace2Tower trajectories"
        )

    report_keys = {
        (item["sample_id"], int(item["repeat_id"]))
        for item in report["audit"]["paired_episode_keys"]
    }
    trajectory_keys = {
        (trajectory.sample_id, trajectory.repeat_id) for trajectory in conditioned
    }
    if len(trajectory_keys) != len(conditioned) or trajectory_keys != report_keys:
        raise ValueError("conditioned trajectories differ from audited exposure keys")
    expected_run_ids = set(report["execution_contract"]["skill_run_ids"])
    if {trajectory.run_id for trajectory in conditioned} != expected_run_ids:
        raise ValueError("conditioned trajectories differ from audited run IDs")
    if report["benchmark"] != benchmark or not report["audit"]["is_complete"]:
        raise ValueError("refinement report is incomplete or belongs to another benchmark")

    combined = (*initial, *conditioned)
    count = write_trajectory_jsonl(combined, options.output)
    report_path = options.output.with_suffix(".audit.json")
    pool_report = {
        "benchmark": benchmark.value,
        "tower_snapshot_id": report["tower_snapshot_id"],
        "initial_pool": options.initial_pool.as_posix(),
        "initial_pool_sha256": hashlib.sha256(options.initial_pool.read_bytes()).hexdigest(),
        "refinement_report": options.refinement_report.as_posix(),
        "refinement_report_sha256": hashlib.sha256(
            options.refinement_report.read_bytes()
        ).hexdigest(),
        "initial_trajectory_count": len(initial),
        "conditioned_trajectory_count": len(conditioned),
        "combined_trajectory_count": count,
        "output": options.output.as_posix(),
        "output_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
    }
    write_json(report_path, pool_report)
    print(yaml.safe_dump(pool_report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--initial-pool", type=Path, required=True)
    parser.add_argument("--skill-run-root", type=Path, required=True)
    parser.add_argument("--refinement-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
