from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.results import MethodName
from trace2tower.trajectory import TrajectoryReader, write_trajectory_jsonl


def main(options: argparse.Namespace) -> int:
    trajectories = tuple(
        trajectory
        for directory in sorted(options.run_root.glob("shard-*/trajectories"))
        for trajectory in TrajectoryReader.read_episode_files(directory)
    )
    if not trajectories or any(
        trajectory.benchmark is not options.benchmark
        or trajectory.split is not options.split
        or trajectory.method is not options.method
        for trajectory in trajectories
    ):
        raise ValueError("run trajectories violate the requested execution contract")
    count = write_trajectory_jsonl(trajectories, options.output)
    report = {
        "run_root": options.run_root.as_posix(),
        "benchmark": options.benchmark.value,
        "split": options.split.value,
        "method": options.method.value,
        "trajectory_count": count,
        "output": options.output.as_posix(),
        "output_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
    }
    options.output.with_suffix(".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--split", type=ExperimentSplit, choices=tuple(ExperimentSplit), required=True)
    parser.add_argument("--method", type=MethodName, choices=tuple(MethodName), required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
