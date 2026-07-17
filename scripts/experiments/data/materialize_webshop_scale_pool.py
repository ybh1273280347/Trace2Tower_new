from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import MethodName
from trace2tower.core.trajectory import (
    EpisodeTrajectory,
    TrajectoryReader,
    write_trajectory_jsonl,
)

RUN_IDS = {
    "p100": "webshop-scale-v1-flash-p100-add50",
    "p200": "webshop-scale-v1-flash-p200-add100",
}


def read_run_trajectories(runs_root: Path, run_id: str) -> tuple[EpisodeTrajectory, ...]:
    method_root = runs_root / run_id / "webshop" / "train" / "no_skill"
    return tuple(
        trajectory
        for directory in sorted(method_root.glob("shard-*/trajectories"))
        for trajectory in TrajectoryReader.read_episode_files(directory)
    )


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    training = protocol["training"]
    pool = training["pools"][options.pool]
    repeat_ids = tuple(int(item) for item in training["repeat_ids"])
    p50_audit_path = Path(training["p50_source_audit"])
    if (
        hashlib.sha256(p50_audit_path.read_bytes()).hexdigest()
        != training["p50_source_audit_sha256"]
    ):
        raise ValueError("P50 audit changed after scale protocol freeze")

    p50_audit = json.loads(p50_audit_path.read_text(encoding="utf-8"))
    source_paths = [Path(p50_audit["output"])]
    trajectories = list(TrajectoryReader.read_jsonl(source_paths[0]))
    if options.pool in ("p100", "p200"):
        trajectories.extend(read_run_trajectories(options.runs_root, RUN_IDS["p100"]))
        source_paths.append(options.runs_root / RUN_IDS["p100"])
    if options.pool == "p200":
        trajectories.extend(read_run_trajectories(options.runs_root, RUN_IDS["p200"]))
        source_paths.append(options.runs_root / RUN_IDS["p200"])

    expected = {
        (sample_id, repeat_id) for sample_id in pool["sample_ids"] for repeat_id in repeat_ids
    }
    actual = [(item.sample_id, item.repeat_id) for item in trajectories]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        missing = sorted(expected - set(actual))
        unexpected = sorted(set(actual) - expected)
        raise ValueError(
            f"{options.pool} coverage mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if any(
        item.benchmark is not Benchmark.WEBSHOP
        or item.split is not ExperimentSplit.TRAIN
        or item.method is not MethodName.NO_SKILL
        for item in trajectories
    ):
        raise ValueError("scale pool must contain only WebShop train NoSkill trajectories")

    ordered = tuple(
        sorted(
            trajectories,
            key=lambda item: (
                pool["sample_ids"].index(item.sample_id),
                item.repeat_id,
            ),
        )
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    count = write_trajectory_jsonl(ordered, options.output)
    report = {
        "protocol_id": protocol["protocol_id"],
        "pool": options.pool,
        "source_paths": [path.as_posix() for path in source_paths],
        "task_count": pool["task_count"],
        "repeat_ids": list(repeat_ids),
        "trajectory_count": count,
        "expected_trajectory_count": pool["expected_episode_count"],
        "full_success_count": sum(item.primary_score >= 0.999 for item in ordered),
        "output": options.output.as_posix(),
        "output_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
    }
    write_json(options.output.with_suffix(".audit.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=("p50", "p100", "p200"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/webshop_scale_v1.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
