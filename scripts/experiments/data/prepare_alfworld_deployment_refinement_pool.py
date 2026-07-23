from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.core.manifests import Benchmark, ExperimentSplit, read_manifest
from trace2tower.core.results import MethodName
from trace2tower.core.trajectory import (
    EpisodeTrajectory,
    TrajectoryReader,
    write_trajectory_jsonl,
)


def validate_refinement_inputs(
    base: tuple[EpisodeTrajectory, ...],
    feedback: tuple[EpisodeTrajectory, ...],
    feedback_sample_ids: frozenset[str],
) -> None:
    if not base or not feedback or not feedback_sample_ids:
        raise ValueError("refinement inputs cannot be empty")
    if any(
        trajectory.benchmark is not Benchmark.ALFWORLD
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.NO_SKILL
        for trajectory in base
    ):
        raise ValueError("base refinement pool must contain ALFWorld train No-Skill trajectories")
    if any(
        trajectory.benchmark is not Benchmark.ALFWORLD
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.TRACE2TOWER
        or trajectory.repeat_id != 0
        for trajectory in feedback
    ):
        raise ValueError("deployment feedback must contain ALFWorld train Tower repeat 0")
    base_sample_ids = {trajectory.sample_id for trajectory in base}
    observed_feedback_ids = {trajectory.sample_id for trajectory in feedback}
    if observed_feedback_ids != feedback_sample_ids:
        raise ValueError(
            "deployment feedback does not exactly cover its manifest: "
            f"expected={len(feedback_sample_ids)}, observed={len(observed_feedback_ids)}"
        )
    if len(feedback) != len(observed_feedback_ids):
        raise ValueError("deployment feedback must contain one trajectory per task")
    if base_sample_ids & feedback_sample_ids:
        raise ValueError("base construction and deployment feedback tasks overlap")


def main(options: argparse.Namespace) -> int:
    base = TrajectoryReader.read_jsonl(options.base_pool)
    feedback_runs = tuple(options.feedback_run or _default_feedback_runs())
    feedback_paths = sorted(
        path for feedback_run in feedback_runs for path in feedback_run.rglob("trajectories/*.json")
    )
    feedback = tuple(
        EpisodeTrajectory.from_record(json.loads(path.read_text(encoding="utf-8")))
        for path in feedback_paths
    )
    manifest = read_manifest(options.feedback_manifest)
    feedback_sample_ids = frozenset(entry.sample_id for entry in manifest)
    validate_refinement_inputs(base, feedback, feedback_sample_ids)

    trajectories = (*base, *feedback)
    write_trajectory_jsonl(trajectories, options.output)
    audit = {
        "protocol_id": "alfworld-deployment-optimization-v1-refinement-input",
        "benchmark": Benchmark.ALFWORLD.value,
        "split": ExperimentSplit.TRAIN.value,
        "base_pool": {
            "path": options.base_pool.as_posix(),
            "sha256": hashlib.sha256(options.base_pool.read_bytes()).hexdigest(),
            "trajectory_count": len(base),
            "task_count": len({trajectory.sample_id for trajectory in base}),
        },
        "feedback": {
            "run_paths": [path.as_posix() for path in feedback_runs],
            "trajectory_tree_sha256": _tree_hash(feedback_paths),
            "trajectory_count": len(feedback),
            "task_count": len(feedback_sample_ids),
            "repeat_ids": sorted({trajectory.repeat_id for trajectory in feedback}),
            "manifest_path": options.feedback_manifest.as_posix(),
            "manifest_sha256": hashlib.sha256(options.feedback_manifest.read_bytes()).hexdigest(),
        },
        "output": {
            "path": options.output.as_posix(),
            "sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
            "trajectory_count": len(trajectories),
            "method_counts": dict(
                sorted(Counter(item.method.value for item in trajectories).items())
            ),
        },
    }
    write_json(options.output.with_suffix(".audit.json"), audit)
    print(yaml.safe_dump(audit, sort_keys=False))
    return 0


def _tree_hash(paths: list[Path]) -> str:
    records = [
        f"{path.as_posix()}\0{hashlib.sha256(path.read_bytes()).hexdigest()}" for path in paths
    ]
    return hashlib.sha256("\n".join(records).encode()).hexdigest()


def _default_feedback_runs() -> tuple[Path, ...]:
    root = Path("artifacts/runs")
    return (
        root / "alfworld-deployment-v1-feedback-pilot-tower-v0-r0",
        root / "alfworld-deployment-v1-feedback-remaining-tower-v0-r0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-pool",
        type=Path,
        default=Path("artifacts/trajectories/alfworld/alfworld-pool-v1-pro-expanded.jsonl"),
    )
    parser.add_argument(
        "--feedback-run",
        action="append",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--feedback-manifest",
        type=Path,
        default=Path(
            "experiments/alfworld/deployment-optimization-v1/manifests/"
            "deployment_feedback_pilot.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/trace2tower/alfworld/deployment-optimization-v1/"
            "refinement/pilot-input-trajectories.jsonl"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
