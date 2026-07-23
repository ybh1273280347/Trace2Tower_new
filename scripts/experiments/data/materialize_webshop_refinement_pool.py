from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trace2tower.core.trajectory import (
    EpisodeTrajectory,
    TrajectoryReader,
    write_trajectory_jsonl,
)


def read_run_trajectories(root: Path) -> tuple[EpisodeTrajectory, ...]:
    trajectories = tuple(
        EpisodeTrajectory.from_record(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.rglob("trajectories/*.json"))
    )
    if not trajectories:
        raise ValueError(f"no trajectory files found under {root}")
    ids = [trajectory.trajectory_id for trajectory in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate trajectory IDs under {root}")
    return trajectories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-pool", type=Path, required=True)
    parser.add_argument("--noskill-root", type=Path, required=True)
    parser.add_argument("--tower-root", type=Path, required=True)
    parser.add_argument("--expected-feedback-count", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    options = parser.parse_args()

    construction = TrajectoryReader.read_jsonl(options.construction_pool)
    noskill = read_run_trajectories(options.noskill_root)
    tower = read_run_trajectories(options.tower_root)
    if len(noskill) != options.expected_feedback_count:
        raise ValueError(f"NoSkill feedback is incomplete: {len(noskill)}")
    if len(tower) != options.expected_feedback_count:
        raise ValueError(f"Tower feedback is incomplete: {len(tower)}")
    baseline_keys = {(item.sample_id, item.repeat_id) for item in noskill}
    tower_keys = {(item.sample_id, item.repeat_id) for item in tower}
    if baseline_keys != tower_keys:
        raise ValueError("NoSkill and Tower feedback keys are not identical")
    if any(item.split.value != "train" for item in (*noskill, *tower)):
        raise ValueError("refinement feedback must be train-only")
    all_trajectories = (*construction, *noskill, *tower)
    all_ids = [item.trajectory_id for item in all_trajectories]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("construction and refinement trajectories overlap")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    count = write_trajectory_jsonl(all_trajectories, options.output)
    audit = {
        "protocol_id": "webshop-train-refinement-v1",
        "construction_pool": options.construction_pool.as_posix(),
        "noskill_run_root": options.noskill_root.as_posix(),
        "tower_run_root": options.tower_root.as_posix(),
        "construction_count": len(construction),
        "feedback_noskill_count": len(noskill),
        "feedback_tower_count": len(tower),
        "materialized_count": count,
        "paired_feedback_count": len(baseline_keys),
        "output": options.output.as_posix(),
        "output_sha256": hashlib.sha256(options.output.read_bytes()).hexdigest(),
        "complete": True,
    }
    options.audit.parent.mkdir(parents=True, exist_ok=True)
    options.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
