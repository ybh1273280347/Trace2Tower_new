from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.analyze_refinement_test import (
    paired_interval,
    read_first_completion,
    summarize,
)


def select_repeat(run_dir: Path, repeat_id: int) -> dict[tuple[str, int], dict]:
    rows, _ = read_first_completion(run_dir)
    selected = {
        (sample_id, repeat_id): row
        for (sample_id, current_repeat), row in rows.items()
        if current_repeat == repeat_id
    }
    if len(selected) != 100:
        raise ValueError(f"repeat {repeat_id} must cover 100 Test-B tasks")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat0-run", type=Path, required=True)
    parser.add_argument("--repeat1-run", type=Path, required=True)
    parser.add_argument("--tower-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    repeat0 = select_repeat(options.repeat0_run, 0)
    repeat1 = select_repeat(options.repeat1_run, 1)
    tower = select_repeat(options.tower_run, 0)
    sample_ids = tuple(sorted(sample_id for sample_id, _ in repeat0))
    if {sample_id for sample_id, _ in repeat1} != set(sample_ids) or {
        sample_id for sample_id, _ in tower
    } != set(sample_ids):
        raise ValueError("Test-B variance runs cover different tasks")

    repeat0_scores = np.array(
        [repeat0[(sample_id, 0)]["primary_score"] for sample_id in sample_ids]
    )
    repeat1_scores = np.array(
        [repeat1[(sample_id, 1)]["primary_score"] for sample_id in sample_ids]
    )
    tower_scores = np.array([tower[(sample_id, 0)]["primary_score"] for sample_id in sample_ids])
    noskill_mean_scores = (repeat0_scores + repeat1_scores) / 2

    payload = {
        "protocol_id": "webshop-test-b-noskill-variance-v1",
        "agent_model": "deepseek-v4-flash",
        "runs": {
            "noskill_repeat0": options.repeat0_run.as_posix(),
            "noskill_repeat1": options.repeat1_run.as_posix(),
            "final_tower_repeat0": options.tower_run.as_posix(),
        },
        "summary": {
            "noskill_repeat0": summarize(repeat0),
            "noskill_repeat1": summarize(repeat1),
            "noskill_two_repeat_task_mean_reward": float(np.mean(noskill_mean_scores)),
            "final_tower_repeat0": summarize(tower),
        },
        "paired_deltas": {
            "noskill_repeat1_minus_repeat0": {
                "mean_reward": float(np.mean(repeat1_scores - repeat0_scores)),
                "bootstrap_ci95": paired_interval(repeat1_scores - repeat0_scores),
            },
            "tower_minus_noskill_repeat0": {
                "mean_reward": float(np.mean(tower_scores - repeat0_scores)),
                "bootstrap_ci95": paired_interval(tower_scores - repeat0_scores),
            },
            "tower_minus_noskill_repeat1": {
                "mean_reward": float(np.mean(tower_scores - repeat1_scores)),
                "bootstrap_ci95": paired_interval(tower_scores - repeat1_scores),
            },
            "tower_minus_noskill_two_repeat_mean": {
                "mean_reward": float(np.mean(tower_scores - noskill_mean_scores)),
                "bootstrap_ci95": paired_interval(tower_scores - noskill_mean_scores),
            },
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
