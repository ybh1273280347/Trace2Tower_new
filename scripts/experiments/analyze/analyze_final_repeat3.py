from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.analyze_refinement_test import (
    paired_interval,
    read_first_completion,
)


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("run must be METHOD=PATH")
    return name, Path(raw_path)


def load_method_runs(paths: list[Path]) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in paths:
        run_rows, _ = read_first_completion(path)
        overlap = set(rows) & set(run_rows)
        if overlap:
            raise ValueError(f"repeat3 run paths overlap on {len(overlap)} keys")
        rows.update(run_rows)
    by_sample: dict[str, set[int]] = defaultdict(set)
    for sample_id, repeat_id in rows:
        by_sample[sample_id].add(repeat_id)
    if len(by_sample) != 100 or any(repeats != {0, 1, 2} for repeats in by_sample.values()):
        raise ValueError("repeat3 requires 100 tasks with real repeat IDs 0, 1, and 2")
    return rows


def metric_summary(rows: dict[tuple[str, int], dict]) -> dict:
    repeat_summaries = {}
    for repeat_id in (0, 1, 2):
        selected = [row for key, row in rows.items() if key[1] == repeat_id]
        repeat_summaries[str(repeat_id)] = {
            "mean_reward": float(np.mean([row["primary_score"] for row in selected])),
            "full_success_rate": float(
                np.mean([row["primary_score"] >= 0.999 for row in selected])
            ),
        }
    task_metrics = {}
    for sample_id in sorted({key[0] for key in rows}):
        sample_rows = [rows[(sample_id, repeat_id)] for repeat_id in (0, 1, 2)]
        task_metrics[sample_id] = {
            "reward": float(np.mean([row["primary_score"] for row in sample_rows])),
            "full_success": float(
                np.mean([row["primary_score"] >= 0.999 for row in sample_rows])
            ),
            "steps": float(np.mean([row["steps"] for row in sample_rows])),
            "invalid_actions": float(
                np.mean([row["invalid_actions"] for row in sample_rows])
            ),
            "input_tokens": float(
                np.mean([row["input_tokens"] for row in sample_rows])
            ),
        }
    return {
        "repeat_summaries": repeat_summaries,
        "task_metrics": task_metrics,
        "aggregate": {
            metric: float(np.mean([values[metric] for values in task_metrics.values()]))
            for metric in (
                "reward",
                "full_success",
                "steps",
                "invalid_actions",
                "input_tokens",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--final-method", default="final_t1")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    paths_by_method: dict[str, list[Path]] = defaultdict(list)
    for method, path in options.run:
        paths_by_method[method].append(path)
    if options.final_method not in paths_by_method:
        raise ValueError("final method is absent from repeat3 runs")
    summaries = {
        method: metric_summary(load_method_runs(paths))
        for method, paths in sorted(paths_by_method.items())
    }
    final_tasks = summaries[options.final_method]["task_metrics"]
    comparisons = {}
    for method, summary in summaries.items():
        if method == options.final_method:
            continue
        other_tasks = summary["task_metrics"]
        if set(other_tasks) != set(final_tasks):
            raise ValueError("repeat3 methods cover different tasks")
        reward_differences = np.array(
            [
                final_tasks[sample_id]["reward"] - other_tasks[sample_id]["reward"]
                for sample_id in sorted(final_tasks)
            ]
        )
        success_differences = np.array(
            [
                final_tasks[sample_id]["full_success"]
                - other_tasks[sample_id]["full_success"]
                for sample_id in sorted(final_tasks)
            ]
        )
        comparisons[f"{options.final_method}_minus_{method}"] = {
            "mean_reward": float(np.mean(reward_differences)),
            "reward_bootstrap_ci95": paired_interval(reward_differences),
            "full_success_rate": float(np.mean(success_differences)),
            "full_success_bootstrap_ci95": paired_interval(success_differences),
        }
    payload = {
        "protocol_id": "webshop-final-graph-cap3-repeat3-v1",
        "model": options.model,
        "statistical_unit": "task_mean_across_real_repeats_0_1_2",
        "runs": {
            method: [path.as_posix() for path in paths]
            for method, paths in sorted(paths_by_method.items())
        },
        "summary": {
            method: {
                "repeat_summaries": summary["repeat_summaries"],
                "aggregate": summary["aggregate"],
            }
            for method, summary in summaries.items()
        },
        "paired_comparisons": comparisons,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
