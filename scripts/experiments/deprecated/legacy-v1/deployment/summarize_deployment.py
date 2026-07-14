from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.experiments.run.rollout_no_skill_train import write_json


FULL_SUCCESS = 0.999
BOOTSTRAP_SAMPLES = 10_000


def read_results(root: Path) -> list[dict]:
    paths = sorted(root.glob("shard-*/results.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"{root} must contain ten result shards")
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    keys = [(str(row["sample_id"]), int(row["repeat_id"])) for row in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError(f"{root} contains duplicate or empty results")
    if any(
        row.get("chat_input_tokens") is None
        or row.get("chat_output_tokens") is None
        for row in rows
    ):
        raise ValueError(f"{root} has incomplete chat-token evidence")
    return rows


def chat_tokens(row: dict) -> int:
    return int(row["chat_input_tokens"]) + int(row["chat_output_tokens"])


def aggregate(rows: list[dict]) -> dict:
    return {
        "episode_count": len(rows),
        "task_count": len({row["sample_id"] for row in rows}),
        "mean_reward": fmean(float(row["primary_score"]) for row in rows),
        "full_success_rate": fmean(
            float(row["primary_score"] >= FULL_SUCCESS) for row in rows
        ),
        "mean_steps": fmean(int(row["steps"]) for row in rows),
        "mean_chat_tokens": fmean(chat_tokens(row) for row in rows),
        "mean_skill_context_chars": fmean(
            int(row["skill_context_chars"]) for row in rows
        ),
    }


def bootstrap(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    estimates = array[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def paired(baseline: list[dict], candidate: list[dict], seed: int) -> dict:
    baseline_by_key = {
        (str(row["sample_id"]), int(row["repeat_id"])): row for row in baseline
    }
    candidate_by_key = {
        (str(row["sample_id"]), int(row["repeat_id"])): row for row in candidate
    }
    if set(baseline_by_key) != set(candidate_by_key):
        raise ValueError("paired comparison requires identical episode keys")
    task_ids = sorted({key[0] for key in baseline_by_key})
    metrics = {
        "reward": lambda row: float(row["primary_score"]),
        "full_success_rate": lambda row: float(row["primary_score"] >= FULL_SUCCESS),
        "steps": lambda row: float(row["steps"]),
        "chat_tokens": lambda row: float(chat_tokens(row)),
    }
    report = {"episode_count": len(baseline_by_key), "task_count": len(task_ids)}
    for offset, (name, value) in enumerate(metrics.items()):
        task_differences = []
        for sample_id in task_ids:
            keys = sorted(key for key in baseline_by_key if key[0] == sample_id)
            if len(keys) != 3:
                raise ValueError("every deployment task must have exactly three repeats")
            task_differences.append(
                fmean(
                    value(candidate_by_key[key]) - value(baseline_by_key[key])
                    for key in keys
                )
            )
        report[f"{name}_difference"] = fmean(task_differences)
        report[f"{name}_confidence_interval"] = bootstrap(
            task_differences, seed + offset
        )
    return report


def assignments(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path or label in parsed:
            raise ValueError("assignments must be unique LABEL=PATH values")
        parsed[label] = Path(raw_path)
    return parsed


def main(options: argparse.Namespace) -> int:
    runs = assignments(options.run)
    rows = {label: read_results(path) for label, path in runs.items()}
    comparisons = {}
    for value in options.compare:
        label, separator, pair = value.partition("=")
        baseline, comma, candidate = pair.partition(",")
        if not separator or not comma or baseline not in rows or candidate not in rows:
            raise ValueError("--compare expects LABEL=BASELINE,CANDIDATE")
        comparisons[label] = {
            "baseline": baseline,
            "candidate": candidate,
            **paired(rows[baseline], rows[candidate], options.bootstrap_seed),
        }
    report = {
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": options.bootstrap_seed,
        "runs": {label: aggregate(value) for label, value in rows.items()},
        "comparisons": comparisons,
    }
    write_json(options.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    raise SystemExit(main(parser.parse_args()))
