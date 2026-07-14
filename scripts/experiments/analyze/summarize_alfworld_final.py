from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
import pyarrow.parquet as parquet

from scripts.experiments.run.rollout_no_skill_train import write_json


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class Condition:
    model: str
    label: str
    method_dir: str

    @property
    def run_id(self) -> str:
        return f"alfworld-final-v1-{self.model}-{self.label}"


CONDITIONS = tuple(
    Condition(model, label, method_dir)
    for model in ("flash", "pro")
    for label, method_dir in (
        ("noskill", "no_skill"),
        ("flat", "flat_skill_summary"),
        ("skillx", "skillx"),
        ("success", "trace2tower_static"),
        ("mixed", "trace2tower_static"),
        ("success_mid", "trace2tower_static"),
        ("mixed_mid", "trace2tower_static"),
    )
)

COMPARISONS = (
    ("success_vs_noskill", "noskill", "success"),
    ("mixed_vs_success", "success", "mixed"),
    ("success_high_effect", "success_mid", "success"),
    ("mixed_high_effect", "mixed_mid", "mixed"),
    ("flat_vs_noskill", "noskill", "flat"),
    ("skillx_vs_noskill", "noskill", "skillx"),
    ("success_vs_flat", "flat", "success"),
    ("success_vs_skillx", "skillx", "success"),
)


def load_rows(root: Path, condition: Condition) -> tuple[list[dict], int]:
    method_root = (
        root / condition.run_id / "alfworld" / "test" / condition.method_dir
    )
    rows = [
        json.loads(line)
        for path in sorted(method_root.glob("shard-*/results.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    error_attempts = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in method_root.glob("shard-*/errors.jsonl")
    )
    return rows, error_attempts


def validate_rows(
    rows: list[dict], condition: Condition, expected_keys: set[tuple[str, int]]
) -> None:
    keys = [(str(row["sample_id"]), int(row["repeat_id"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError(f"{condition.run_id} does not have complete unique coverage")
    if any(
        row["run_id"] != condition.run_id
        or row["benchmark"] != "alfworld"
        or row["split"] != "test"
        or row["error"] is not None
        for row in rows
    ):
        raise ValueError(f"{condition.run_id} contains a scope or result mismatch")


def aggregate(rows: list[dict]) -> dict:
    successful = [row for row in rows if bool(row["success"])]
    total_steps = sum(int(row["steps"]) for row in rows)
    return {
        "episode_count": len(rows),
        "task_count": len({row["sample_id"] for row in rows}),
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows),
        "completion_rate": fmean(row["finish_reason"] == "completed" for row in rows),
        "mean_steps": fmean(int(row["steps"]) for row in rows),
        "mean_success_steps": (
            fmean(int(row["steps"]) for row in successful) if successful else None
        ),
        "invalid_action_rate": sum(int(row["invalid_actions"]) for row in rows)
        / max(total_steps, 1),
        "mean_chat_input_tokens": fmean(int(row["chat_input_tokens"]) for row in rows),
        "mean_chat_output_tokens": fmean(int(row["chat_output_tokens"]) for row in rows),
        "mean_skill_context_chars": fmean(
            int(row["skill_context_chars"]) for row in rows
        ),
    }


def task_families(dataset_root: Path) -> dict[str, str]:
    table = parquet.read_table(
        dataset_root / "valid_unseen.parquet", columns=["extra_info"]
    )
    return {
        f"alfworld:valid_unseen:{item['task_id']}": item["task_family"]
        for item in table.column(0).combine_chunks().to_pylist()
    }


def task_values(rows: list[dict], field) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(field(row)))
    if any(len(values) != 3 for values in grouped.values()):
        raise ValueError("every final-test task must have exactly three repeats")
    return {sample_id: fmean(values) for sample_id, values in grouped.items()}


def paired_comparison(
    baseline: list[dict], candidate: list[dict], rng: np.random.Generator
) -> dict:
    metrics = {
        "success_rate": lambda row: bool(row["success"]),
        "mean_steps": lambda row: int(row["steps"]),
        "mean_chat_input_tokens": lambda row: int(row["chat_input_tokens"]),
    }
    output = {}
    for name, value in metrics.items():
        before = task_values(baseline, value)
        after = task_values(candidate, value)
        task_ids = sorted(before)
        differences = np.asarray(
            [after[task_id] - before[task_id] for task_id in task_ids],
            dtype=np.float64,
        )
        indices = rng.integers(
            0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences))
        )
        estimates = differences[indices].mean(axis=1)
        output[name] = {
            "difference": float(differences.mean()),
            "ci95": [
                float(np.quantile(estimates, 0.025)),
                float(np.quantile(estimates, 0.975)),
            ],
        }
    return output


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    final_test = protocol["test"]
    expected_keys = {
        (sample_id, int(repeat_id))
        for sample_id in final_test["sample_ids"]
        for repeat_id in final_test["repeat_ids"]
    }
    families = task_families(options.dataset_root)
    rows_by_condition = {}
    output = {"protocol_id": protocol["protocol_id"], "conditions": {}, "comparisons": {}}
    for condition in CONDITIONS:
        rows, error_attempts = load_rows(options.runs_root, condition)
        validate_rows(rows, condition, expected_keys)
        rows_by_condition[(condition.model, condition.label)] = rows
        grouped = defaultdict(list)
        for row in rows:
            grouped[families[str(row["sample_id"])]].append(row)
        output["conditions"][condition.run_id] = {
            "aggregate": aggregate(rows),
            "by_family": {
                family: aggregate(grouped[family]) for family in sorted(grouped)
            },
            "error_attempt_count": error_attempts,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for model in ("flash", "pro"):
        output["comparisons"][model] = {
            name: paired_comparison(
                rows_by_condition[(model, baseline)],
                rows_by_condition[(model, candidate)],
                rng,
            )
            for name, baseline, candidate in COMPARISONS
        }
    write_json(options.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/alfworld_protocol_v1.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiments/alfworld/official/analysis/final-summary.json"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
