from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.experiments.run.rollout_no_skill_train import write_json


BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260715
METHOD_DIRS = {
    "noskill": "no_skill",
    "flat": "flat_skill_summary",
    "skillx": "skillx",
    "success": "trace2tower_static",
    "mixed": "trace2tower_static",
}


def run_id(pool: str | None, method: str) -> str:
    label = "noskill" if pool is None else f"{pool}-{method}"
    return f"webshop-scale-v1-pro-{label}"


def load_rows(root: Path, current_run_id: str, method_dir: str) -> tuple[list[dict], int]:
    method_root = root / current_run_id / "webshop" / "test" / method_dir
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
    rows: list[dict], current_run_id: str, expected: set[tuple[str, int]]
) -> None:
    keys = [(str(row["sample_id"]), int(row["repeat_id"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"{current_run_id} does not have complete unique coverage")
    if any(
        row["run_id"] != current_run_id
        or row["benchmark"] != "webshop"
        or row["split"] != "test"
        or row["error"] is not None
        for row in rows
    ):
        raise ValueError(f"{current_run_id} contains a scope or result mismatch")


def aggregate(rows: list[dict]) -> dict:
    total_steps = sum(int(row["steps"]) for row in rows)
    return {
        "episode_count": len(rows),
        "task_count": len({row["sample_id"] for row in rows}),
        "mean_reward": fmean(float(row["primary_score"]) for row in rows),
        "full_success_rate": fmean(
            float(row["primary_score"]) >= 0.999 for row in rows
        ),
        "completion_rate": fmean(row["finish_reason"] == "completed" for row in rows),
        "mean_steps": fmean(int(row["steps"]) for row in rows),
        "invalid_action_rate": sum(int(row["invalid_actions"]) for row in rows)
        / max(total_steps, 1),
        "mean_chat_input_tokens": fmean(
            int(row["chat_input_tokens"]) for row in rows
        ),
        "mean_chat_output_tokens": fmean(
            int(row["chat_output_tokens"]) for row in rows
        ),
        "mean_skill_context_chars": fmean(
            int(row["skill_context_chars"]) for row in rows
        ),
    }


def task_values(rows: list[dict], field) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(float(field(row)))
    if any(len(values) != 3 for values in grouped.values()):
        raise ValueError("every scale-study task must have exactly three repeats")
    return {sample_id: fmean(values) for sample_id, values in grouped.items()}


def paired_comparison(
    baseline: list[dict], candidate: list[dict], rng: np.random.Generator
) -> dict:
    metrics = {
        "mean_reward": lambda row: float(row["primary_score"]),
        "full_success_rate": lambda row: float(row["primary_score"]) >= 0.999,
        "mean_steps": lambda row: int(row["steps"]),
        "mean_chat_input_tokens": lambda row: int(row["chat_input_tokens"]),
    }
    output = {}
    for name, value in metrics.items():
        before = task_values(baseline, value)
        after = task_values(candidate, value)
        task_ids = sorted(before)
        if task_ids != sorted(after):
            raise ValueError("paired scale comparison uses different task sets")
        differences = np.asarray(
            [after[sample_id] - before[sample_id] for sample_id in task_ids],
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
    scale = json.loads(options.protocol.read_text(encoding="utf-8"))
    evaluation = scale["evaluation"]
    expected = {
        (sample_id, int(repeat_id))
        for sample_id in evaluation["sample_ids"]
        for repeat_id in evaluation["repeat_ids"]
    }
    pools = options.pool or ["p50", "p100", "p200"]
    rows_by_key = {}
    conditions = {}

    no_skill_id = run_id(None, "noskill")
    no_skill_rows, no_skill_errors = load_rows(
        options.runs_root, no_skill_id, METHOD_DIRS["noskill"]
    )
    validate_rows(no_skill_rows, no_skill_id, expected)
    rows_by_key[(None, "noskill")] = no_skill_rows
    conditions[no_skill_id] = {
        "aggregate": aggregate(no_skill_rows),
        "error_attempt_count": no_skill_errors,
    }

    for pool in pools:
        for method in ("flat", "skillx", "success", "mixed"):
            current_run_id = run_id(pool, method)
            rows, error_attempts = load_rows(
                options.runs_root, current_run_id, METHOD_DIRS[method]
            )
            validate_rows(rows, current_run_id, expected)
            rows_by_key[(pool, method)] = rows
            conditions[current_run_id] = {
                "aggregate": aggregate(rows),
                "error_attempt_count": error_attempts,
            }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {}
    for pool in pools:
        comparisons[pool] = {
            "flat_vs_noskill": paired_comparison(
                no_skill_rows, rows_by_key[(pool, "flat")], rng
            ),
            "skillx_vs_noskill": paired_comparison(
                no_skill_rows, rows_by_key[(pool, "skillx")], rng
            ),
            "success_vs_noskill": paired_comparison(
                no_skill_rows, rows_by_key[(pool, "success")], rng
            ),
            "mixed_vs_noskill": paired_comparison(
                no_skill_rows, rows_by_key[(pool, "mixed")], rng
            ),
            "success_vs_flat": paired_comparison(
                rows_by_key[(pool, "flat")], rows_by_key[(pool, "success")], rng
            ),
            "success_vs_skillx": paired_comparison(
                rows_by_key[(pool, "skillx")], rows_by_key[(pool, "success")], rng
            ),
            "mixed_vs_success": paired_comparison(
                rows_by_key[(pool, "success")], rows_by_key[(pool, "mixed")], rng
            ),
        }

    scale_comparisons = {}
    for before, after in zip(pools, pools[1:]):
        scale_comparisons[f"{after}_vs_{before}"] = {
            method: paired_comparison(
                rows_by_key[(before, method)], rows_by_key[(after, method)], rng
            )
            for method in ("flat", "skillx", "success", "mixed")
        }

    output = {
        "protocol_id": scale["protocol_id"],
        "pools": pools,
        "conditions": conditions,
        "within_pool_comparisons": comparisons,
        "scale_comparisons": scale_comparisons,
    }
    write_json(options.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="append", choices=("p50", "p100", "p200"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/webshop_scale_v1.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/webshop-scale-v1/evaluation-summary.json"),
    )
    raise SystemExit(main(parser.parse_args()))
