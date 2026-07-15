from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_first_completion(run_dir: Path) -> tuple[dict[tuple[str, int], dict], dict]:
    rows_by_key: dict[tuple[str, int], dict] = {}
    duplicates: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path in sorted(run_dir.glob("**/results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (str(row["sample_id"]), int(row["repeat_id"]))
            if key in rows_by_key:
                duplicates[key].append(
                    {
                        "result_path": path.as_posix(),
                        "primary_score": row["primary_score"],
                        "steps": row["steps"],
                    }
                )
            else:
                rows_by_key[key] = row
    score_disagreements = sum(
        row["primary_score"] != rows_by_key[key]["primary_score"]
        for key, rows in duplicates.items()
        for row in rows
    )
    step_disagreements = sum(
        row["steps"] != rows_by_key[key]["steps"]
        for key, rows in duplicates.items()
        for row in rows
    )
    return rows_by_key, {
        "duplicate_key_count": len(duplicates),
        "discarded_row_count": sum(len(rows) for rows in duplicates.values()),
        "primary_score_disagreement_count": score_disagreements,
        "step_disagreement_count": step_disagreements,
        "keys": [
            {
                "sample_id": sample_id,
                "repeat_id": repeat_id,
                "selected": {
                    "primary_score": rows_by_key[(sample_id, repeat_id)]["primary_score"],
                    "steps": rows_by_key[(sample_id, repeat_id)]["steps"],
                },
                "discarded": rows,
            }
            for (sample_id, repeat_id), rows in sorted(duplicates.items())
        ],
    }


def paired_interval(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260715)
    samples = rng.choice(
        differences,
        size=(10000, differences.size),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.975)))


def summarize(rows: dict[tuple[str, int], dict]) -> dict:
    values = list(rows.values())
    return {
        "task_count": len(values),
        "mean_reward": float(np.mean([row["primary_score"] for row in values])),
        "full_success_rate": float(
            np.mean([row["primary_score"] >= 0.999 for row in values])
        ),
        "mean_steps": float(np.mean([row["steps"] for row in values])),
        "mean_invalid_actions": float(np.mean([row["invalid_actions"] for row in values])),
        "mean_input_tokens": float(np.mean([row["input_tokens"] for row in values])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-run", type=Path, required=True)
    parser.add_argument("--noskill-run", type=Path, required=True)
    parser.add_argument("--previous-tower-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    structural, structural_audit = read_first_completion(options.structural_run)
    noskill, noskill_audit = read_first_completion(options.noskill_run)
    previous_tower, previous_tower_audit = read_first_completion(options.previous_tower_run)
    keys = set(structural)
    if keys != set(noskill) or keys != set(previous_tower) or len(keys) != 100:
        raise ValueError("paired Test-A runs must cover the same 100 unique task keys")

    ordered_keys = tuple(sorted(keys))
    structural_scores = np.array([structural[key]["primary_score"] for key in ordered_keys])
    noskill_scores = np.array([noskill[key]["primary_score"] for key in ordered_keys])
    previous_scores = np.array(
        [previous_tower[key]["primary_score"] for key in ordered_keys]
    )
    payload = {
        "protocol_id": "webshop-test-a-refinement-v1",
        "split": "test",
        "agent_model": "deepseek-v4-flash",
        "repeat_id": 0,
        "direct_mid_cap": 8,
        "duplicate_policy": "first_completion_in_sorted_shard_file_order",
        "runs": {
            "structural_v1": options.structural_run.as_posix(),
            "noskill": options.noskill_run.as_posix(),
            "previous_tower_v0": options.previous_tower_run.as_posix(),
        },
        "audits": {
            "structural_v1": structural_audit,
            "noskill": noskill_audit,
            "previous_tower_v0": previous_tower_audit,
        },
        "summary": {
            "structural_v1": summarize(structural),
            "noskill": summarize(noskill),
            "previous_tower_v0": summarize(previous_tower),
        },
        "paired_deltas": {
            "structural_v1_minus_noskill": {
                "mean_reward": float(np.mean(structural_scores - noskill_scores)),
                "bootstrap_ci95": paired_interval(structural_scores - noskill_scores),
            },
            "structural_v1_minus_previous_tower_v0": {
                "mean_reward": float(np.mean(structural_scores - previous_scores)),
                "bootstrap_ci95": paired_interval(structural_scores - previous_scores),
            },
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
