from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np


FULL_SUCCESS_THRESHOLD = 0.999
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_715


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_manifest(path: Path) -> set[tuple[str, int]]:
    records = read_jsonl(path)
    keys = {(record["sample_id"], int(record["repeat_id"])) for record in records}
    if len(keys) != len(records):
        raise ValueError("manifest contains duplicate sample/repeat keys")
    return keys


def load_run(
    path: Path,
    *,
    expected_run_id: str,
    expected_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int], dict], dict]:
    result_files = sorted(path.rglob("results.jsonl"))
    if not result_files:
        raise ValueError(f"no results.jsonl files under {path}")
    records = [record for result_file in result_files for record in read_jsonl(result_file)]
    keys = [(record["sample_id"], int(record["repeat_id"])) for record in records]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    observed_keys = set(keys)
    missing_keys = sorted(expected_keys - observed_keys)
    unexpected_keys = sorted(observed_keys - expected_keys)
    scope_errors = [
        {
            "sample_id": record.get("sample_id"),
            "run_id": record.get("run_id"),
            "benchmark": record.get("benchmark"),
            "split": record.get("split"),
            "method": record.get("method"),
            "repeat_id": record.get("repeat_id"),
        }
        for record in records
        if record.get("run_id") != expected_run_id
        or record.get("benchmark") != "alfworld"
        or record.get("split") != "test"
        or record.get("method") != "trace2tower"
        or int(record.get("repeat_id", -1)) != 0
    ]
    success_disagreements = [
        record["sample_id"]
        for record in records
        if bool(record["success"])
        != (float(record["primary_score"]) >= FULL_SUCCESS_THRESHOLD)
    ]
    if duplicate_keys or missing_keys or unexpected_keys or scope_errors or success_disagreements:
        raise ValueError(
            json.dumps(
                {
                    "run": expected_run_id,
                    "duplicate_keys": duplicate_keys,
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "scope_errors": scope_errors,
                    "success_disagreements": success_disagreements,
                },
                ensure_ascii=False,
            )
        )
    by_key = {key: record for key, record in zip(keys, records, strict=True)}
    audit = {
        "run_id": expected_run_id,
        "run_path": path.as_posix(),
        "expected_count": len(expected_keys),
        "result_count": len(records),
        "unique_key_count": len(by_key),
        "duplicate_key_count": 0,
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "scope_error_count": 0,
        "success_disagreement_count": 0,
        "result_files": {
            result_file.as_posix(): hashlib.sha256(result_file.read_bytes()).hexdigest()
            for result_file in result_files
        },
    }
    return by_key, audit


def summarize(rows: dict[tuple[str, int], dict]) -> dict:
    records = list(rows.values())
    return {
        "episode_count": len(records),
        "success_count": sum(bool(record["success"]) for record in records),
        "success_rate": fmean(bool(record["success"]) for record in records),
        "mean_steps": fmean(int(record["steps"]) for record in records),
        "mean_invalid_actions": fmean(int(record["invalid_actions"]) for record in records),
        "mean_input_tokens": fmean(int(record["input_tokens"]) for record in records),
        "mean_skill_context_chars": fmean(
            int(record["skill_context_chars"]) for record in records
        ),
        "finish_reason_counts": dict(
            sorted(Counter(record["finish_reason"] for record in records).items())
        ),
    }


def bootstrap_interval(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 512):
        end = min(start + 512, BOOTSTRAP_SAMPLES)
        indices = rng.integers(
            0,
            differences.size,
            size=(end - start, differences.size),
        )
        means[start:end] = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def mcnemar_exact(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare(
    full: dict[tuple[str, int], dict],
    ablation: dict[tuple[str, int], dict],
) -> dict:
    keys = tuple(sorted(full))
    if set(keys) != set(ablation):
        raise ValueError("paired comparison requires identical keys")
    full_success = np.asarray([bool(full[key]["success"]) for key in keys], dtype=float)
    ablation_success = np.asarray(
        [bool(ablation[key]["success"]) for key in keys], dtype=float
    )
    differences = full_success - ablation_success
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    return {
        "direction": "full_minus_ablation",
        "success_rate_difference": float(differences.mean()),
        "bootstrap_ci95": bootstrap_interval(differences),
        "full_wins": wins,
        "full_losses": losses,
        "ties": int(np.sum(differences == 0)),
        "mcnemar_exact_p": mcnemar_exact(wins, losses),
        "mean_step_difference": fmean(
            int(full[key]["steps"]) - int(ablation[key]["steps"]) for key in keys
        ),
        "mean_invalid_action_difference": fmean(
            int(full[key]["invalid_actions"])
            - int(ablation[key]["invalid_actions"])
            for key in keys
        ),
        "mean_input_token_difference": fmean(
            int(full[key]["input_tokens"]) - int(ablation[key]["input_tokens"])
            for key in keys
        ),
        "mean_skill_context_char_difference": fmean(
            int(full[key]["skill_context_chars"])
            - int(ablation[key]["skill_context_chars"])
            for key in keys
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--no-transition-run", type=Path, required=True)
    parser.add_argument("--no-outcome-run", type=Path, required=True)
    parser.add_argument("--no-contrastive-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    expected_keys = load_manifest(options.manifest)
    run_paths = {
        "full": options.full_run,
        "no_transition": options.no_transition_run,
        "no_outcome": options.no_outcome_run,
        "no_contrastive": options.no_contrastive_run,
    }
    run_ids = {
        "full": "alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0",
        "no_transition": "alfworld-ablation-v1-no-transition-flash-r0",
        "no_outcome": "alfworld-ablation-v1-no-outcome-flash-r0",
        "no_contrastive": "alfworld-ablation-v1-no-contrastive-flash-r0",
    }
    rows = {}
    audits = {}
    for name, path in run_paths.items():
        rows[name], audits[name] = load_run(
            path,
            expected_run_id=run_ids[name],
            expected_keys=expected_keys,
        )

    payload = {
        "protocol_id": "alfworld-trace2tower-component-ablation-v1",
        "benchmark": "alfworld",
        "split": "valid_unseen",
        "repeat_id": 0,
        "agent_model": "deepseek-v4-flash",
        "success_threshold": FULL_SUCCESS_THRESHOLD,
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "unit": "task",
        },
        "manifest": {
            "path": options.manifest.as_posix(),
            "sha256": hashlib.sha256(options.manifest.read_bytes()).hexdigest(),
            "expected_count": len(expected_keys),
        },
        "audits": audits,
        "summary": {name: summarize(run_rows) for name, run_rows in rows.items()},
        "full_minus_ablation": {
            name: compare(rows["full"], rows[name])
            for name in ("no_transition", "no_outcome", "no_contrastive")
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
