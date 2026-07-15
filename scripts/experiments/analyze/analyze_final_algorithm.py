from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.analyze_graph_retrieval_test import (
    count_error_attempts,
    read_retrieval_audit,
)
from scripts.experiments.analyze.analyze_refinement_test import (
    paired_interval,
    read_first_completion,
    summarize,
)


def analyze_scope(run_paths: dict[str, Path], comparisons: tuple[tuple[str, str], ...]) -> dict:
    rows = {}
    audits = {}
    for name, path in run_paths.items():
        rows[name], audits[name] = read_first_completion(path)
    keys = set(next(iter(rows.values())))
    if len(keys) != 100 or any(set(run_rows) != keys for run_rows in rows.values()):
        raise ValueError("all runs in one scope must cover the same 100 task keys")
    ordered_keys = tuple(sorted(keys))
    scores = {
        name: np.array([run_rows[key]["primary_score"] for key in ordered_keys])
        for name, run_rows in rows.items()
    }
    return {
        "runs": {name: path.as_posix() for name, path in run_paths.items()},
        "summary": {name: summarize(run_rows) for name, run_rows in rows.items()},
        "paired_deltas": {
            f"{left}_minus_{right}": {
                "mean_reward": float(np.mean(scores[left] - scores[right])),
                "bootstrap_ci95": paired_interval(scores[left] - scores[right]),
            }
            for left, right in comparisons
        },
        "result_audits": audits,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "test_a_final",
        "test_a_v0_graph",
        "test_a_noskill",
        "test_b_final",
        "test_b_noskill",
        "test_b_skillx",
        "test_b_legacy_v0",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    test_a_paths = {
        "final_t1": options.test_a_final_run,
        "v0_graph": options.test_a_v0_graph_run,
        "noskill": options.test_a_noskill_run,
    }
    test_b_paths = {
        "final_t1": options.test_b_final_run,
        "noskill": options.test_b_noskill_run,
        "skillx": options.test_b_skillx_run,
        "legacy_v0": options.test_b_legacy_v0_run,
    }
    payload = {
        "protocol_id": "webshop-final-graph-cap3-v1",
        "agent_model": "deepseek-v4-flash",
        "repeat_id": 0,
        "test_a_refinement_isolation": analyze_scope(
            test_a_paths,
            (("final_t1", "v0_graph"), ("final_t1", "noskill")),
        ),
        "test_b_robustness": analyze_scope(
            test_b_paths,
            (
                ("final_t1", "noskill"),
                ("final_t1", "skillx"),
                ("final_t1", "legacy_v0"),
            ),
        ),
        "final_run_error_attempts": {
            "test_a": count_error_attempts(options.test_a_final_run),
            "test_b": count_error_attempts(options.test_b_final_run),
        },
        "final_retrieval_audits": {
            "test_a": read_retrieval_audit(options.test_a_final_run),
            "test_b": read_retrieval_audit(options.test_b_final_run),
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
