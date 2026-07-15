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


def paired_comparison(
    treatment: dict[tuple[str, int], dict],
    control: dict[tuple[str, int], dict],
) -> dict:
    if len(treatment) != 100 or set(treatment) != set(control):
        raise ValueError("mechanism comparison requires the same 100 task keys")
    keys = tuple(sorted(treatment))
    metrics = {
        "reward": "primary_score",
        "steps": "steps",
        "invalid_actions": "invalid_actions",
        "input_tokens": "input_tokens",
    }
    comparisons = {}
    for metric, field in metrics.items():
        differences = np.array(
            [treatment[key][field] - control[key][field] for key in keys],
            dtype=np.float64,
        )
        comparisons[metric] = {
            "mean": float(np.mean(differences)),
            "bootstrap_ci95": paired_interval(differences),
        }
    success_differences = np.array(
        [
            (treatment[key]["primary_score"] >= 0.999)
            - (control[key]["primary_score"] >= 0.999)
            for key in keys
        ],
        dtype=np.float64,
    )
    comparisons["full_success_rate"] = {
        "mean": float(np.mean(success_differences)),
        "bootstrap_ci95": paired_interval(success_differences),
    }
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-graph-run", type=Path, required=True)
    parser.add_argument("--no-mixed-graph-run", type=Path, required=True)
    parser.add_argument("--legacy-full-run", type=Path, required=True)
    parser.add_argument("--semantic-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    runs = {
        "mixed_graph_cap3": options.mixed_graph_run,
        "no_mixed_graph_cap3": options.no_mixed_graph_run,
        "legacy_full_cap8": options.legacy_full_run,
        "semantic_cap8": options.semantic_run,
    }
    rows = {name: read_first_completion(path)[0] for name, path in runs.items()}
    payload = {
        "protocol_id": "webshop-final-mechanism-ablations-v1",
        "agent_model": "deepseek-v4-flash",
        "repeat_id": 0,
        "runs": {name: path.as_posix() for name, path in runs.items()},
        "summary": {name: summarize(run_rows) for name, run_rows in rows.items()},
        "comparisons": {
            "no_mixed_minus_mixed": paired_comparison(
                rows["no_mixed_graph_cap3"], rows["mixed_graph_cap3"]
            ),
            "legacy_full_minus_semantic": paired_comparison(
                rows["legacy_full_cap8"], rows["semantic_cap8"]
            ),
        },
        "comparison_contracts": {
            "no_mixed_minus_mixed": {
                "shared": "P100 rollout pool, graph-aware retrieval, total Mid cap3",
                "variable": "mixed partial/failure evidence",
            },
            "legacy_full_minus_semantic": {
                "shared": "P100 mixed pool, nine Mid clusters, native renderer, legacy retrieval cap8",
                "variable": "signed relational EigenTrace graph and High induction",
            },
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
