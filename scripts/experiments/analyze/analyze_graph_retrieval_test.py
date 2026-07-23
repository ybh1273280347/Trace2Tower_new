from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.analyze_refinement_test import (
    paired_interval,
    read_first_completion,
    summarize,
)
from trace2tower.methods.trace2tower.adapters.webshop.events import infer_webshop_page_type


def read_retrieval_audit(run_dir: Path) -> dict:
    trajectories = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in run_dir.glob("**/trajectories/*.json")
    ]
    if len(trajectories) != 100:
        raise ValueError("graph retrieval audit requires 100 trajectories")
    context_counts = []
    high_counts = Counter()
    high_by_page: dict[str, Counter] = defaultdict(Counter)
    for trajectory in trajectories:
        for step in trajectory["steps"]:
            skill_ids = step.get("retrieved_context_skill_ids", ())
            context_counts.append(len(skill_ids))
            high_id = (
                str(skill_ids[0]) if skill_ids and str(skill_ids[0]).startswith("high_") else "none"
            )
            page = infer_webshop_page_type(step["observation"]).value
            high_counts[high_id] += 1
            high_by_page[page][high_id] += 1
    return {
        "trajectory_count": len(trajectories),
        "step_count": len(context_counts),
        "mean_context_skill_count": float(np.mean(context_counts)),
        "max_context_skill_count": max(context_counts),
        "high_selection_counts": dict(sorted(high_counts.items())),
        "high_selection_by_page": {
            page: dict(sorted(counts.items())) for page, counts in sorted(high_by_page.items())
        },
    }


def count_error_attempts(run_dir: Path) -> int:
    return sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in run_dir.glob("**/errors.jsonl")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "noskill",
        "v0",
        "legacy_cap8",
        "legacy_cap3",
        "graph_cap3",
        "graph_cap8",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    run_paths = {
        name: getattr(options, f"{name}_run")
        for name in (
            "noskill",
            "v0",
            "legacy_cap8",
            "legacy_cap3",
            "graph_cap3",
            "graph_cap8",
        )
    }
    rows = {}
    audits = {}
    for name, path in run_paths.items():
        rows[name], audits[name] = read_first_completion(path)
    keys = set(rows["noskill"])
    if len(keys) != 100 or any(set(run_rows) != keys for run_rows in rows.values()):
        raise ValueError("all Test-A runs must cover the same 100 unique keys")

    ordered_keys = tuple(sorted(keys))
    scores = {
        name: np.array([run_rows[key]["primary_score"] for key in ordered_keys])
        for name, run_rows in rows.items()
    }
    comparisons = (
        ("graph_cap3", "noskill"),
        ("graph_cap3", "v0"),
        ("graph_cap3", "legacy_cap3"),
        ("graph_cap3", "legacy_cap8"),
        ("graph_cap8", "legacy_cap8"),
        ("graph_cap3", "graph_cap8"),
    )
    payload = {
        "protocol_id": "webshop-test-a-graph-retrieval-v1",
        "agent_model": "deepseek-v4-flash",
        "repeat_id": 0,
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
        "rate_limit_error_attempts": {
            name: count_error_attempts(run_paths[name]) for name in ("graph_cap3", "graph_cap8")
        },
        "retrieval_audits": {
            name: read_retrieval_audit(run_paths[name]) for name in ("graph_cap3", "graph_cap8")
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
