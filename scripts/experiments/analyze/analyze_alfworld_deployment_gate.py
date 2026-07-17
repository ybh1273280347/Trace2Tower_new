from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.core.manifests import read_manifest
from trace2tower.methods.trace2tower.deployment_optimization.pareto import rank_fronts


def main(options: argparse.Namespace) -> int:
    manifest = read_manifest(options.manifest)
    expected_ids = {entry.sample_id for entry in manifest}
    baseline = _read_run(options.baseline, expected_ids)
    candidates = dict(_candidate(value) for value in options.candidate)
    if len(candidates) != len(options.candidate):
        raise ValueError("gate candidate names must be unique")

    rng = np.random.default_rng(options.bootstrap_seed)
    baseline_values = tuple(baseline[sample_id] for sample_id in sorted(expected_ids))
    reports = {}
    for name, path in sorted(candidates.items()):
        candidate = _read_run(path, expected_ids)
        candidate_values = tuple(candidate[sample_id] for sample_id in sorted(expected_ids))
        success_differences = np.asarray(
            [float(right["success"]) - float(left["success"]) for left, right in zip(baseline_values, candidate_values)]
        )
        guarded_differences = np.asarray(
            [_guarded_step(left, right) for left, right in zip(baseline_values, candidate_values)]
        )
        success_bootstrap = _bootstrap_means(success_differences, rng, options.bootstrap_samples)
        guarded_bootstrap = _bootstrap_means(guarded_differences, rng, options.bootstrap_samples)
        success_delta = float(success_differences.mean())
        success_lower = float(np.quantile(success_bootstrap, 0.05))
        guarded_mean = float(guarded_differences.mean())
        reports[name] = {
            "path": path.as_posix(),
            "result_tree_sha256": _result_tree_hash(path),
            "success_rate": float(np.mean([row["success"] for row in candidate_values])),
            "mean_steps": float(np.mean([row["steps"] for row in candidate_values])),
            "paired_success_delta": success_delta,
            "paired_success_one_sided_95_lower": success_lower,
            "guarded_step_saving": guarded_mean,
            "guarded_step_one_sided_95_lower": float(np.quantile(guarded_bootstrap, 0.05)),
            "paired_wins": int((success_differences > 0).sum()),
            "paired_losses": int((success_differences < 0).sum()),
            "paired_ties": int((success_differences == 0).sum()),
            "gate_pass": success_delta >= 0 and success_lower >= options.noninferiority_margin,
        }

    passing = {name: report for name, report in reports.items() if report["gate_pass"]}
    ranks = rank_fronts(
        {
            name: (
                report["success_rate"],
                report["paired_success_delta"],
                report["guarded_step_saving"],
            )
            for name, report in passing.items()
        }
    )
    for name, rank in ranks.items():
        reports[name]["pareto_front_rank"] = rank
    selected = min(
        (name for name in passing if ranks[name] == 1),
        key=lambda name: (
            -reports[name]["guarded_step_one_sided_95_lower"],
            options.action_count.get(name, 999),
            name,
        ),
        default=None,
    )
    payload = {
        "protocol_id": "alfworld-deployment-optimization-v1-gate",
        "manifest": {
            "path": options.manifest.as_posix(),
            "sha256": hashlib.sha256(options.manifest.read_bytes()).hexdigest(),
            "task_count": len(expected_ids),
        },
        "baseline": {
            "path": options.baseline.as_posix(),
            "result_tree_sha256": _result_tree_hash(options.baseline),
            "success_rate": float(np.mean([row["success"] for row in baseline_values])),
            "mean_steps": float(np.mean([row["steps"] for row in baseline_values])),
        },
        "gate_contract": {
            "paired_success_point_minimum": 0.0,
            "one_sided_95_lower_minimum": options.noninferiority_margin,
            "bootstrap_samples": options.bootstrap_samples,
            "bootstrap_seed": options.bootstrap_seed,
            "selection_tie_break": [
                "guarded_step_one_sided_95_lower",
                "fewer_actions",
                "candidate_name",
            ],
        },
        "candidates": reports,
        "selected": selected,
    }
    write_json(options.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def _read_run(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    rows = {}
    for result_path in sorted(path.rglob("results.jsonl")):
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"gate run contains duplicate task: {sample_id}")
            if int(row["repeat_id"]) != 0 or row["split"] != "train":
                raise ValueError("gate run violates the frozen split/repeat contract")
            rows[sample_id] = row
    if set(rows) != expected_ids:
        raise ValueError(
            f"gate run coverage differs from manifest: expected={len(expected_ids)}, observed={len(rows)}"
        )
    return rows


def _guarded_step(baseline: dict, candidate: dict) -> float:
    raw = (float(baseline["steps"]) - float(candidate["steps"])) / max(
        float(baseline["steps"]), 1.0
    )
    return min(raw, 0.0) if candidate["success"] < baseline["success"] else raw


def _bootstrap_means(values: np.ndarray, rng, samples: int) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    return values[indices].mean(axis=1)


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError("candidate must use NAME=PATH")
    return name, Path(path)


def _result_tree_hash(path: Path) -> str:
    records = [
        f"{result_path.as_posix()}\0{hashlib.sha256(result_path.read_bytes()).hexdigest()}"
        for result_path in sorted(path.rglob("results.jsonl"))
    ]
    return hashlib.sha256("\n".join(records).encode()).hexdigest()


def _action_count(value: str) -> tuple[str, int]:
    name, separator, count = value.partition("=")
    if not separator or not name or not count:
        raise ValueError("action count must use NAME=COUNT")
    return name, int(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--action-count", action="append", default=[])
    parser.add_argument("--noninferiority-margin", type=float, default=-0.03)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.action_count = dict(
        _action_count(value) for value in options.action_count
    )
    raise SystemExit(main(options))
