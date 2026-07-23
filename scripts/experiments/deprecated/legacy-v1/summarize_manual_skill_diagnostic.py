from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.summarize_webshop_scale import (
    aggregate,
    load_rows,
    paired_comparison,
    validate_rows,
)
from scripts.experiments.run.rollout_no_skill_train import write_json


CONDITIONS = {
    "noskill": ("webshop-scale-v1-pro-noskill", "no_skill"),
    "p50_skillx": ("webshop-scale-v1-pro-p50-skillx", "skillx"),
    "p50_success": (
        "webshop-scale-v1-pro-p50-success",
        "trace2tower_static",
    ),
    "manual_v1": (
        "webshop-scale-v1-pro-manual-skill-v1",
        "manual_skill",
    ),
    "manual_v2": (
        "webshop-scale-v1-pro-manual-skill-v2",
        "manual_skill",
    ),
}
COMPARISONS = {
    "manual_v1_vs_noskill": ("noskill", "manual_v1"),
    "manual_v2_vs_noskill": ("noskill", "manual_v2"),
    "manual_v2_vs_manual_v1": ("manual_v1", "manual_v2"),
    "manual_v2_vs_p50_skillx": ("p50_skillx", "manual_v2"),
    "manual_v2_vs_p50_success": ("p50_success", "manual_v2"),
}


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    expected = {
        (sample_id, int(repeat_id))
        for sample_id in evaluation["sample_ids"]
        for repeat_id in evaluation["repeat_ids"]
    }
    rows_by_label = {}
    conditions = {}
    for label, (run_id, method_dir) in CONDITIONS.items():
        rows, error_attempts = load_rows(options.runs_root, run_id, method_dir)
        validate_rows(rows, run_id, expected)
        rows_by_label[label] = rows
        conditions[label] = {
            "run_id": run_id,
            "aggregate": aggregate(rows),
            "error_attempt_count": error_attempts,
        }

    rng = np.random.default_rng(20260715)
    output = {
        "protocol_id": protocol["protocol_id"],
        "conditions": conditions,
        "comparisons": {
            name: paired_comparison(
                rows_by_label[baseline],
                rows_by_label[candidate],
                rng,
            )
            for name, (baseline, candidate) in COMPARISONS.items()
        },
    }
    write_json(options.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/webshop_scale_v1.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiments/webshop-scale-v1/"
            "manual-skill-diagnostic-summary.json"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
