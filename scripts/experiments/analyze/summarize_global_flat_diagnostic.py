from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.summarize_webshop_scale import (
    BOOTSTRAP_SEED,
    aggregate,
    load_rows,
    paired_comparison,
    validate_rows,
)
from scripts.experiments.run.rollout_no_skill_train import write_json

CONDITIONS = {
    "noskill": ("webshop-scale-v1-pro-noskill", "no_skill"),
    "old_flat": ("webshop-scale-v1-pro-p50-flat", "flat_skill_summary"),
    "skillx": ("webshop-scale-v1-pro-p50-skillx", "skillx"),
    "success_tower": (
        "webshop-scale-v1-pro-p50-success",
        "trace2tower_static",
    ),
    "global_flat_gpt54": (
        "webshop-scale-v1-pro-p50-flat-global-gpt54",
        "flat_skill_summary",
    ),
    "e2e_flat_gpt54_top1": (
        "webshop-scale-v1-pro-p50-flat-e2e-gpt54-top1",
        "flat_skill_summary",
    ),
}


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    expected = {
        (sample_id, int(repeat_id))
        for sample_id in evaluation["sample_ids"]
        for repeat_id in evaluation["repeat_ids"]
    }
    rows = {}
    conditions = {}
    for label, (run_id, method_dir) in CONDITIONS.items():
        current_rows, error_attempts = load_rows(
            options.runs_root,
            run_id,
            method_dir,
        )
        validate_rows(current_rows, run_id, expected)
        rows[label] = current_rows
        conditions[label] = {
            "run_id": run_id,
            "aggregate": aggregate(current_rows),
            "error_attempt_count": error_attempts,
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    candidate = rows["global_flat_gpt54"]
    comparisons = {
        f"global_flat_gpt54_vs_{baseline}": paired_comparison(
            rows[baseline],
            candidate,
            rng,
        )
        for baseline in ("noskill", "old_flat", "skillx", "success_tower")
    }
    e2e_candidate = rows["e2e_flat_gpt54_top1"]
    comparisons.update(
        {
            f"e2e_flat_gpt54_top1_vs_{baseline}": paired_comparison(
                rows[baseline],
                e2e_candidate,
                rng,
            )
            for baseline in (
                "noskill",
                "old_flat",
                "global_flat_gpt54",
                "skillx",
                "success_tower",
            )
        }
    )
    output = {
        "protocol_id": protocol["protocol_id"],
        "comparison_direction": "candidate named before _vs_ minus baseline",
        "conditions": conditions,
        "comparisons": comparisons,
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
            "artifacts/experiments/webshop-scale-v1/global-flat-gpt54-summary.json"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
