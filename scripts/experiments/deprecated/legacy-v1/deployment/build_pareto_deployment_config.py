from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_yaml
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot


def main(options: argparse.Namespace) -> int:
    base = load_yaml(options.base_config)
    refinement = load_yaml(options.refinement_config)
    report = json.loads(options.refinement_report.read_text(encoding="utf-8"))
    tower = TowerSnapshot.from_record(
        json.loads(options.tower.read_text(encoding="utf-8"))
    )
    if base.get("method") != "trace2tower_static":
        raise ValueError("deployment config requires a Static Tower base config")
    if (
        report.get("ranking_status") != "complete"
        or report.get("tower_snapshot_id") != tower.snapshot_id
    ):
        raise ValueError("refinement report does not bind complete lifecycle evidence")
    epsilon = float(refinement["status_tie_epsilon"])
    config = {
        **base,
        "lifecycle_report": options.refinement_report.as_posix(),
        "status_tie_epsilon": epsilon,
    }
    write_yaml(options.output, config)
    audit = {
        "tower_snapshot_id": tower.snapshot_id,
        "base_config": options.base_config.as_posix(),
        "base_config_sha256": hashlib.sha256(options.base_config.read_bytes()).hexdigest(),
        "refinement_report": options.refinement_report.as_posix(),
        "refinement_report_sha256": hashlib.sha256(
            options.refinement_report.read_bytes()
        ).hexdigest(),
        "downweighted_skill_ids": sorted(
            update["skill_id"] for update in report["downweight"]
        ),
        "status_tie_epsilon": epsilon,
        "output": options.output.as_posix(),
    }
    options.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(yaml.safe_dump(audit, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--refinement-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/experiments/trace2tower_static_diverse3.yaml"),
    )
    parser.add_argument(
        "--refinement-config",
        type=Path,
        default=Path("configs/experiments/pareto_refinement.yaml"),
    )
    raise SystemExit(main(parser.parse_args()))
