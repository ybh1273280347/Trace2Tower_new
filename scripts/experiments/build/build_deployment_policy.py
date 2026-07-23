from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot
from trace2tower.methods.trace2tower.deployment_optimization.policy import (
    build_deployment_policy,
)


def main(options: argparse.Namespace) -> int:
    report = json.loads(options.feedback_report.read_text(encoding="utf-8"))
    proposals = {
        item["primary_high_id"]: item for item in report.get("downweight_proposals", ())
    }
    if options.high_id not in proposals:
        raise ValueError("requested High is not a feedback downweight proposal")
    if options.target_tower:
        tower = TowerSnapshot.from_record(json.loads(options.target_tower.read_text(encoding="utf-8")))
        if options.high_id not in {card.skill_id for card in tower.high_cards}:
            raise ValueError("downweighted High does not exist in the target Tower")
        snapshot_id = tower.snapshot_id
    else:
        snapshot_id = str(report["base_snapshot_id"])
    policy = build_deployment_policy(
        snapshot_id,
        {options.high_id: options.score_penalty},
        hashlib.sha256(options.feedback_report.read_bytes()).hexdigest(),
    )
    write_json(options.output, policy.to_record())
    print(json.dumps(policy.to_record(), indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-report", type=Path, required=True)
    parser.add_argument("--high-id", required=True)
    parser.add_argument("--score-penalty", type=float, required=True)
    parser.add_argument("--target-tower", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
