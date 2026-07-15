from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2tower.methods.trace2tower.lineage import (
    build_mid_lineage,
    legal_merge_proposals,
    legal_split_proposals,
)
from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.refinement import (
    LifecycleAction,
    LifecycleUpdate,
    RefinementActionPlan,
    SkillStatus,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--candidate-clusters", type=Path, required=True)
    parser.add_argument("--pareto-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    tower = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    candidate = tuple(
        MidCluster.from_record(record)
        for record in json.loads(options.candidate_clusters.read_text(encoding="utf-8"))["clusters"]
    )
    pareto = json.loads(options.pareto_report.read_text(encoding="utf-8"))
    if (
        pareto["tower_snapshot_id"] != tower.snapshot_id
        or pareto["split"] != "train"
        or pareto["ranking_status"] != "complete"
    ):
        raise ValueError("Pareto feedback does not bind to the train-side Tower snapshot")
    lineage = build_mid_lineage(tower.mid_clusters, candidate)
    splits = legal_split_proposals(lineage)
    merges = legal_merge_proposals(lineage, tower.mid_clusters, candidate)
    if splits or merges:
        raise ValueError("this conservative planner requires explicit Split/Merge selection")
    updates = tuple(pareto.get("downweight", ()))
    if len(updates) > 1:
        raise ValueError("one refinement round permits at most one downweight")
    downweight = None
    if updates:
        update = updates[0]
        downweight = LifecycleUpdate(
            skill_id=str(update["skill_id"]),
            action=LifecycleAction(update["action"]),
            previous_status=SkillStatus(update["previous_status"]),
            new_status=SkillStatus(update["new_status"]),
            refinement_round=int(update["refinement_round"]),
            pareto_front_rank=int(update["pareto_front_rank"]),
        )
    plan = RefinementActionPlan(
        split=None,
        merge=None,
        promote=None,
        downweight=downweight,
    )
    payload = {
        "protocol_id": "webshop-train-refinement-v1",
        "tower_snapshot_id": tower.snapshot_id,
        "ranking_status": "complete",
        "direct_mid_top_k": 8,
        "status_tie_epsilon": 0.01,
        "old_mid_count": len(tower.mid_clusters),
        "candidate_mid_count": len(candidate),
        "lineage": lineage.to_record(),
        "legal_atomic_proposal_counts": {
            "split": len(splits),
            "merge": len(merges),
            "promote": 0,
            "downweight": len(updates),
        },
        "complex_repartition_policy": {
            "candidate_count": len(lineage.complex_repartitions),
            "applied": False,
            "reason": "global repartition is not one independently auditable atomic Split or Merge",
        },
        "action_plan": plan.to_record(),
        "downweight": [downweight.to_record()] if downweight is not None else [],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
