from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2tower.methods.trace2tower.lineage import (
    build_mid_lineage,
    decompose_mid_lineage,
)
from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.refinement import (
    LifecycleAction,
    LifecycleUpdate,
    SkillStatus,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--refined-tower", type=Path, required=True)
    parser.add_argument("--candidate-clusters", type=Path, required=True)
    parser.add_argument("--usage-pareto", type=Path, required=True)
    parser.add_argument("--structural-pareto", type=Path, required=True)
    parser.add_argument("--refined-build-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    tower = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    refined_tower = TowerSnapshot.from_record(
        json.loads(options.refined_tower.read_text(encoding="utf-8"))
    )
    candidate = tuple(
        MidCluster.from_record(record)
        for record in json.loads(options.candidate_clusters.read_text(encoding="utf-8"))["clusters"]
    )
    usage_pareto = json.loads(options.usage_pareto.read_text(encoding="utf-8"))
    structural_pareto = json.loads(
        options.structural_pareto.read_text(encoding="utf-8")
    )
    build_report = json.loads(
        options.refined_build_report.read_text(encoding="utf-8")
    )
    if (
        usage_pareto["tower_snapshot_id"] != tower.snapshot_id
        or structural_pareto["tower_snapshot_id"] != tower.snapshot_id
        or usage_pareto["split"] != "train"
        or usage_pareto["ranking_status"] != "complete"
    ):
        raise ValueError("Pareto feedback does not bind to the train-side Tower snapshot")
    if refined_tower.version.value != "v1":
        raise ValueError("refined action plan requires a Tower v1 target")
    lineage = build_mid_lineage(tower.mid_clusters, candidate)
    decomposition = decompose_mid_lineage(lineage)
    selected_component_id = structural_pareto["selected_component_id"]
    if selected_component_id != build_report["selected_component_id"]:
        raise ValueError("refined build does not match structural Pareto selection")
    selected_steps = tuple(
        step.to_record()
        for step in decomposition.steps
        if step.component_id == selected_component_id
    )
    if not selected_steps:
        raise ValueError("selected structural component contains no atomic steps")
    update = usage_pareto["selected_usage_actions"]["downweight"]
    downweight = None
    if update is not None:
        downweight = LifecycleUpdate(
            skill_id=str(update["skill_id"]),
            action=LifecycleAction(update["action"]),
            previous_status=SkillStatus(update["previous_status"]),
            new_status=SkillStatus(update["new_status"]),
            refinement_round=int(update["refinement_round"]),
            pareto_front_rank=int(update["pareto_front_rank"]),
        )
    split_count = sum(step["action"] == "split" for step in selected_steps)
    merge_count = sum(step["action"] == "merge" for step in selected_steps)
    promoted_high_id = build_report["promoted_high_id"]
    known_target_high_ids = {path.path_id for path in refined_tower.high_paths}
    if promoted_high_id not in known_target_high_ids:
        raise ValueError("promoted High is absent from the refined Tower")
    if downweight is not None and downweight.skill_id not in known_target_high_ids:
        raise ValueError("downweighted High is absent from the refined Tower")
    payload = {
        "protocol_id": "webshop-train-refinement-v1",
        "source_tower_snapshot_id": tower.snapshot_id,
        "tower_snapshot_id": refined_tower.snapshot_id,
        "ranking_status": "complete",
        "direct_mid_top_k": 8,
        "status_tie_epsilon": 0.01,
        "old_mid_count": len(tower.mid_clusters),
        "candidate_mid_count": len(candidate),
        "lineage": lineage.to_record(),
        "structural_decomposition": decomposition.to_record(),
        "applied_atomic_action_counts": {
            "split": split_count,
            "merge": merge_count,
            "promote": 1,
            "downweight": int(downweight is not None),
        },
        "selected_actions": {
            "structural_transaction": {
                "component_id": selected_component_id,
                "atomic_steps": selected_steps,
            },
            "promote": {"skill_id": promoted_high_id},
            "downweight": downweight.to_record() if downweight is not None else None,
        },
        "structural_selection": {
            "report": options.structural_pareto.as_posix(),
            "selected_component_id": selected_component_id,
            "weak_edges": "retained in the lineage audit but excluded from structural actions",
        },
        "usage_selection": {
            "report": options.usage_pareto.as_posix(),
            "mid_usage_identifiable": usage_pareto["mid_usage_identifiable"],
        },
        "downweight": [downweight.to_record()] if downweight is not None else [],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
