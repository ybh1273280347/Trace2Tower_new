from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from trace2tower.methods.trace2tower.refinement import (
    RefinementEpisode,
    SkillLevel,
    audit_refinement_evidence,
    build_skill_objectives,
    rank_skill_objectives,
    select_downweight,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def read_results(root: Path) -> tuple[RefinementEpisode, ...]:
    records = {}
    for path in sorted(root.glob("**/results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            episode = RefinementEpisode.from_record(json.loads(line))
            if episode.pair_key in records:
                raise ValueError(f"duplicate episode key in {root}: {episode.pair_key}")
            records[episode.pair_key] = episode
    if not records:
        raise ValueError(f"no results found under {root}")
    return tuple(records[key] for key in sorted(records))


def ranked_record(item) -> dict:
    return {
        "skill_id": item.skill_id,
        "skill_level": item.skill_level.value,
        "objective_vector": asdict(item.objective_vector),
        "exposure_count": item.exposure_count,
        "pareto_front_rank": item.pareto_front_rank,
        "dominated_by": item.dominated_by,
        "dominates": item.dominates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--tower-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    tower = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    baseline = read_results(options.baseline_run)
    tower_episodes = read_results(options.tower_run)
    skill_levels = {
        **{cluster.cluster_id: SkillLevel.MID for cluster in tower.mid_clusters},
        **{path.path_id: SkillLevel.HIGH for path in tower.high_paths},
    }
    audit = audit_refinement_evidence(baseline, tower_episodes, skill_levels)
    audit.require_complete()
    ranked = rank_skill_objectives(
        build_skill_objectives(
            baseline,
            tower_episodes,
            skill_levels,
            refinement_round=1,
        )
    )
    ranked_mid = tuple(item for item in ranked if item.skill_level is SkillLevel.MID)
    ranked_high = tuple(item for item in ranked if item.skill_level is SkillLevel.HIGH)

    exposure_sets = {item.paired_episode_keys for item in ranked_mid}
    mid_usage_identifiable = len(exposure_sets) > 1
    minimum_exposure = 10
    downweight_ids = {
        item.skill_id
        for item in ranked_high
        if item.exposure_count >= minimum_exposure and item.pareto_front_rank > 1
    }
    downweight = (
        select_downweight(ranked_high, downweight_ids).to_record()
        if downweight_ids
        else None
    )
    payload = {
        "protocol_id": "webshop-train-refinement-v1-pareto-selection",
        "tower_snapshot_id": tower.snapshot_id,
        "benchmark": tower.benchmark.value,
        "split": "train",
        "agent_model": "deepseek-v4-flash",
        "paired_episode_count": len(audit.paired_episode_keys),
        "primary_objectives": [
            "performance_level",
            "paired_reward_gain",
            "guarded_step_saving",
        ],
        "secondary_metrics": ["guarded_cost_saving"],
        "ranking_status": "complete",
        "ranked_skills": [ranked_record(item) for item in ranked],
        "mid_usage_identifiable": mid_usage_identifiable,
        "mid_usage_evidence": {
            "unique_exposure_sets": len(exposure_sets),
            "mid_skill_count": len(ranked_mid),
            "reason": (
                None
                if mid_usage_identifiable
                else "all Mid skills were co-injected on the same paired episodes"
            ),
        },
        "selected_usage_actions": {
            "downweight": downweight,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
