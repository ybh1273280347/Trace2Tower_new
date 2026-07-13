from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from rollout_no_skill_train import load_yaml, write_json

from trace2tower.methods.trace2tower.refinement import (
    RefinementEpisode,
    SkillLevel,
    audit_refinement_evidence,
    build_paired_episode_evidence,
    build_skill_objectives,
    rank_skill_objectives,
    select_downweight,
    validate_execution_contract,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def read_results(paths: list[Path]) -> tuple[RefinementEpisode, ...]:
    return tuple(
        RefinementEpisode.from_record(json.loads(line))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def file_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main(options: argparse.Namespace) -> int:
    config = load_yaml(options.config)
    expected_config = {
        "refinement_round": 1,
        "cost_field": "chat_tokens",
        "missing_cost_policy": "reject_ranking",
        "physical_deletion": False,
    }
    if config != expected_config:
        raise ValueError("Pareto refinement config changed outside the frozen contract")
    tower = TowerSnapshot.from_record(
        json.loads(options.tower.read_text(encoding="utf-8"))
    )
    skill_levels = {
        **{cluster.cluster_id: SkillLevel.MID for cluster in tower.mid_clusters},
        **{path.path_id: SkillLevel.HIGH for path in tower.high_paths},
    }
    baseline = read_results(options.baseline_results)
    skill = read_results(options.skill_results)
    baseline_metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in options.baseline_run_metadata
    ]
    skill_report = json.loads(options.skill_run_report.read_text(encoding="utf-8"))
    execution_contract = validate_execution_contract(
        tower_snapshot_id=tower.snapshot_id,
        benchmark=tower.benchmark,
        baseline_episodes=baseline,
        skill_episodes=skill,
        baseline_metadata=baseline_metadata,
        skill_report=skill_report,
    )
    audit = audit_refinement_evidence(baseline, skill, skill_levels)
    report = {
        "tower_snapshot_id": tower.snapshot_id,
        "benchmark": tower.benchmark.value,
        "refinement_config": config,
        "refinement_config_sha256": hashlib.sha256(
            options.config.read_bytes()
        ).hexdigest(),
        "refinement_round": config["refinement_round"],
        "execution_contract": {
            **execution_contract.to_record(),
            "baseline_metadata_hashes": file_hashes(
                options.baseline_run_metadata
            ),
            "skill_report_sha256": hashlib.sha256(
                options.skill_run_report.read_bytes()
            ).hexdigest(),
        },
        "baseline_result_hashes": file_hashes(options.baseline_results),
        "skill_result_hashes": file_hashes(options.skill_results),
        "audit": audit.to_record(),
        "paired_episode_evidence": [
            evidence.to_record()
            for evidence in build_paired_episode_evidence(baseline, skill)
        ],
        "ranking_status": "unavailable",
        "ranked_skills": [],
        "downweight": [],
    }
    if audit.is_complete:
        objectives = build_skill_objectives(
            baseline,
            skill,
            skill_levels,
            refinement_round=config["refinement_round"],
        )
        ranked = rank_skill_objectives(objectives)
        report["ranking_status"] = "complete"
        report["ranked_skills"] = [item.to_record() for item in ranked]
        by_scope = {}
        for item in ranked:
            scope = (item.benchmark, item.skill_level, item.refinement_round)
            by_scope.setdefault(scope, []).append(item)
        report["downweight"] = [
            select_downweight(items, {item.skill_id for item in items}).to_record()
            for items in by_scope.values()
        ]
    write_json(options.output, report)
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    if options.require_complete and not audit.is_complete:
        return 2
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, nargs="+", required=True)
    parser.add_argument("--skill-results", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--baseline-run-metadata", type=Path, nargs="+", required=True
    )
    parser.add_argument("--skill-run-report", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/pareto_refinement.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    raise SystemExit(main(parser.parse_args()))
