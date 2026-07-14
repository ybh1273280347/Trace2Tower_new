from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    ManifestEntry,
    write_manifest,
)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_entries(
    sample_ids: list[str], split: ExperimentSplit
) -> list[ManifestEntry]:
    return [
        ManifestEntry(
            benchmark=Benchmark.WEBSHOP,
            split=split,
            sample_id=sample_id,
            dataset_index=int(sample_id.split(":")[1]),
            source_split="goals",
            repeat_id=0,
        )
        for sample_id in sample_ids
    ]


def main(options: argparse.Namespace) -> int:
    goals = json.loads(options.goals.read_text(encoding="utf-8"))
    if len(goals) < options.candidate_end:
        raise ValueError("candidate range exceeds the WebShop goal set")
    if options.validation_tasks <= 0 or options.test_tasks <= 0:
        raise ValueError("validation and test sizes must be positive")

    selected = random.Random(options.seed).sample(
        range(options.candidate_start, options.candidate_end),
        options.validation_tasks + options.test_tasks,
    )
    validation_ids = sorted(selected[: options.validation_tasks])
    test_ids = sorted(selected[options.validation_tasks :])
    validation_samples = [f"webshop:{index}" for index in validation_ids]
    test_samples = [f"webshop:{index}" for index in test_ids]
    if set(validation_samples) & set(test_samples):
        raise RuntimeError("validation and test selections overlap")

    write_manifest(
        options.validation_manifest,
        manifest_entries(validation_samples, ExperimentSplit.DEV),
    )
    write_manifest(
        options.test_manifest,
        manifest_entries(test_samples, ExperimentSplit.TEST),
    )

    validation_conditions = [
        {
            "condition_id": f"p50_trace2tower_cap{cap}_{model}",
            "method": "trace2tower",
            "pool": "p50",
            "direct_mid_top_k": cap,
            "agent_model": model,
        }
        for model in ("deepseek-v4-flash", "deepseek-v4-pro")
        for cap in (3, 5, 8)
    ]
    primary_methods = [
        "no_skill",
        "manual_skill",
        "global_e2e_gpt",
        "skillx",
        "trace2tower",
    ]
    primary_conditions = [
        {
            "condition_id": f"p50_{method}_{model}",
            "method": method,
            "pool": "p50" if method not in ("no_skill", "manual_skill") else None,
            "agent_model": model,
            "direct_mid_top_k": "frozen_from_stage_3"
            if method == "trace2tower"
            else None,
        }
        for model in ("deepseek-v4-flash", "deepseek-v4-pro")
        for method in primary_methods
    ]
    ablation_conditions = [
        {
            "condition_id": f"p50_{method}_{model}",
            "method": method,
            "pool": "p50",
            "agent_model": model,
            "direct_mid_top_k": "frozen_from_stage_3",
            "run_policy": "required"
            if model == "deepseek-v4-pro"
            else "only_if_flash_gate_passes",
        }
        for model in ("deepseek-v4-pro", "deepseek-v4-flash")
        for method in (
            "trace2tower_semantic_only",
            "trace2tower_mid_only",
            "trace2tower_no_mixed",
        )
    ]
    scale_conditions = [
        {
            "condition_id": f"p100_{method}_{model}",
            "method": method,
            "pool": "p100",
            "agent_model": model,
            "direct_mid_top_k": "frozen_from_stage_3"
            if method == "trace2tower"
            else None,
            "run_policy": "required"
            if model == "deepseek-v4-pro"
            else "only_if_flash_gate_passes",
        }
        for model in ("deepseek-v4-pro", "deepseek-v4-flash")
        for method in ("global_e2e_gpt", "skillx", "trace2tower")
    ]
    protocol = {
        "protocol_id": "webshop-event-tower-v2",
        "benchmark": "webshop",
        "status": "frozen_before_rollout",
        "candidate_index_range": [options.candidate_start, options.candidate_end],
        "selection": {
            "algorithm": "random.Random(seed).sample without replacement",
            "seed": options.seed,
            "history_filtering": False,
            "validation_sample_ids": validation_samples,
            "test_sample_ids": test_samples,
            "overlap_count": 0,
        },
        "execution": {
            "agent_models": ["deepseek-v4-flash", "deepseek-v4-pro"],
            "renderer_model": "gpt-5.4",
            "repeat_ids": [0, 1, 2],
            "tasks_per_split": 100,
            "episodes_per_condition_per_split": 300,
            "tower_cap_candidates": [3, 5, 8],
            "skillx_max_skills": 8,
            "max_steps": 20,
        },
        "manifests": {
            "validation": options.validation_manifest.as_posix(),
            "test": options.test_manifest.as_posix(),
        },
        "training_pools": {
            "p50": {
                "task_count": 50,
                "repeat_count": 4,
                "trajectory_count": 200,
                "path": (
                    "artifacts/trajectories/webshop/scale-v1/"
                    "webshop-scale-v1-p50.jsonl"
                ),
            },
            "p100": {
                "task_count": 100,
                "repeat_count": 4,
                "trajectory_count": 400,
                "path": (
                    "artifacts/trajectories/webshop/scale-v1/"
                    "webshop-scale-v1-p100.jsonl"
                ),
            },
        },
        "stages": [
            {
                "stage": 1,
                "name": "audit_training_pools",
                "pools": ["p50", "p100"],
                "rollout_episodes": 0,
            },
            {
                "stage": 2,
                "name": "build_p50_skills",
                "methods": [
                    "global_e2e_gpt",
                    "skillx",
                    "trace2tower",
                    "trace2tower_semantic_only",
                    "trace2tower_no_mixed",
                ],
                "rollout_episodes": 0,
            },
            {
                "stage": 3,
                "name": "select_tower_cap",
                "split": "dev",
                "conditions": validation_conditions,
                "rollout_episodes": len(validation_conditions) * 300,
            },
            {
                "stage": 4,
                "name": "p50_baselines_and_full_tower",
                "split": "test",
                "conditions": primary_conditions,
                "rollout_episodes": len(primary_conditions) * 300,
            },
            {
                "stage": 5,
                "name": "flash_significance_gate",
                "comparisons": [
                    f"{method} - no_skill"
                    for method in (
                        "manual_skill",
                        "global_e2e_gpt",
                        "skillx",
                        "trace2tower",
                    )
                ],
                "multiplicity_control": "Holm, family-wise alpha 0.05",
                "rollout_episodes": 0,
            },
            {
                "stage": 6,
                "name": "p50_ablations",
                "split": "test",
                "conditions": ablation_conditions,
                "rollout_episodes_min": 900,
                "rollout_episodes_max": 1800,
            },
            {
                "stage": 7,
                "name": "freeze_p50_results",
                "rollout_episodes": 0,
            },
            {
                "stage": 8,
                "name": "p100_scale",
                "split": "test",
                "conditions": scale_conditions,
                "rollout_episodes_min": 900,
                "rollout_episodes_max": 1800,
            },
        ],
        "episode_budget": {
            "validation": 1800,
            "test_min": 4800,
            "test_max": 6600,
            "total_min": 6600,
            "total_max": 8400,
        },
        "cap_selection": {
            "scope": "P50 Full Trace2Tower only, both agent models",
            "rule": (
                "choose the smallest cap whose paired task-bootstrap reward "
                "difference from the empirical best includes zero at 95%"
            ),
            "excluded": ["skillx", "trace2tower_no_mixed", "all other methods"],
        },
        "primary_comparisons": [
            "P50 trace2tower - each P50 baseline within agent model",
            "P50 trace2tower - each P50 ablation within agent model",
            "P100 - P50 within global_e2e_gpt, skillx, and trace2tower",
        ],
        "artifact_policy": {
            "reusable": [
                "audited P50/P100 No-Skill training trajectories",
                "Global E2E artifacts built from the exact pool and frozen contract",
                "SkillX artifacts built from the exact pool and frozen contract",
                "the frozen Manual Skill text",
            ],
            "must_rebuild": [
                "Trace2Tower P50 event-stratified mixed snapshot",
                "Trace2Tower P100 event-stratified mixed snapshot",
                "P50 semantic-only ablation snapshot",
                "P50 no-mixed event-stratified snapshot",
            ],
            "forbidden_as_full": "every pre-v2 Tower snapshot",
        },
    }
    protocol["selection_id"] = f"selection_{canonical_hash(protocol['selection'])[:16]}"
    write_json(options.output, protocol)
    print(
        json.dumps(
            {
                "selection_id": protocol["selection_id"],
                "validation_tasks": len(validation_samples),
                "test_tasks": len(test_samples),
                "validation_conditions": len(validation_conditions),
                "total_episodes_min": protocol["episode_budget"]["total_min"],
                "total_episodes_max": protocol["episode_budget"]["total_max"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", type=Path, default=Path("Datasets/webshop/goals.json"))
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--candidate-end", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--validation-tasks", type=int, default=100)
    parser.add_argument("--test-tasks", type=int, default=100)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/manifests/validation.jsonl"
        ),
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("experiments/webshop/event-tower-v2/manifests/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/experiments/webshop_event_tower_v2.json"),
    )
    raise SystemExit(main(parser.parse_args()))
