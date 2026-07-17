from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.core.manifests import expand_manifest_repeats, read_manifest
from trace2tower.methods.trace2tower.deployment_optimization.feedback import (
    bootstrap_pareto,
    bundle_metrics,
    pair_feedback,
    read_no_skill_trajectories,
    read_results,
)


def main(options: argparse.Namespace) -> int:
    config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    manifest = expand_manifest_repeats(read_manifest(options.manifest), config["repeat_ids"])
    expected_keys = {(entry.sample_id, entry.repeat_id) for entry in manifest}
    no_skill_results = read_no_skill_trajectories(
        options.no_skill_trajectories,
        sample_ids=frozenset(entry.sample_id for entry in manifest),
        repeat_ids=frozenset(config["repeat_ids"]),
    )
    tower_results = read_results(options.tower_run)
    observed_no_skill = {(result.sample_id, result.repeat_id) for result in no_skill_results}
    observed_tower = {(result.sample_id, result.repeat_id) for result in tower_results}
    if observed_no_skill != expected_keys or observed_tower != expected_keys:
        raise ValueError(
            "feedback runs are incomplete: "
            f"expected={len(expected_keys)}, no_skill={len(observed_no_skill)}, "
            f"tower={len(observed_tower)}"
        )

    tower_config = _read_resolved_config(options.tower_run)
    _audit_inputs(
        tower_config,
        options.manifest,
        options.no_skill_trajectories,
        config,
    )
    pairs = pair_feedback(no_skill_results, tower_results)
    all_metrics = bundle_metrics(pairs)
    pareto_config = config["deployment_pareto"]
    bootstrap_config = config["bootstrap"]
    estimates = bootstrap_pareto(
        pairs,
        samples=int(bootstrap_config["samples"]),
        seed=int(bootstrap_config["seed"]),
        min_exposure=int(pareto_config["ranking_min_exposure"]),
    )
    estimate_by_id = {item.metrics.primary_high_id: item for item in estimates}
    eligible = [
        item
        for item in estimates
        if item.metrics.exposure_count >= int(pareto_config["action_min_exposure"])
        and item.pareto_front_rank > 1
        and item.front_1_probability <= float(pareto_config["max_front_1_probability"])
        and item.dominated_probability >= float(pareto_config["min_dominated_probability"])
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item.pareto_front_rank,
            item.dominated_probability,
            -item.metrics.objectives.paired_success_gain,
            -item.metrics.objectives.performance_level,
            item.metrics.exposure_count,
            item.metrics.primary_high_id,
        ),
        default=None,
    )
    payload = {
        "protocol_id": "alfworld-deployment-optimization-v1-feedback",
        "benchmark": "alfworld",
        "split": "train",
        "base_snapshot_id": config["base_snapshot_id"],
        "manifest": {
            "path": options.manifest.as_posix(),
            "sha256": hashlib.sha256(options.manifest.read_bytes()).hexdigest(),
            "task_count": len({entry.sample_id for entry in manifest}),
            "episode_count": len(manifest),
        },
        "runs": {
            "no_skill": {
                "path": options.no_skill_trajectories.as_posix(),
                "sha256": hashlib.sha256(options.no_skill_trajectories.read_bytes()).hexdigest(),
                "source": "reused_trajectory_pool",
            },
            "tower_v0": _run_record(options.tower_run),
        },
        "objectives": pareto_config["objectives"],
        "bootstrap": bootstrap_config,
        "bundles": [
            {
                **asdict(metric),
                "pareto": (
                    {
                        "front_rank": estimate_by_id[metric.primary_high_id].pareto_front_rank,
                        "front_1_probability": estimate_by_id[
                            metric.primary_high_id
                        ].front_1_probability,
                        "dominated_probability": estimate_by_id[
                            metric.primary_high_id
                        ].dominated_probability,
                    }
                    if metric.primary_high_id in estimate_by_id
                    else None
                ),
            }
            for metric in all_metrics
        ],
        "downweight_proposals": (
            [
                {
                    "action": "downweight",
                    "primary_high_id": selected.metrics.primary_high_id,
                    "pareto_front_rank": selected.pareto_front_rank,
                    "front_1_probability": selected.front_1_probability,
                    "dominated_probability": selected.dominated_probability,
                }
            ]
            if selected is not None
            else []
        ),
    }
    write_json(options.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _read_resolved_config(run_root: Path) -> dict:
    return yaml.safe_load((run_root / "resolved-config.yaml").read_text(encoding="utf-8"))


def _audit_inputs(
    tower: dict,
    manifest: Path,
    no_skill_trajectories: Path,
    experiment_config: dict,
) -> None:
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if tower["agent_model"] != experiment_config["agent_model"]:
        raise ValueError("Tower agent model differs from the deployment protocol")
    if tower["manifests"]["alfworld"]["sha256"] != manifest_hash:
        raise ValueError("Tower manifest differs from the deployment protocol")
    artifact = tower["artifacts"]["alfworld"]
    if artifact["artifact_id"] != experiment_config["base_snapshot_id"]:
        raise ValueError("Tower feedback uses the wrong base snapshot")
    trajectory_hash = hashlib.sha256(no_skill_trajectories.read_bytes()).hexdigest()
    if trajectory_hash != experiment_config["no_skill_trajectory_pool_sha256"]:
        raise ValueError("No-Skill trajectory pool differs from the deployment protocol")


def _run_record(run_root: Path) -> dict:
    result_paths = sorted(run_root.rglob("results.jsonl"))
    return {
        "path": run_root.as_posix(),
        "result_files": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in result_paths
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/alfworld_deployment_optimization.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/alfworld/deployment-optimization-v1/manifests/deployment_feedback.jsonl"
        ),
    )
    parser.add_argument(
        "--no-skill-trajectories",
        type=Path,
        default=Path("artifacts/trajectories/alfworld/alfworld-pool-p1000-global.jsonl"),
    )
    parser.add_argument(
        "--tower-run",
        type=Path,
        default=Path("artifacts/runs/alfworld-deployment-v1-feedback-tower-v0-r3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/trace2tower/alfworld/deployment-optimization-v1/feedback/pareto-report.json"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
