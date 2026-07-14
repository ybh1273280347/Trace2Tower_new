from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.results import FinishReason, MethodName
from trace2tower.trajectory import TrajectoryReader

EXPECTED_REPEAT_IDS = (0, 1, 2, 3)
POOL_SIZES = {"p50": 50, "p100": 100}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_run_audit(
    run_id: str,
    expected_trajectory_count: int,
    runs_root: Path,
) -> dict:
    run_root = runs_root / run_id
    metadata_path = run_root / "matrix-metadata.json"
    config_path = run_root / "resolved-config.yaml"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shards = metadata["shards"]

    if (
        metadata["run_id"] != run_id
        or metadata["benchmarks"] != [Benchmark.WEBSHOP.value]
        or metadata["split"] != ExperimentSplit.TRAIN.value
        or metadata["method"] != MethodName.NO_SKILL.value
        or tuple(metadata["repeat_ids"]) != EXPECTED_REPEAT_IDS
    ):
        raise ValueError(f"source run contract mismatch: {run_id}")

    official_result_count = sum(item["official_result_count"] for item in shards)
    trajectory_count = sum(item["trajectory_count"] for item in shards)
    error_attempt_count = sum(item["error_attempt_count"] for item in shards)
    invocation_failed_count = sum(
        item["invocation_summary"]["failed"] for item in shards
    )
    if official_result_count != expected_trajectory_count:
        raise ValueError(f"source result coverage mismatch: {run_id}")
    if trajectory_count != expected_trajectory_count:
        raise ValueError(f"source trajectory coverage mismatch: {run_id}")
    if error_attempt_count or invocation_failed_count:
        raise ValueError(f"source run contains unresolved errors: {run_id}")

    return {
        "run_id": run_id,
        "agent_model": metadata["agent_model"],
        "expected_trajectory_count": expected_trajectory_count,
        "official_result_count": official_result_count,
        "trajectory_count": trajectory_count,
        "error_attempt_count": error_attempt_count,
        "invocation_failed_count": invocation_failed_count,
        "matrix_metadata": metadata_path.as_posix(),
        "matrix_metadata_sha256": sha256_file(metadata_path),
        "resolved_config": config_path.as_posix(),
        "resolved_config_sha256": sha256_file(config_path),
    }


def audit_pool(
    label: str,
    path: Path,
    task_count: int,
    runs_root: Path,
) -> tuple[dict, dict]:
    trajectories = TrajectoryReader.read_jsonl(path)
    expected_trajectory_count = task_count * len(EXPECTED_REPEAT_IDS)
    if len(trajectories) != expected_trajectory_count:
        raise ValueError(f"{label} trajectory count mismatch")
    if any(
        trajectory.benchmark is not Benchmark.WEBSHOP
        or trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.NO_SKILL
        for trajectory in trajectories
    ):
        raise ValueError(f"{label} contains trajectories outside the pool contract")

    sample_ids = sorted(
        {trajectory.sample_id for trajectory in trajectories},
        key=lambda sample_id: int(sample_id.split(":")[1]),
    )
    if len(sample_ids) != task_count:
        raise ValueError(f"{label} task count mismatch")
    episode_keys = {
        (trajectory.sample_id, trajectory.repeat_id) for trajectory in trajectories
    }
    expected_keys = {
        (sample_id, repeat_id)
        for sample_id in sample_ids
        for repeat_id in EXPECTED_REPEAT_IDS
    }
    if episode_keys != expected_keys:
        raise ValueError(f"{label} does not cover every task/repeat pair")

    source_counts = Counter(trajectory.run_id for trajectory in trajectories)
    source_runs = [
        source_run_audit(run_id, count, runs_root)
        for run_id, count in sorted(source_counts.items())
    ]
    score_counts = Counter(trajectory.primary_score for trajectory in trajectories)
    finish_counts = Counter(trajectory.finish_reason.value for trajectory in trajectories)
    invalid_action_count = sum(
        not step.valid_action
        for trajectory in trajectories
        for step in trajectory.steps
    )
    system_failure_reasons = {
        FinishReason.AGENT_VALIDATION_FAILED,
        FinishReason.TASK_ERROR,
        FinishReason.CANCELLED,
    }
    system_failure_count = sum(
        trajectory.finish_reason in system_failure_reasons
        for trajectory in trajectories
    )

    report = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "task_count": task_count,
        "sample_ids": sample_ids,
        "sample_ids_sha256": canonical_hash(sample_ids),
        "repeat_ids": list(EXPECTED_REPEAT_IDS),
        "trajectory_count": len(trajectories),
        "trajectory_ids_sha256": canonical_hash(
            sorted(trajectory.trajectory_id for trajectory in trajectories)
        ),
        "source_runs": source_runs,
        "reward": {
            "mean": fmean(
                trajectory.primary_score for trajectory in trajectories
            ),
            "full_success_count": sum(
                trajectory.primary_score >= 0.999 for trajectory in trajectories
            ),
            "positive_partial_count": sum(
                0 < trajectory.primary_score < 0.999
                for trajectory in trajectories
            ),
            "zero_count": sum(
                trajectory.primary_score == 0 for trajectory in trajectories
            ),
            "histogram": {
                format(score, ".12g"): count
                for score, count in sorted(score_counts.items())
            },
        },
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "system_failure_count": system_failure_count,
        "invalid_action_count": invalid_action_count,
    }
    records = {
        trajectory.trajectory_id: trajectory.to_record()
        for trajectory in trajectories
    }
    return report, records


def render_report(audit: dict) -> str:
    p50 = audit["pools"]["p50"]
    p100 = audit["pools"]["p100"]
    conclusion = (
        "P50 和 P100 均满足 Event Tower V2 训练池契约。P50 的 200 条轨迹在 "
        "P100 中逐记录一致；P100 只新增 50 个任务和 200 条轨迹。两个池均为 "
        "Flash No-Skill train rollout，source metadata 中没有 error attempt 或 "
        "failed invocation。"
    )
    table_header = (
        "| 池 | Tasks | Repeats | Trajectories | Mean reward | Full success | "
        "Partial | Zero | System failures |"
    )
    p50_row = (
        f"| P50 | {p50['task_count']} | 4 | {p50['trajectory_count']} | "
        f"{p50['reward']['mean']:.6f} | {p50['reward']['full_success_count']} | "
        f"{p50['reward']['positive_partial_count']} | "
        f"{p50['reward']['zero_count']} | {p50['system_failure_count']} |"
    )
    p100_row = (
        f"| P100 | {p100['task_count']} | 4 | {p100['trajectory_count']} | "
        f"{p100['reward']['mean']:.6f} | {p100['reward']['full_success_count']} | "
        f"{p100['reward']['positive_partial_count']} | "
        f"{p100['reward']['zero_count']} | {p100['system_failure_count']} |"
    )
    return f"""# Stage 1: WebShop 训练轨迹池审计

状态：`complete`
审计 ID：`{audit['audit_id']}`

## 结论

{conclusion}

{table_header}
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{p50_row}
{p100_row}

## 冻结输入

- P50 SHA-256：`{p50['sha256']}`
- P100 SHA-256：`{p100['sha256']}`
- P50 source run：`{p50['source_runs'][0]['run_id']}`
- P100 additional source run：`{p100['source_runs'][-1]['run_id']}`
- Validation/Test selection：`{audit['evaluation_selection']['selection_id']}`

## 不变量

- P50 task/repeat coverage：完整的 50 x 4 笛卡尔积。
- P100 task/repeat coverage：完整的 100 x 4 笛卡尔积。
- P50 task set 是 P100 task set 的严格子集。
- P50 对应 trajectory records 在 P100 中完全相同。
- 训练任务与 validation/test indices `0..999` 零重叠。
- 本阶段只读取并审计已有轨迹，没有重新 rollout，也没有修改 validation/test manifests。

完整 sample IDs、reward histogram、finish reasons、source metadata hashes 和机器可验证不变量
见 `audit.json`。
"""


def main(options: argparse.Namespace) -> int:
    p50, p50_records = audit_pool(
        "p50",
        options.p50,
        POOL_SIZES["p50"],
        options.runs_root,
    )
    p100, p100_records = audit_pool(
        "p100",
        options.p100,
        POOL_SIZES["p100"],
        options.runs_root,
    )
    evaluation_protocol = json.loads(
        options.evaluation_protocol.read_text(encoding="utf-8")
    )
    validation_ids = set(evaluation_protocol["selection"]["validation_sample_ids"])
    test_ids = set(evaluation_protocol["selection"]["test_sample_ids"])
    training_ids = set(p100["sample_ids"])

    invariants = {
        "p50_task_count": p50["task_count"] == 50,
        "p50_trajectory_count": p50["trajectory_count"] == 200,
        "p100_task_count": p100["task_count"] == 100,
        "p100_trajectory_count": p100["trajectory_count"] == 400,
        "p50_tasks_strictly_nested": set(p50["sample_ids"]) < training_ids,
        "p50_episodes_nested": set(p50_records) < set(p100_records),
        "p50_records_unchanged": all(
            record == p100_records[trajectory_id]
            for trajectory_id, record in p50_records.items()
        ),
        "source_runs_error_free": all(
            not source["error_attempt_count"]
            and not source["invocation_failed_count"]
            for pool in (p50, p100)
            for source in pool["source_runs"]
        ),
        "validation_test_disjoint": not validation_ids & test_ids,
        "training_evaluation_disjoint": not training_ids
        & (validation_ids | test_ids),
    }
    if not all(invariants.values()):
        failures = [name for name, passed in invariants.items() if not passed]
        raise ValueError(f"training pool audit failed: {', '.join(failures)}")

    audit = {
        "protocol_id": "webshop-event-tower-v2",
        "stage": 1,
        "status": "complete",
        "pools": {"p50": p50, "p100": p100},
        "nesting": {
            "shared_task_count": len(set(p50["sample_ids"]) & training_ids),
            "additional_p100_task_count": len(
                training_ids - set(p50["sample_ids"])
            ),
            "shared_trajectory_count": len(set(p50_records) & set(p100_records)),
            "additional_p100_trajectory_count": len(
                set(p100_records) - set(p50_records)
            ),
        },
        "evaluation_selection": {
            "path": options.evaluation_protocol.as_posix(),
            "sha256": sha256_file(options.evaluation_protocol),
            "selection_id": evaluation_protocol["selection_id"],
            "validation_task_count": len(validation_ids),
            "test_task_count": len(test_ids),
            "training_overlap_count": len(
                training_ids & (validation_ids | test_ids)
            ),
        },
        "invariants": invariants,
    }
    audit["audit_id"] = f"poolaudit_{canonical_hash(audit)[:16]}"
    write_json(options.output, audit)
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(render_report(audit), encoding="utf-8", newline="\n")
    print(json.dumps({"audit_id": audit["audit_id"], **invariants}, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p50",
        type=Path,
        default=Path(
            "artifacts/trajectories/webshop/scale-v1/"
            "webshop-scale-v1-p50.jsonl"
        ),
    )
    parser.add_argument(
        "--p100",
        type=Path,
        default=Path(
            "artifacts/trajectories/webshop/scale-v1/"
            "webshop-scale-v1-p100.jsonl"
        ),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=Path("configs/experiments/webshop_event_tower_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/stage-1-pools/audit.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/stage-1-pools/REPORT.md"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
