from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.evaluation import aggregate_method
from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    expand_manifest_repeats,
    read_manifest,
)
from trace2tower.results import EpisodeResult, FinishReason, MethodName

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260719
CONFIDENCE_LEVEL = 0.95


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


def read_condition_results(run_id: str, method: MethodName) -> tuple[EpisodeResult, ...]:
    paths = sorted(
        Path("artifacts/runs").glob(
            f"{run_id}/webshop/dev/{method.value}/shard-*/results.jsonl"
        )
    )
    if len(paths) != 10:
        raise ValueError(f"condition does not have ten result shards: {run_id}")
    return tuple(
        EpisodeResult.from_record(json.loads(line))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def condition_summary(
    condition: dict,
    ledger_record: dict,
    expected_entries: list,
) -> tuple[dict, tuple[EpisodeResult, ...]]:
    method = MethodName(condition["method"])
    run_id = ledger_record["run_id"]
    results = read_condition_results(run_id, method)
    coverage, aggregate = aggregate_method(
        expected_entries,
        results,
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.DEV,
        method=method,
    )
    coverage.require_complete()
    run_root = Path("artifacts/runs") / run_id
    matrix_metadata_path = run_root / "matrix-metadata.json"
    resolved_config_path = run_root / "resolved-config.yaml"
    matrix_metadata = json.loads(matrix_metadata_path.read_text(encoding="utf-8"))
    if (
        matrix_metadata["agent_model"] != condition["agent_model"]
        or matrix_metadata["method"] != condition["method"]
    ):
        raise ValueError(f"condition metadata mismatch: {condition['condition_id']}")
    error_attempt_count = sum(
        shard["error_attempt_count"] for shard in matrix_metadata["shards"]
    )
    ordered_records = [
        result.to_record()
        for result in sorted(
            results,
            key=lambda item: (item.sample_id, item.repeat_id),
        )
    ]
    chat_input = [
        result.chat_input_tokens
        for result in results
        if result.chat_input_tokens is not None
    ]
    return (
        {
            **condition,
            "run_id": run_id,
            "episode_count": len(results),
            "task_count": len({result.sample_id for result in results}),
            "coverage": coverage.official_result_coverage,
            "mean_reward": aggregate.primary_metric_mean,
            "full_success_rate": aggregate.full_success_rate,
            "completion_rate": aggregate.completion_rate,
            "mean_steps": aggregate.mean_steps,
            "invalid_action_rate": aggregate.invalid_action_rate,
            "mean_skill_context_chars": aggregate.mean_skill_context_chars,
            "mean_context_skill_count": fmean(
                len(result.context_skill_ids) for result in results
            ),
            "mean_chat_input_tokens": fmean(chat_input) if chat_input else None,
            "task_limit_count": sum(
                result.finish_reason is FinishReason.TASK_LIMIT_REACHED
                for result in results
            ),
            "error_attempt_count": error_attempt_count,
            "result_set_sha256": canonical_hash(ordered_records),
            "resolved_config": resolved_config_path.as_posix(),
            "resolved_config_sha256": sha256_file(resolved_config_path),
            "matrix_metadata": matrix_metadata_path.as_posix(),
            "matrix_metadata_sha256": sha256_file(matrix_metadata_path),
        },
        results,
    )


def task_reward_means(results: tuple[EpisodeResult, ...]) -> dict[str, float]:
    scores = defaultdict(list)
    for result in results:
        scores[result.sample_id].append(result.primary_score)
    if set(map(len, scores.values())) != {3}:
        raise ValueError("every validation task must have three repeats")
    return {sample_id: fmean(values) for sample_id, values in scores.items()}


def bootstrap_interval(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    batch_size = 512
    for start in range(0, BOOTSTRAP_SAMPLES, batch_size):
        end = min(start + batch_size, BOOTSTRAP_SAMPLES)
        indices = rng.integers(
            0,
            len(differences),
            size=(end - start, len(differences)),
        )
        means[start:end] = differences[indices].mean(axis=1)
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    interval = np.quantile(means, (alpha, 1 - alpha), method="linear")
    return float(interval[0]), float(interval[1])


def select_cap(
    method: MethodName,
    results: dict[tuple[MethodName, str, int], tuple[EpisodeResult, ...]],
) -> dict:
    caps = (3, 5, 8)
    models = ("deepseek-v4-flash", "deepseek-v4-pro")
    means_by_cap = {}
    task_means_by_cap = {}
    for cap in caps:
        by_model = [task_reward_means(results[(method, model, cap)]) for model in models]
        if set(by_model[0]) != set(by_model[1]):
            raise ValueError("agent models do not cover the same validation tasks")
        task_means_by_cap[cap] = {
            sample_id: fmean(model_scores[sample_id] for model_scores in by_model)
            for sample_id in sorted(by_model[0])
        }
        means_by_cap[cap] = fmean(task_means_by_cap[cap].values())

    empirical_best = max(caps, key=lambda cap: (means_by_cap[cap], -cap))
    comparisons = {}
    eligible = []
    for cap in caps:
        differences = np.asarray(
            [
                task_means_by_cap[empirical_best][sample_id]
                - task_means_by_cap[cap][sample_id]
                for sample_id in sorted(task_means_by_cap[cap])
            ],
            dtype=np.float64,
        )
        interval = bootstrap_interval(differences)
        includes_zero = interval[0] <= 0 <= interval[1]
        if includes_zero:
            eligible.append(cap)
        comparisons[str(cap)] = {
            "empirical_best_minus_candidate": float(differences.mean()),
            "confidence_interval": list(interval),
            "includes_zero": includes_zero,
        }
    return {
        "method": method.value,
        "macro_mean_reward_by_cap": {
            str(cap): means_by_cap[cap] for cap in caps
        },
        "empirical_best_cap": empirical_best,
        "comparisons_to_empirical_best": comparisons,
        "selected_cap": min(eligible),
    }


def render_report(audit: dict) -> str:
    rows = []
    for condition in audit["conditions"]:
        model = condition["agent_model"].removeprefix("deepseek-v4-")
        rows.append(
            f"| {condition['method']} | {model} | {condition['direct_mid_top_k']} | "
            f"{condition['mean_reward']:.6f} | "
            f"{condition['full_success_rate']:.3f} | "
            f"{condition['mean_steps']:.3f} | "
            f"{condition['mean_chat_input_tokens']:.1f} |"
        )
    selections = []
    for method, selection in audit["cap_selection"].items():
        means = selection["macro_mean_reward_by_cap"]
        selections.append(
            f"| {method} | {means['3']:.6f} | {means['5']:.6f} | "
            f"{means['8']:.6f} | {selection['empirical_best_cap']} | "
            f"**{selection['selected_cap']}** |"
        )
    coverage_summary = (
        "12 个条件均完成 100 tasks x 3 repeats，共 3,600 个 official episodes。"
        "每个条件覆盖率均为 1.0，结果键完全一致。Validation 只用于分别选择 "
        "Semantic Clustering 和 Full Trace2Tower 的直接 Mid cap。"
    )
    selection_summary = (
        "选择规则按 task 聚合：先平均 3 repeats，再平均 Flash/Pro。以经验均值"
        "最优 cap 为参照，对“最优减候选”的 task differences 做 10,000 次配对 "
        "bootstrap；选择 95% 区间包含 0 的最小 cap。完整区间、结果哈希、"
        "resolved config 哈希和运行 metadata 见 `audit.json`；机器冻结结果见 "
        "`selected-caps.json`。"
    )
    return f"""# Stage 3: Validation Cap 冻结

状态：`complete`
审计 ID：`{audit['audit_id']}`

## 覆盖

{coverage_summary}

## 条件结果

| 方法 | 模型 | Cap | Mean reward | Full success | Mean steps | Mean chat input tokens |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 冻结选择

| 方法 | cap3 | cap5 | cap8 | 经验最优 | 冻结 cap |
|---|---:|---:|---:|---:|---:|
{chr(10).join(selections)}

{selection_summary}
"""


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    stage = next(item for item in protocol["stages"] if item["stage"] == 3)
    protocol_conditions = stage["conditions"]
    ledger = json.loads(options.ledger.read_text(encoding="utf-8"))
    ledger_by_id = {item["condition_id"]: item for item in ledger["conditions"]}
    if (
        set(ledger_by_id)
        != {condition["condition_id"] for condition in protocol_conditions}
        or any(item["return_code"] != 0 for item in ledger_by_id.values())
    ):
        raise ValueError("stage 3 ledger is incomplete or differs from the protocol")
    if ledger["manifest_sha256"] != sha256_file(options.manifest):
        raise ValueError("stage 3 ledger uses a different validation manifest")

    expected_entries = expand_manifest_repeats(
        read_manifest(options.manifest),
        (0, 1, 2),
    )
    summaries = []
    results = {}
    for condition in protocol_conditions:
        summary, condition_results = condition_summary(
            condition,
            ledger_by_id[condition["condition_id"]],
            expected_entries,
        )
        summaries.append(summary)
        results[
            (
                MethodName(condition["method"]),
                condition["agent_model"],
                condition["direct_mid_top_k"],
            )
        ] = condition_results

    selections = {
        method.value: select_cap(method, results)
        for method in (
            MethodName.SEMANTIC_CLUSTERING,
            MethodName.TRACE2TOWER,
        )
    }
    selected_caps = {
        "protocol_id": protocol["protocol_id"],
        "stage": 3,
        "selection_id": protocol["selection_id"],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "rule": protocol["cap_selection"]["rule"],
        "methods": selections,
    }
    write_json(options.selected_caps, selected_caps)
    audit = {
        "protocol_id": protocol["protocol_id"],
        "stage": 3,
        "status": "complete",
        "protocol": options.protocol.as_posix(),
        "protocol_sha256": sha256_file(options.protocol),
        "stage_contract_sha256": canonical_hash(stage),
        "validation_manifest": options.manifest.as_posix(),
        "validation_manifest_sha256": sha256_file(options.manifest),
        "selection_id": protocol["selection_id"],
        "ledger": options.ledger.as_posix(),
        "ledger_sha256": sha256_file(options.ledger),
        "condition_count": len(summaries),
        "episode_count": sum(item["episode_count"] for item in summaries),
        "conditions": sorted(
            summaries,
            key=lambda item: item["condition_id"],
        ),
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "unit": "task after averaging repeats and agent models",
        },
        "cap_selection": selections,
        "selected_caps": options.selected_caps.as_posix(),
        "selected_caps_sha256": sha256_file(options.selected_caps),
    }
    audit["audit_id"] = f"validation_{canonical_hash(audit)[:16]}"
    write_json(options.output, audit)
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(render_report(audit), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "audit_id": audit["audit_id"],
                "episode_count": audit["episode_count"],
                "selected_caps": {
                    method: item["selected_cap"]
                    for method, item in selections.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/webshop_event_tower_v2.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/manifests/validation.jsonl"
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(
            "artifacts/experiments/webshop-event-tower-v2/"
            "stage-3-validation/ledger.json"
        ),
    )
    parser.add_argument(
        "--selected-caps",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/stage-3-validation/"
            "selected-caps.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/stage-3-validation/audit.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/stage-3-validation/REPORT.md"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
