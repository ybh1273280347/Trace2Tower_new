from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import fmean

import numpy as np

from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.results import EpisodeResult, FinishReason, MethodName


class PrimaryMetricName(StrEnum):
    SUCCESS_RATE = "success_rate"
    MEAN_REWARD = "mean_reward"


@dataclass(frozen=True, slots=True, order=True)
class EvaluationKey:
    benchmark: Benchmark
    split: ExperimentSplit
    sample_id: str
    repeat_id: int

    @classmethod
    def from_manifest(cls, entry: ManifestEntry) -> EvaluationKey:
        return cls(entry.benchmark, entry.split, entry.sample_id, entry.repeat_id)

    @classmethod
    def from_result(cls, result: EpisodeResult) -> EvaluationKey:
        return cls(result.benchmark, result.split, result.sample_id, result.repeat_id)

    def to_record(self) -> dict:
        return {
            "benchmark": self.benchmark.value,
            "split": self.split.value,
            "sample_id": self.sample_id,
            "repeat_id": self.repeat_id,
        }


@dataclass(frozen=True, slots=True)
class ResultSetAudit:
    benchmark: Benchmark
    split: ExperimentSplit
    method: MethodName
    expected_count: int
    result_count: int
    official_result_count: int
    official_result_coverage: float
    run_ids: tuple[str, ...]
    missing_keys: tuple[EvaluationKey, ...]
    unexpected_keys: tuple[EvaluationKey, ...]
    duplicate_keys: tuple[EvaluationKey, ...]
    scope_mismatch_count: int

    @property
    def is_complete(self) -> bool:
        return (
            self.expected_count > 0
            and self.official_result_coverage == 1
            and self.result_count == self.expected_count
            and not self.missing_keys
            and not self.unexpected_keys
            and not self.duplicate_keys
            and self.scope_mismatch_count == 0
        )

    def require_complete(self) -> None:
        if not self.is_complete:
            raise ValueError("result set does not have complete official manifest coverage")

    def to_record(self) -> dict:
        return {
            "benchmark": self.benchmark.value,
            "split": self.split.value,
            "method": self.method.value,
            "expected_count": self.expected_count,
            "result_count": self.result_count,
            "official_result_count": self.official_result_count,
            "official_result_coverage": self.official_result_coverage,
            "run_ids": self.run_ids,
            "missing_keys": [key.to_record() for key in self.missing_keys],
            "unexpected_keys": [key.to_record() for key in self.unexpected_keys],
            "duplicate_keys": [key.to_record() for key in self.duplicate_keys],
            "scope_mismatch_count": self.scope_mismatch_count,
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True, slots=True)
class ConstructionCost:
    method: MethodName
    construction_llm_calls: int
    construction_input_tokens: int
    construction_output_tokens: int
    construction_billable_tokens: int | None
    construction_latency_ms: int
    embedding_calls: int
    embedding_input_count: int
    final_skill_count: int
    final_mid_skill_count: int
    final_high_skill_count: int

    def __post_init__(self) -> None:
        values = (
            self.construction_llm_calls,
            self.construction_input_tokens,
            self.construction_output_tokens,
            self.construction_latency_ms,
            self.embedding_calls,
            self.embedding_input_count,
            self.final_skill_count,
            self.final_mid_skill_count,
            self.final_high_skill_count,
        )
        if any(value < 0 for value in values):
            raise ValueError("construction cost fields must be non-negative")
        if (
            self.construction_billable_tokens is not None
            and self.construction_billable_tokens < 0
        ):
            raise ValueError("construction billable tokens must be non-negative")

    @classmethod
    def from_record(cls, record: Mapping) -> ConstructionCost:
        return cls(
            method=MethodName(record["method"]),
            construction_llm_calls=int(record["construction_llm_calls"]),
            construction_input_tokens=int(record["construction_input_tokens"]),
            construction_output_tokens=int(record["construction_output_tokens"]),
            construction_billable_tokens=(
                int(record["construction_billable_tokens"])
                if record.get("construction_billable_tokens") is not None
                else None
            ),
            construction_latency_ms=int(record["construction_latency_ms"]),
            embedding_calls=int(record["embedding_calls"]),
            embedding_input_count=int(record["embedding_input_count"]),
            final_skill_count=int(record["final_skill_count"]),
            final_mid_skill_count=int(record["final_mid_skill_count"]),
            final_high_skill_count=int(record["final_high_skill_count"]),
        )

    def to_record(self) -> dict:
        return {**asdict(self), "method": self.method.value}


@dataclass(frozen=True, slots=True)
class MethodAggregate:
    benchmark: Benchmark
    split: ExperimentSplit
    method: MethodName
    run_ids: tuple[str, ...]
    episode_count: int
    official_result_coverage: float
    primary_metric: PrimaryMetricName
    primary_metric_mean: float
    completion_rate: float
    mean_steps: float
    total_steps: int
    total_invalid_actions: int
    invalid_action_rate: float
    mean_latency_ms: float
    skill_injection_rate: float
    mean_skill_context_chars: float
    input_token_coverage: float
    observed_input_tokens: int
    mean_observed_input_tokens: float | None
    output_token_coverage: float
    observed_output_tokens: int
    mean_observed_output_tokens: float | None
    billable_token_coverage: float
    observed_billable_tokens: int
    mean_observed_billable_tokens: float | None
    construction_cost: ConstructionCost | None

    def to_record(self) -> dict:
        return {
            **asdict(self),
            "benchmark": self.benchmark.value,
            "split": self.split.value,
            "method": self.method.value,
            "primary_metric": self.primary_metric.value,
            "construction_cost": (
                self.construction_cost.to_record() if self.construction_cost else None
            ),
        }


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    benchmark: Benchmark
    split: ExperimentSplit
    baseline_method: MethodName
    candidate_method: MethodName
    pair_count: int
    task_count: int
    primary_metric: PrimaryMetricName
    mean_difference: float
    confidence_level: float
    confidence_interval: tuple[float, float]
    bootstrap_samples: int
    bootstrap_seed: int
    candidate_wins: int
    ties: int
    candidate_losses: int
    mean_step_difference: float
    mean_invalid_action_difference: float
    input_token_pair_coverage: float
    mean_input_token_difference: float | None
    output_token_pair_coverage: float
    mean_output_token_difference: float | None
    billable_token_pair_coverage: float
    mean_billable_token_difference: float | None

    def to_record(self) -> dict:
        return {
            **asdict(self),
            "benchmark": self.benchmark.value,
            "split": self.split.value,
            "baseline_method": self.baseline_method.value,
            "candidate_method": self.candidate_method.value,
            "primary_metric": self.primary_metric.value,
        }


def audit_result_set(
    expected_entries: Sequence[ManifestEntry],
    results: Sequence[EpisodeResult],
    *,
    benchmark: Benchmark,
    split: ExperimentSplit,
    method: MethodName,
) -> ResultSetAudit:
    expected_keys = {EvaluationKey.from_manifest(entry) for entry in expected_entries}
    result_keys = [EvaluationKey.from_result(result) for result in results]
    counts = Counter(result_keys)
    scope_mismatches = sum(
        result.benchmark is not benchmark
        or result.split is not split
        or result.method is not method
        for result in results
    )
    official_keys = {
        EvaluationKey.from_result(result)
        for result in results
        if result.benchmark is benchmark
        and result.split is split
        and result.method is method
        and result.error is None
    }
    return ResultSetAudit(
        benchmark=benchmark,
        split=split,
        method=method,
        expected_count=len(expected_keys),
        result_count=len(results),
        official_result_count=len(official_keys & expected_keys),
        official_result_coverage=(
            len(official_keys & expected_keys) / len(expected_keys)
            if expected_keys
            else 0
        ),
        run_ids=tuple(sorted({result.run_id for result in results})),
        missing_keys=tuple(sorted(expected_keys - set(result_keys))),
        unexpected_keys=tuple(sorted(set(result_keys) - expected_keys)),
        duplicate_keys=tuple(
            sorted(key for key, count in counts.items() if count > 1)
        ),
        scope_mismatch_count=scope_mismatches,
    )


def aggregate_method(
    expected_entries: Sequence[ManifestEntry],
    results: Sequence[EpisodeResult],
    *,
    benchmark: Benchmark,
    split: ExperimentSplit,
    method: MethodName,
    construction_cost: ConstructionCost | None = None,
) -> tuple[ResultSetAudit, MethodAggregate]:
    audit = audit_result_set(
        expected_entries,
        results,
        benchmark=benchmark,
        split=split,
        method=method,
    )
    audit.require_complete()
    if construction_cost and construction_cost.method is not method:
        raise ValueError("construction cost method does not match aggregate method")
    primary_metric = (
        PrimaryMetricName.SUCCESS_RATE
        if benchmark is Benchmark.ALFWORLD
        else PrimaryMetricName.MEAN_REWARD
    )
    return audit, MethodAggregate(
        benchmark=benchmark,
        split=split,
        method=method,
        run_ids=audit.run_ids,
        episode_count=len(results),
        official_result_coverage=audit.official_result_coverage,
        primary_metric=primary_metric,
        primary_metric_mean=fmean(result.primary_score for result in results),
        completion_rate=sum(
            result.finish_reason is FinishReason.COMPLETED for result in results
        )
        / len(results),
        mean_steps=fmean(result.steps for result in results),
        total_steps=sum(result.steps for result in results),
        total_invalid_actions=sum(result.invalid_actions for result in results),
        invalid_action_rate=sum(result.invalid_actions for result in results)
        / max(sum(result.steps for result in results), 1),
        mean_latency_ms=fmean(result.latency_ms for result in results),
        skill_injection_rate=sum(bool(result.skill_ids) for result in results)
        / len(results),
        mean_skill_context_chars=fmean(
            result.skill_context_chars for result in results
        ),
        **_usage_summary(results, "input_tokens", "input"),
        **_usage_summary(results, "output_tokens", "output"),
        **_usage_summary(results, "billable_tokens", "billable"),
        construction_cost=construction_cost,
    )


def paired_bootstrap(
    baseline_results: Sequence[EpisodeResult],
    candidate_results: Sequence[EpisodeResult],
    *,
    benchmark: Benchmark,
    split: ExperimentSplit,
    baseline_method: MethodName,
    candidate_method: MethodName,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> PairwiseComparison:
    if bootstrap_samples <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid paired bootstrap configuration")
    if any(
        result.benchmark is not benchmark
        or result.split is not split
        or result.method is not baseline_method
        for result in baseline_results
    ) or any(
        result.benchmark is not benchmark
        or result.split is not split
        or result.method is not candidate_method
        for result in candidate_results
    ):
        raise ValueError("paired bootstrap inputs do not match the requested scope")
    baseline = {EvaluationKey.from_result(result): result for result in baseline_results}
    candidate = {
        EvaluationKey.from_result(result): result for result in candidate_results
    }
    if len(baseline) != len(baseline_results) or len(candidate) != len(candidate_results):
        raise ValueError("paired bootstrap inputs contain duplicate episode keys")
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired bootstrap requires identical non-empty episode keys")
    ordered_keys = tuple(sorted(baseline))
    episode_differences = np.asarray(
        [candidate[key].primary_score - baseline[key].primary_score for key in ordered_keys],
        dtype=np.float64,
    )
    differences_by_sample = {}
    for key, difference in zip(ordered_keys, episode_differences, strict=True):
        differences_by_sample.setdefault(key.sample_id, []).append(float(difference))
    task_differences = np.asarray(
        [fmean(values) for _, values in sorted(differences_by_sample.items())],
        dtype=np.float64,
    )
    rng = np.random.default_rng(bootstrap_seed)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    batch_size = 512
    for start in range(0, bootstrap_samples, batch_size):
        end = min(start + batch_size, bootstrap_samples)
        indices = rng.integers(
            0,
            len(task_differences),
            size=(end - start, len(task_differences)),
        )
        means[start:end] = task_differences[indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    interval = np.quantile(means, (alpha, 1 - alpha), method="linear")
    metric = (
        PrimaryMetricName.SUCCESS_RATE
        if benchmark is Benchmark.ALFWORLD
        else PrimaryMetricName.MEAN_REWARD
    )
    input_summary = _paired_optional_difference(
        baseline, candidate, ordered_keys, "input_tokens"
    )
    output_summary = _paired_optional_difference(
        baseline, candidate, ordered_keys, "output_tokens"
    )
    billable_summary = _paired_optional_difference(
        baseline, candidate, ordered_keys, "billable_tokens"
    )
    return PairwiseComparison(
        benchmark=benchmark,
        split=split,
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        pair_count=len(ordered_keys),
        task_count=len(task_differences),
        primary_metric=metric,
        mean_difference=float(task_differences.mean()),
        confidence_level=confidence_level,
        confidence_interval=(float(interval[0]), float(interval[1])),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        candidate_wins=int(np.sum(episode_differences > 0)),
        ties=int(np.sum(episode_differences == 0)),
        candidate_losses=int(np.sum(episode_differences < 0)),
        mean_step_difference=fmean(
            candidate[key].steps - baseline[key].steps for key in ordered_keys
        ),
        mean_invalid_action_difference=fmean(
            candidate[key].invalid_actions - baseline[key].invalid_actions
            for key in ordered_keys
        ),
        input_token_pair_coverage=input_summary[0],
        mean_input_token_difference=input_summary[1],
        output_token_pair_coverage=output_summary[0],
        mean_output_token_difference=output_summary[1],
        billable_token_pair_coverage=billable_summary[0],
        mean_billable_token_difference=billable_summary[1],
    )


def unresolved_failures(
    error_records: Sequence[Mapping],
    completed_results: Sequence[EpisodeResult],
) -> tuple[dict, ...]:
    completed = {
        (
            result.benchmark.value,
            result.split.value,
            result.method.value,
            result.sample_id,
            result.repeat_id,
        )
        for result in completed_results
    }
    latest = {}
    attempt_counts = Counter()
    for record in error_records:
        key = (
            str(record["benchmark"]),
            str(record["split"]),
            str(record["method"]),
            str(record["sample_id"]),
            int(record["repeat_id"]),
        )
        error = record.get("error")
        if not isinstance(error, str) or not error:
            raise ValueError("failure record requires a non-empty error")
        if key not in completed:
            latest[key] = dict(record)
            attempt_counts[key] += 1
    return tuple(
        {
            **latest[key],
            "error_attempt_count": attempt_counts[key],
        }
        for key in sorted(latest)
    )


def _usage_summary(
    results: Sequence[EpisodeResult], field: str, prefix: str
) -> dict:
    values = [getattr(result, field) for result in results if getattr(result, field) is not None]
    return {
        f"{prefix}_token_coverage": len(values) / len(results),
        f"observed_{prefix}_tokens": sum(values),
        f"mean_observed_{prefix}_tokens": fmean(values) if values else None,
    }


def _paired_optional_difference(
    baseline: Mapping[EvaluationKey, EpisodeResult],
    candidate: Mapping[EvaluationKey, EpisodeResult],
    keys: Sequence[EvaluationKey],
    field: str,
) -> tuple[float, float | None]:
    differences = [
        getattr(candidate[key], field) - getattr(baseline[key], field)
        for key in keys
        if getattr(candidate[key], field) is not None
        and getattr(baseline[key], field) is not None
    ]
    return len(differences) / len(keys), fmean(differences) if differences else None
