from __future__ import annotations

import pytest

from trace2tower.evaluation import (
    aggregate_method,
    audit_result_set,
    paired_bootstrap,
    unresolved_failures,
)
from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.results import EpisodeResult, FinishReason, MethodName


def entry(index: int, benchmark: Benchmark = Benchmark.WEBSHOP) -> ManifestEntry:
    return ManifestEntry(
        benchmark,
        ExperimentSplit.TEST,
        f"{benchmark}:{index}",
        index,
        "source",
        0,
    )


def result(
    index: int,
    method: MethodName,
    score: float,
    *,
    benchmark: Benchmark = Benchmark.WEBSHOP,
    steps: int = 4,
    invalid: int = 0,
    input_tokens: int | None = 10,
    billable_tokens: int | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        "run",
        benchmark,
        ExperimentSplit.TEST,
        method,
        f"{benchmark}:{index}",
        0,
        0,
        score,
        bool(score) if benchmark is Benchmark.ALFWORLD else None,
        steps,
        invalid,
        FinishReason.COMPLETED,
        input_tokens,
        2 if input_tokens is not None else None,
        billable_tokens,
        5,
        ("skill",) if method is not MethodName.NO_SKILL else (),
        20 if method is not MethodName.NO_SKILL else 0,
    )


def test_aggregate_uses_benchmark_metric_and_pooled_invalid_rate() -> None:
    entries = (entry(0), entry(1))
    results = (
        result(0, MethodName.NO_SKILL, 1.0, steps=2, invalid=1),
        result(1, MethodName.NO_SKILL, 0.5, steps=8, invalid=1),
    )
    audit, aggregate = aggregate_method(
        entries,
        results,
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TEST,
        method=MethodName.NO_SKILL,
    )
    assert audit.is_complete
    assert aggregate.primary_metric.value == "mean_reward"
    assert aggregate.primary_metric_mean == 0.75
    assert aggregate.invalid_action_rate == 0.2
    assert aggregate.input_token_coverage == 1
    assert aggregate.billable_token_coverage == 0
    assert aggregate.mean_observed_billable_tokens is None


def test_official_result_protocol_round_trips_for_aggregation() -> None:
    current = result(0, MethodName.NO_SKILL, 0.5)
    assert EpisodeResult.from_record(current.to_record()) == current
    record = current.to_record()
    record["error"] = "not official"
    with pytest.raises(ValueError, match="cannot contain errors"):
        EpisodeResult.from_record(record)


def test_alfworld_aggregate_reports_success_rate() -> None:
    entries = (entry(0, Benchmark.ALFWORLD), entry(1, Benchmark.ALFWORLD))
    _, aggregate = aggregate_method(
        entries,
        (
            result(0, MethodName.NO_SKILL, 1.0, benchmark=Benchmark.ALFWORLD),
            result(1, MethodName.NO_SKILL, 0.0, benchmark=Benchmark.ALFWORLD),
        ),
        benchmark=Benchmark.ALFWORLD,
        split=ExperimentSplit.TEST,
        method=MethodName.NO_SKILL,
    )
    assert aggregate.primary_metric.value == "success_rate"
    assert aggregate.primary_metric_mean == 0.5


def test_audit_rejects_missing_duplicate_or_wrong_scope_results() -> None:
    entries = (entry(0), entry(1))
    duplicate = result(0, MethodName.NO_SKILL, 1.0)
    audit = audit_result_set(
        entries,
        (duplicate, duplicate),
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TEST,
        method=MethodName.NO_SKILL,
    )
    assert not audit.is_complete
    assert audit.missing_keys
    assert audit.duplicate_keys
    with pytest.raises(ValueError, match="complete official"):
        audit.require_complete()


def test_paired_bootstrap_is_deterministic_and_reports_optional_coverage() -> None:
    baseline = (
        result(0, MethodName.NO_SKILL, 0.0, billable_tokens=12),
        result(1, MethodName.NO_SKILL, 1.0, billable_tokens=None),
    )
    candidate = (
        result(0, MethodName.TRACE2TOWER_STATIC, 1.0, billable_tokens=10),
        result(1, MethodName.TRACE2TOWER_STATIC, 1.0, billable_tokens=None),
    )
    first = paired_bootstrap(
        baseline,
        candidate,
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TEST,
        baseline_method=MethodName.NO_SKILL,
        candidate_method=MethodName.TRACE2TOWER_STATIC,
        bootstrap_samples=10000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )
    second = paired_bootstrap(
        baseline,
        candidate,
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TEST,
        baseline_method=MethodName.NO_SKILL,
        candidate_method=MethodName.TRACE2TOWER_STATIC,
        bootstrap_samples=10000,
        bootstrap_seed=42,
        confidence_level=0.95,
    )
    assert first == second
    assert first.mean_difference == 0.5
    assert first.confidence_interval == (0.0, 1.0)
    assert (first.candidate_wins, first.ties, first.candidate_losses) == (1, 1, 0)
    assert first.billable_token_pair_coverage == 0.5
    assert first.mean_billable_token_difference == -2


def test_resolved_errors_are_not_reported_as_final_failures() -> None:
    completed = (result(0, MethodName.SKILLX, 1.0),)
    errors = (
        {
            "benchmark": "webshop",
            "split": "test",
            "method": "skillx",
            "sample_id": "webshop:0",
            "repeat_id": 0,
            "error": "old failure",
        },
        {
            "benchmark": "webshop",
            "split": "test",
            "method": "skillx",
            "sample_id": "webshop:1",
            "repeat_id": 0,
            "error": "first unresolved",
        },
        {
            "benchmark": "webshop",
            "split": "test",
            "method": "skillx",
            "sample_id": "webshop:1",
            "repeat_id": 0,
            "error": "latest unresolved",
        },
    )
    unresolved = unresolved_failures(errors, completed)
    assert len(unresolved) == 1
    assert unresolved[0]["error"] == "latest unresolved"
    assert unresolved[0]["error_attempt_count"] == 2
