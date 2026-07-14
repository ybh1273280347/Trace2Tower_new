from __future__ import annotations

from scripts.experiments.analyze.analyze_webshop_stage3_validation import select_cap
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.results import EpisodeResult, FinishReason, MethodName


def result(
    method: MethodName,
    model: str,
    cap: int,
    sample_index: int,
    repeat_id: int,
    score: float,
) -> EpisodeResult:
    return EpisodeResult(
        run_id=f"{model}-cap{cap}",
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.DEV,
        method=method,
        sample_id=f"webshop:{sample_index}",
        repeat_id=repeat_id,
        shard_id=0,
        primary_score=score,
        success=None,
        steps=4,
        invalid_actions=0,
        finish_reason=FinishReason.COMPLETED,
        input_tokens=100,
        output_tokens=10,
        billable_tokens=None,
        latency_ms=10,
        skill_ids=("mid",),
        skill_context_chars=100,
        context_skill_ids=("mid",),
    )


def test_cap_selection_chooses_smallest_bootstrap_equivalent_candidate() -> None:
    method = MethodName.TRACE2TOWER
    scores = {
        3: {0: 0.6, 1: 0.6},
        5: {0: 0.8, 1: 0.5},
        8: {0: 0.4, 1: 0.2},
    }
    results = {
        (method, model, cap): tuple(
            result(method, model, cap, sample_index, repeat_id, score)
            for sample_index, score in task_scores.items()
            for repeat_id in range(3)
        )
        for model in ("deepseek-v4-flash", "deepseek-v4-pro")
        for cap, task_scores in scores.items()
    }

    selection = select_cap(method, results)

    assert selection["empirical_best_cap"] == 5
    assert selection["selected_cap"] == 3
    assert selection["comparisons_to_empirical_best"]["3"]["includes_zero"]
    assert not selection["comparisons_to_empirical_best"]["8"]["includes_zero"]
