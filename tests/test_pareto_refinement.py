from __future__ import annotations

from types import SimpleNamespace

import pytest

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.methods.trace2tower.refinement import (
    LegalMergeProposal,
    LegalPromoteProposal,
    LegalSplitProposal,
    ObjectiveVector,
    RefinementActionPlan,
    RefinementAction,
    RefinementEpisode,
    SkillLevel,
    SkillObjective,
    audit_refinement_evidence,
    build_paired_episode_evidence,
    build_skill_objectives,
    prioritize_merges,
    prioritize_splits,
    rank_promotions,
    rank_skill_objectives,
    select_downweight,
    select_promote,
    validate_execution_contract,
)
from trace2tower.results import MethodName


def episode(
    sample_id: str,
    method: MethodName,
    score: float,
    steps: int,
    cost: int | None,
    skill_ids: tuple[str, ...] = (),
) -> RefinementEpisode:
    return RefinementEpisode(
        "run",
        Benchmark.WEBSHOP,
        ExperimentSplit.TRAIN,
        method,
        sample_id,
        0,
        score,
        steps,
        cost,
        skill_ids,
    )


def objective(
    skill_id: str,
    values: tuple[float, float, float, float],
    exposure: int = 1,
) -> SkillObjective:
    from trace2tower.methods.trace2tower.refinement import EpisodePairKey

    keys = tuple(
        EpisodePairKey(Benchmark.WEBSHOP, f"sample-{index}", 0)
        for index in range(exposure)
    )
    return SkillObjective(
        skill_id,
        Benchmark.WEBSHOP,
        SkillLevel.MID,
        1,
        ObjectiveVector(*values),
        exposure,
        keys,
    )


def test_guarded_savings_do_not_reward_faster_cheaper_regression() -> None:
    baseline = (
        episode("a", MethodName.NO_SKILL, 1.0, 10, 100),
        episode("b", MethodName.NO_SKILL, 0.5, 10, 100),
    )
    skill = (
        episode("a", MethodName.TRACE2TOWER, 0.5, 5, 50, ("mid",)),
        episode("b", MethodName.TRACE2TOWER, 0.5, 12, 120, ("mid",)),
    )
    result = build_skill_objectives(
        baseline,
        skill,
        {"mid": SkillLevel.MID},
        refinement_round=1,
    )[0]
    assert result.objective_vector.performance_level == 0.5
    assert result.objective_vector.paired_reward_gain == -0.25
    assert result.objective_vector.guarded_step_saving == pytest.approx(-0.1)
    assert result.objective_vector.guarded_cost_saving == pytest.approx(-0.1)


def test_missing_chat_cost_is_audited_and_blocks_ranking() -> None:
    baseline = (episode("a", MethodName.NO_SKILL, 1.0, 10, None),)
    skill = (
        episode("a", MethodName.TRACE2TOWER, 1.0, 8, None, ("mid",)),
    )
    audit = audit_refinement_evidence(
        baseline, skill, {"mid": SkillLevel.MID}
    )
    assert not audit.is_complete
    assert audit.skill_methods == (MethodName.TRACE2TOWER,)
    assert audit.missing_baseline_chat_token_keys
    assert audit.missing_skill_chat_token_keys
    assert audit.to_record()["paired_episode_keys"] == [
        {"benchmark": "webshop", "sample_id": "a", "repeat_id": 0}
    ]
    with pytest.raises(ValueError, match="evidence is incomplete"):
        build_skill_objectives(
            baseline,
            skill,
            {"mid": SkillLevel.MID},
            refinement_round=1,
        )


def test_paired_evidence_preserves_partial_metrics_without_inventing_cost() -> None:
    baseline = (episode("a", MethodName.NO_SKILL, 1.0, 4, None),)
    skill = (
        episode("a", MethodName.TRACE2TOWER, 1.0, 9, None, ("mid",)),
    )
    evidence = build_paired_episode_evidence(baseline, skill)[0]
    assert evidence.paired_reward_gain == 0
    assert evidence.guarded_step_saving == -1.25
    assert evidence.raw_cost_saving is None
    assert evidence.guarded_cost_saving is None


def test_execution_contract_rejects_snapshot_or_agent_model_mismatch() -> None:
    baseline = (episode("a", MethodName.NO_SKILL, 1.0, 4, None),)
    skill = (
        episode("a", MethodName.TRACE2TOWER, 1.0, 9, None, ("mid",)),
    )
    metadata = (
        {
            "run_id": "run",
            "benchmark": "webshop",
            "method": "no_skill",
            "agent_model": "flash",
        },
    )
    report = {
        "run_id": "run",
        "benchmark": "webshop",
        "method": "trace2tower",
        "agent_model": "flash",
        "snapshot_id": "tower_one",
    }
    contract = validate_execution_contract(
        tower_snapshot_id="tower_one",
        benchmark=Benchmark.WEBSHOP,
        baseline_episodes=baseline,
        skill_episodes=skill,
        baseline_metadata=metadata,
        skill_report=report,
    )
    assert contract.to_record()["is_valid"] is True
    with pytest.raises(ValueError, match="fair execution contract"):
        validate_execution_contract(
            tower_snapshot_id="tower_two",
            benchmark=Benchmark.WEBSHOP,
            baseline_episodes=baseline,
            skill_episodes=skill,
            baseline_metadata=metadata,
            skill_report={**report, "agent_model": "pro"},
        )


def test_non_dominated_sort_records_fronts_and_relationships() -> None:
    ranked = rank_skill_objectives(
        (
            objective("a", (1.0, 0.0, 0.0, 0.0)),
            objective("b", (0.9, -0.1, -0.1, -0.1)),
            objective("c", (0.8, 0.2, 0.2, 0.2)),
        )
    )
    by_id = {item.skill_id: item for item in ranked}
    assert by_id["a"].pareto_front_rank == 1
    assert by_id["c"].pareto_front_rank == 1
    assert by_id["b"].pareto_front_rank == 2
    assert by_id["b"].dominated_by == ("a",)
    assert by_id["a"].dominates == ("b",)


def test_split_and_downweight_prioritize_deep_weak_evidence_rich_skill() -> None:
    ranked = rank_skill_objectives(
        (
            objective("strong", (1.0, 0.1, 0.0, 0.0), exposure=1),
            objective("weak-a", (0.5, -0.2, -0.1, -0.1), exposure=2),
            objective("weak-b", (0.5, -0.2, -0.1, -0.1), exposure=3),
        )
    )
    by_id = {item.skill_id: item for item in ranked}
    proposals = (
        LegalSplitProposal("split-a", "weak-a"),
        LegalSplitProposal("split-b", "weak-b"),
    )
    assert prioritize_splits(proposals, by_id)[0].source_skill_id == "weak-b"
    update = select_downweight(ranked, set(by_id))
    assert update.skill_id == "weak-b"
    assert update.new_status.value == "downweighted"


def test_merge_protects_mutually_non_dominating_first_front() -> None:
    ranked = rank_skill_objectives(
        (
            objective("fast", (0.8, 0.0, 0.5, 0.5)),
            objective("quality", (1.0, 0.2, 0.0, 0.0)),
            objective("weak", (0.5, -0.2, -0.2, -0.2)),
        )
    )
    by_id = {item.skill_id: item for item in ranked}
    result = prioritize_merges(
        (
            LegalMergeProposal("protected", "fast", "quality", 0.9, 0.1),
            LegalMergeProposal("eligible", "quality", "weak", 0.8, 0.2),
        ),
        by_id,
    )
    assert tuple(item.proposal_id for item in result.pareto_protected) == (
        "protected",
    )
    assert tuple(item.proposal_id for item in result.eligible) == ("eligible",)


def test_promote_uses_exposure_weighted_front_then_structural_ties() -> None:
    ranked = rank_skill_objectives(
        (
            objective("a", (1.0, 0.0, 0.0, 0.0), exposure=3),
            objective("b", (0.0, 1.0, 1.0, 1.0), exposure=1),
            objective("c", (0.7, 0.7, 0.7, 0.7), exposure=2),
        )
    )
    by_id = {item.skill_id: item for item in ranked}
    proposals = rank_promotions(
        (
            LegalPromoteProposal(
                "path-ab", Benchmark.WEBSHOP, 1, ("a", "b"), 2.0, 0.5
            ),
            LegalPromoteProposal(
                "path-bc", Benchmark.WEBSHOP, 1, ("b", "c"), 3.0, 0.4
            ),
        ),
        by_id,
    )
    by_path = {item.proposal.path_id: item for item in proposals}
    assert by_path["path-ab"].path_objective.performance_level == 0.75
    assert select_promote(proposals).proposal.path_id == "path-bc"


def test_runtime_preserves_only_explicit_billable_usage() -> None:
    explicit = CommonLLMRuntime._usage(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            billable_tokens=7,
            prompt_tokens_details=None,
        )
    )
    extra = CommonLLMRuntime._usage(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            prompt_tokens_details=None,
            model_extra={"billable_tokens": 8},
        )
    )
    absent = CommonLLMRuntime._usage(
        SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            prompt_tokens_details=None,
        )
    )
    assert explicit.billable_tokens == 7
    assert extra.billable_tokens == 8
    assert absent.billable_tokens is None


def test_refinement_action_plan_maps_original_four_actions() -> None:
    from trace2tower.methods.trace2tower.refinement import (
        LifecycleAction,
        LifecycleUpdate,
        LegalSplitProposal,
        SkillStatus,
    )

    plan = RefinementActionPlan(
        split=LegalSplitProposal("split:mid", "mid"),
        merge=None,
        promote=None,
        downweight=LifecycleUpdate(
            "mid", LifecycleAction.DOWNWEIGHT, SkillStatus.ACTIVE,
            SkillStatus.DOWNWEIGHTED, 1, 2,
        ),
    )
    record = plan.to_record()
    assert set(record) == {
        RefinementAction.SPLIT.value,
        RefinementAction.MERGE.value,
        RefinementAction.PROMOTE.value,
        RefinementAction.DOWNWEIGHT.value,
    }
    assert record["split"]["proposal_id"] == "split:mid"
    assert record["downweight"]["new_status"] == "downweighted"


def test_primary_pareto_rank_ignores_chat_cost() -> None:
    ranked = rank_skill_objectives(
        (
            objective("cheap", (0.8, 0.1, 0.1, 0.9)),
            objective("expensive", (0.8, 0.1, 0.1, -0.9)),
        )
    )
    assert {item.pareto_front_rank for item in ranked} == {1}
