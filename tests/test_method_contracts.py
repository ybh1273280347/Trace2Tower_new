from __future__ import annotations

import asyncio

from scripts.experiments.run.run_matrix import EXECUTABLE_METHODS
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.core.results import MethodName
from trace2tower.methods.expert_crafted import ExpertCraftedSkillProvider
from trace2tower.methods.expel.induction import (
    RuleOperation,
    RuleOperationName,
    apply_rule_operations,
    parse_rule_operations,
)


def test_method_vocabulary_matches_the_experiment_contract() -> None:
    assert tuple(MethodName) == (
        MethodName.NO_SKILL,
        MethodName.SKILLX,
        MethodName.EXPEL,
        MethodName.EXPERT_CRAFTED_SKILLS,
        MethodName.TRACE2TOWER,
    )
    assert MethodName.EXPEL in EXECUTABLE_METHODS


def test_expert_crafted_provider_returns_the_frozen_context() -> None:
    provider = ExpertCraftedSkillProvider("expert_one", "  Use the expert procedure.  ")
    selection = asyncio.run(
        provider.select(
            "complete the task",
            EnvironmentState("state", (), {}, False, 0, False, True),
        )
    )

    assert selection.skill_ids == ("expert_one",)
    assert selection.context == "Use the expert procedure."


def test_expel_rule_operations_follow_the_native_update_contract() -> None:
    operations = parse_rule_operations(
        "AGREE 1: Keep the target constraints explicit.\n"
        "EDIT 2: Verify all prerequisites before the terminal action.\n"
        "ADD 3: Recover by changing strategy after repeated invalid actions."
    )
    assert operations == (
        RuleOperation(RuleOperationName.AGREE, 1, "Keep the target constraints explicit."),
        RuleOperation(
            RuleOperationName.EDIT,
            2,
            "Verify all prerequisites before the terminal action.",
        ),
        RuleOperation(
            RuleOperationName.ADD,
            3,
            "Recover by changing strategy after repeated invalid actions.",
        ),
    )
    assert apply_rule_operations(
        (
            ("Keep the target constraints explicit.", 2),
            ("Act only after checking prerequisites.", 1),
        ),
        operations,
    ) == (
        ("Keep the target constraints explicit.", 3),
        ("Verify all prerequisites before the terminal action.", 2),
        ("Recover by changing strategy after repeated invalid actions.", 2),
    )
