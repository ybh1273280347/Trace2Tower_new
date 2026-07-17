from __future__ import annotations

import asyncio

from scripts.experiments.run.run_matrix import EXECUTABLE_METHODS
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.core.results import MethodName
from trace2tower.methods.expert_crafted import ExpertCraftedSkillProvider


def test_method_vocabulary_matches_the_experiment_contract() -> None:
    assert tuple(MethodName) == (
        MethodName.NO_SKILL,
        MethodName.SKILLX,
        MethodName.EXPEL,
        MethodName.EXPERT_CRAFTED_SKILLS,
        MethodName.TRACE2TOWER,
    )
    assert MethodName.EXPEL not in EXECUTABLE_METHODS


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
