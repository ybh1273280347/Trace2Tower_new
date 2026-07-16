from __future__ import annotations

import asyncio
from types import SimpleNamespace

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.labeled_mid_provider import (
    LabeledMidDiagnosticProvider,
    LabeledMidTask,
)
from trace2tower.methods.trace2tower.models import PrimitiveAction
from trace2tower.methods.trace2tower.skills import MidSkillCard


def mid_card(skill_id: str, action: PrimitiveAction) -> MidSkillCard:
    return MidSkillCard(
        skill_id,
        (f"trajectory:segment:{skill_id}",),
        skill_id,
        "Use in the labeled phase.",
        ("Execute the phase.",),
        (),
        (action,),
    )


class FakeBaseProvider:
    snapshot = SimpleNamespace(benchmark=Benchmark.ALFWORLD)

    def __init__(self):
        self.mid_cards = {
            "mid_manipulate": mid_card("mid_manipulate", PrimitiveAction.PICK),
            "mid_clean": mid_card("mid_clean", PrimitiveAction.CLEAN),
        }

    async def select_task(self, task_goal, state) -> SkillSelection:
        return SkillSelection(("high",), "High")


def test_labeled_mid_moves_from_transformation_to_placement() -> None:
    goal = "Clean the ladle and put it on the counter."
    provider = LabeledMidDiagnosticProvider(
        FakeBaseProvider(),
        (LabeledMidTask(goal, "ladle", "countertop", "mid_clean"),),
        manipulation_mid_id="mid_manipulate",
    )
    initial = EnvironmentState("room", (), {}, False, 0, False, True)
    asyncio.run(provider.select_task(goal, initial))

    carrying = EnvironmentState(
        "You pick up the ladle 1.",
        ("go to sinkbasin 1", "move ladle 1 to countertop 1"),
        {},
        False,
        0,
        False,
        True,
    )
    assert asyncio.run(provider.select_state(goal, carrying)).skill_ids == ("mid_clean",)

    cleaned = EnvironmentState(
        "You clean the ladle 1 using the sinkbasin 1.",
        ("go to countertop 1", "move ladle 1 to sinkbasin 1"),
        {},
        False,
        0,
        False,
        True,
    )
    assert asyncio.run(provider.select_state(goal, cleaned)).skill_ids == ()

    destination = EnvironmentState(
        "You arrive at countertop 1.",
        ("move ladle 1 to countertop 1",),
        {},
        False,
        0,
        False,
        True,
    )
    assert asyncio.run(provider.select_state(goal, destination)).skill_ids == (
        "mid_manipulate",
    )
