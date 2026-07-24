from __future__ import annotations

from trace2tower.core.trajectory import StepRecord
from trace2tower.methods.trace2tower.adapters.webshop.events import classify_webshop_steps
from trace2tower.methods.trace2tower.core.models import WebShopEventType


def step(
    index: int,
    observation: str,
    action_name: str,
    action_arguments: dict,
    next_observation: str,
    admissible_actions: tuple[str, ...] = (),
) -> StepRecord:
    return StepRecord(
        step_index=index,
        observation=observation,
        action_name=action_name,
        action_arguments=action_arguments,
        next_observation=next_observation,
        reward=0,
        done=False,
        valid_action=True,
        admissible_actions=admissible_actions,
        clickable_types={},
    )


def test_item_prev_is_candidate_backtracking() -> None:
    steps = (
        step(
            0,
            "WebShop search page.",
            "search_action",
            {"keywords": "running shoes"},
            "Search results page 1:",
        ),
        step(
            1,
            "Search results page 1:",
            "click_action",
            {"value": "SHOE123"},
            "Product: Running shoes",
            ("SHOE123",),
        ),
        step(
            2,
            "Product: Running shoes",
            "click_action",
            {"value": "< Prev"},
            "Search results page 1:",
            ("< Prev",),
        ),
    )

    assert classify_webshop_steps(steps)[-1] is WebShopEventType.CANDIDATE_BACKTRACKING
