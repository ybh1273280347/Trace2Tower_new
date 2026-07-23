from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from trace2tower.benchmarks.models import ClickableKind
from trace2tower.components.llm_runtime import LLMUsage
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory, StepRecord
from trace2tower.methods.trace2tower.adapters.alfworld.actions import (
    parse_alfworld_action,
)
from trace2tower.methods.trace2tower.adapters.alfworld.events import (
    alfworld_applicable_events,
    alfworld_goal_events,
    alfworld_segment_signature,
    classify_alfworld_transitions,
    segment_alfworld_trajectory,
)
from trace2tower.methods.trace2tower.adapters.transitions import (
    build_benchmark_transitions,
)
from trace2tower.methods.trace2tower.adapters.webshop.actions import (
    parse_webshop_action,
)
from trace2tower.methods.trace2tower.adapters.webshop.events import (
    WebShopEventClassifier,
    classify_webshop_steps,
    segment_webshop_trajectory,
    webshop_segment_signature,
)
from trace2tower.methods.trace2tower.core.models import (
    AlfworldEventType,
    PrimitiveAction,
    SegmentInstance,
    StepTransition,
    WebShopEventType,
    WebShopPageType,
)
from trace2tower.methods.trace2tower.preprocessing.segmentation import (
    calibrate_segmentation_penalty,
    segment_boundaries,
)
from trace2tower.methods.trace2tower.preprocessing.transition_encoder import TransitionEncoder


@pytest.mark.parametrize(
    ("action", "expected"),
    (
        ("go to fridge 1", PrimitiveAction.GOTO),
        ("take tomato 1 from fridge 1", PrimitiveAction.PICK),
        ("put tomato 1 in/on microwave 1", PrimitiveAction.PUT),
        ("move tomato 1 to garbagecan 1", PrimitiveAction.PUT),
        ("open fridge 1", PrimitiveAction.OPEN),
        ("close fridge 1", PrimitiveAction.CLOSE),
        ("use desklamp 1", PrimitiveAction.TOGGLE),
        ("heat tomato 1 with microwave 1", PrimitiveAction.HEAT),
        ("clean egg 1 with sinkbasin 1", PrimitiveAction.CLEAN),
        ("cool apple 1 with fridge 1", PrimitiveAction.COOL),
        ("slice apple 1 with knife 1", PrimitiveAction.SLICE),
        ("inventory", PrimitiveAction.INVENTORY),
        ("examine shelf 1", PrimitiveAction.EXAMINE),
        ("look", PrimitiveAction.LOOK),
    ),
)
def test_parse_alfworld_official_commands(action: str, expected: PrimitiveAction) -> None:
    assert parse_alfworld_action("take_action", {"action": action}) is expected


def test_action_parsers_reject_wrong_tool_or_argument() -> None:
    assert (
        parse_alfworld_action("take_action", {"action": "looking glass"}) is PrimitiveAction.INVALID
    )
    assert parse_alfworld_action("click_action", {"action": "look"}) is PrimitiveAction.INVALID
    assert parse_webshop_action("search_action", {"keywords": "rug"}) is PrimitiveAction.SEARCH
    assert parse_webshop_action("click_action", {"value": "Buy Now"}) is PrimitiveAction.CLICK
    assert parse_webshop_action("click_action", {"keywords": "rug"}) is PrimitiveAction.INVALID


def alfworld_step(index: int, action: str) -> StepRecord:
    return StepRecord(
        step_index=index,
        observation=f"before-{index}",
        action_name="take_action",
        action_arguments={"action": action},
        next_observation=f"after-{index}",
        reward=0,
        done=False,
        valid_action=True,
        admissible_actions=(),
        clickable_types={},
    )


def test_alfworld_uses_official_events_and_task_entity_signatures() -> None:
    steps = (
        alfworld_step(0, "go to countertop 1"),
        alfworld_step(1, "go to fridge 2"),
        alfworld_step(2, "open fridge 2"),
        alfworld_step(3, "take apple 1 from fridge 2"),
        alfworld_step(4, "heat apple 1 with microwave 1"),
        alfworld_step(5, "put apple 1 in/on countertop 1"),
    )
    trajectory = EpisodeTrajectory(
        run_id="test-run",
        benchmark=Benchmark.ALFWORLD,
        split=ExperimentSplit.TRAIN,
        method=MethodName.NO_SKILL,
        sample_id="alfworld:train:trial",
        repeat_id=0,
        task_goal="put a hot apple on a countertop",
        steps=steps,
        primary_score=1,
        finish_reason=FinishReason.COMPLETED,
    )
    transitions = build_benchmark_transitions(trajectory)

    assert classify_alfworld_transitions(transitions) == (
        AlfworldEventType.GOTO_LOCATION,
        AlfworldEventType.GOTO_LOCATION,
        AlfworldEventType.OPEN_OBJECT,
        AlfworldEventType.PICKUP_OBJECT,
        AlfworldEventType.HEAT_OBJECT,
        AlfworldEventType.PUT_OBJECT,
    )
    segments = segment_alfworld_trajectory(trajectory, transitions)
    assert [(segment.event_type, segment.start_step, segment.end_step) for segment in segments] == [
        (AlfworldEventType.GOTO_LOCATION, 0, 1),
        (AlfworldEventType.OPEN_OBJECT, 2, 2),
        (AlfworldEventType.PICKUP_OBJECT, 3, 3),
        (AlfworldEventType.HEAT_OBJECT, 4, 4),
        (AlfworldEventType.PUT_OBJECT, 5, 5),
    ]
    signature = alfworld_segment_signature(
        segments[0],
        goal=trajectory.task_goal,
        next_event=segments[1].event_type,
    )
    assert signature == (
        "Task: put a hot apple on a countertop\n"
        "Event context: START -> GotoLocation -> OpenObject\n"
        "Event: GotoLocation\nLength: 2\n"
        "Actions: GO_TO(countertop) -> GO_TO(fridge)"
    )
    assert "countertop 1" not in signature
    assert "fridge 2" not in signature
    assert trajectory.task_goal in signature
    assert "before-0" not in signature
    assert SegmentInstance.from_record(segments[0].to_record()) == segments[0]


def test_alfworld_current_state_limits_applicable_events() -> None:
    searching = alfworld_applicable_events(
        (
            "go to countertop 1",
            "take apple 1 from countertop 1",
            "look",
        )
    )
    ready_to_clean = alfworld_applicable_events(
        (
            "go to sinkbasin 1",
            "clean apple 1 with sinkbasin 1",
            "put apple 1 in sinkbasin 1",
        )
    )

    assert AlfworldEventType.PICKUP_OBJECT in searching
    assert AlfworldEventType.PUT_OBJECT not in searching
    assert AlfworldEventType.CLEAN_OBJECT not in searching
    assert AlfworldEventType.PUT_OBJECT in ready_to_clean
    assert AlfworldEventType.CLEAN_OBJECT in ready_to_clean
    assert AlfworldEventType.PICKUP_OBJECT not in ready_to_clean


@pytest.mark.parametrize(
    ("goal", "event"),
    (
        ("wash a knife and put it on the table", AlfworldEventType.CLEAN_OBJECT),
        ("place a wet soap on the toilet", AlfworldEventType.CLEAN_OBJECT),
        ("put a microwaved tomato in the fridge", AlfworldEventType.HEAT_OBJECT),
        ("put a chilled pan on the stove", AlfworldEventType.COOL_OBJECT),
        ("cut an apple and place it on a plate", AlfworldEventType.SLICE_OBJECT),
        ("inspect a mug under the lamp", AlfworldEventType.TOGGLE_OBJECT),
    ),
)
def test_alfworld_goal_event_synonyms(goal: str, event: AlfworldEventType) -> None:
    assert event in alfworld_goal_events(goal)


def webshop_step(
    index: int,
    action_name: str,
    arguments: dict,
    clickable_types: dict[str, ClickableKind] | None = None,
) -> StepRecord:
    return StepRecord(
        step_index=index,
        observation=f"before-{index}",
        action_name=action_name,
        action_arguments=arguments,
        next_observation=f"after-{index}",
        reward=0,
        done=False,
        valid_action=True,
        admissible_actions=tuple((clickable_types or {}).keys()),
        clickable_types=clickable_types or {},
    )


def test_webshop_event_priority_and_page_state() -> None:
    steps = (
        webshop_step(0, "search_action", {"keywords": "blue rug"}),
        webshop_step(1, "search_action", {"keywords": "blue round rug"}),
        webshop_step(2, "click_action", {"value": "next >"}, {"next >": ClickableKind.BUTTON}),
        webshop_step(
            3, "click_action", {"value": "B012345678"}, {"B012345678": ClickableKind.PRODUCT_LINK}
        ),
        webshop_step(4, "click_action", {"value": "blue"}, {"blue": ClickableKind.OPTION}),
        webshop_step(5, "click_action", {"value": "Features"}, {"Features": ClickableKind.BUTTON}),
        webshop_step(6, "click_action", {"value": "< prev"}, {"< prev": ClickableKind.BUTTON}),
        webshop_step(
            7, "click_action", {"value": "Back to Search"}, {"Back to Search": ClickableKind.BUTTON}
        ),
        webshop_step(8, "search_action", {"keywords": "rug"}),
        webshop_step(9, "click_action", {"value": "B087654321"}),
        webshop_step(10, "click_action", {"value": "Buy Now"}, {"Buy Now": ClickableKind.BUTTON}),
    )
    assert classify_webshop_steps(steps) == (
        WebShopEventType.QUERY_FORMULATION,
        WebShopEventType.QUERY_REFINEMENT,
        WebShopEventType.RESULT_NAVIGATION,
        WebShopEventType.CANDIDATE_SELECTION,
        WebShopEventType.OPTION_SELECTION,
        WebShopEventType.ATTRIBUTE_INSPECTION,
        WebShopEventType.DETAIL_BACKTRACKING,
        WebShopEventType.SEARCH_BACKTRACKING,
        WebShopEventType.QUERY_REFINEMENT,
        WebShopEventType.CANDIDATE_SELECTION,
        WebShopEventType.PURCHASE_DECISION,
    )


def test_webshop_dom_types_override_fallback_and_other_click_keeps_page() -> None:
    classifier = WebShopEventClassifier(WebShopPageType.RESULTS, searched=True)
    other = webshop_step(
        0,
        "click_action",
        {"value": "B012345678"},
        {"B012345678": ClickableKind.BUTTON},
    )
    assert classifier.classify(other) is WebShopEventType.OTHER_CLICK
    assert classifier.page is WebShopPageType.RESULTS
    assert (
        classifier.classify(webshop_step(1, "click_action", {"value": "< prev"}))
        is WebShopEventType.RESULT_NAVIGATION
    )

    unknown = WebShopEventClassifier(WebShopPageType.UNKNOWN, searched=True)
    assert (
        unknown.classify(webshop_step(0, "click_action", {"value": "B012345678"}))
        is WebShopEventType.CANDIDATE_SELECTION
    )


def test_webshop_consecutive_events_merge_with_closed_boundaries() -> None:
    steps = (
        webshop_step(0, "search_action", {"keywords": "blue shirt"}),
        webshop_step(
            1, "click_action", {"value": "B012345678"}, {"B012345678": ClickableKind.PRODUCT_LINK}
        ),
        webshop_step(2, "click_action", {"value": "blue"}, {"blue": ClickableKind.OPTION}),
        webshop_step(3, "click_action", {"value": "large"}, {"large": ClickableKind.OPTION}),
        webshop_step(4, "click_action", {"value": "Buy Now"}, {"Buy Now": ClickableKind.BUTTON}),
    )
    trajectory = EpisodeTrajectory(
        run_id="test-run",
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TRAIN,
        method=MethodName.NO_SKILL,
        sample_id="webshop:1000",
        repeat_id=0,
        task_goal="buy a blue large shirt",
        steps=steps,
        primary_score=1,
        finish_reason=FinishReason.COMPLETED,
    )
    transitions = build_benchmark_transitions(trajectory)
    segments = segment_webshop_trajectory(
        trajectory,
        transitions,
        tuple((float(index), 1.0) for index in range(len(steps))),
    )
    assert [(segment.event_type, segment.start_step, segment.end_step) for segment in segments] == [
        (WebShopEventType.QUERY_FORMULATION, 0, 0),
        (WebShopEventType.CANDIDATE_SELECTION, 1, 1),
        (WebShopEventType.OPTION_SELECTION, 2, 3),
        (WebShopEventType.PURCHASE_DECISION, 4, 4),
    ]
    option_segment = segments[2]
    assert len(option_segment.raw_actions) == 2
    assert option_segment.observation_before == "before-2"
    assert option_segment.observation_after == "after-3"
    assert option_segment.embedding == (2.5, 1.0)
    assert SegmentInstance.from_record(option_segment.to_record()) == option_segment
    assert StepTransition.from_record(transitions[0].to_record()) == transitions[0]
    assert [transition.primitive_action for transition in transitions] == [
        PrimitiveAction.SEARCH,
        PrimitiveAction.CLICK,
        PrimitiveAction.CLICK,
        PrimitiveAction.CLICK,
        PrimitiveAction.CLICK,
    ]

    signature = webshop_segment_signature(
        option_segment,
        goal=trajectory.task_goal,
        previous_event=segments[1].event_type,
        next_event=segments[3].event_type,
    )
    assert "Goal: buy a blue large shirt" in signature
    assert "Previous event: CANDIDATE_SELECTION" in signature
    assert "SELECT_OPTION(value=blue" in signature
    assert "SELECT_OPTION(value=large" in signature
    assert "Next event: PURCHASE_DECISION" in signature


def test_change_point_dp_uses_semantic_groups_and_maximum_length() -> None:
    grouped = [(1.0, 0.0)] * 3 + [(0.0, 1.0)] * 3
    assert segment_boundaries(grouped, penalty=0.1) == ((0, 2), (3, 5))
    assert segment_boundaries([(1.0, 0.0)] * 7, penalty=1) == ((0, 5), (6, 6))

    calibration = calibrate_segmentation_penalty(
        (grouped, grouped),
        target_segment_length=3,
        max_segment_length=6,
    )
    assert calibration.median_segment_length == 3
    assert calibration.segment_count == 4

    unreachable = calibrate_segmentation_penalty(
        (((1.0, 0.0),),),
        target_segment_length=3,
        max_segment_length=6,
    )
    assert unreachable.median_segment_length == 1
    assert unreachable.segment_count == 1


def test_transition_encoder_reuses_content_hash_cache(tmp_path: Path) -> None:
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        async def embed(self, texts):
            self.calls.append(tuple(texts))
            return SimpleNamespace(
                vectors=tuple((float(len(text)), 1.0) for text in texts),
                usage=LLMUsage(0, 0, 0),
            )

    async def run() -> None:
        cache_path = tmp_path / "embeddings.sqlite"
        first_runtime = FakeRuntime()
        first_encoder = TransitionEncoder(
            first_runtime,
            cache_path=cache_path,
            model="test-model",
            dimension=2,
            batch_size=8,
        )
        first = await first_encoder.embed(("a", "bb", "a"))
        assert first_runtime.calls == [("a", "bb")]
        assert first == ((1.0, 1.0), (2.0, 1.0), (1.0, 1.0))

        second_runtime = FakeRuntime()
        second_encoder = TransitionEncoder(
            second_runtime,
            cache_path=cache_path,
            model="test-model",
            dimension=2,
            batch_size=8,
        )
        second = await second_encoder.embed(("bb", "a"))
        assert second_runtime.calls == []
        assert second == ((2.0, 1.0), (1.0, 1.0))

    asyncio.run(run())


def test_transition_encoder_keeps_successful_batches_after_peer_failure(tmp_path: Path) -> None:
    class FailingRuntime:
        def __init__(self, fail_text: str | None):
            self.fail_text = fail_text
            self.calls = []

        async def embed(self, texts):
            self.calls.append(tuple(texts))
            if self.fail_text in texts:
                raise RuntimeError("simulated provider failure")
            return SimpleNamespace(
                vectors=tuple((float(len(text)), 1.0) for text in texts),
                usage=LLMUsage(0, 0, 0),
            )

    async def run() -> None:
        cache_path = tmp_path / "embeddings.sqlite"
        first_runtime = FailingRuntime("bad")
        encoder = TransitionEncoder(
            first_runtime,
            cache_path=cache_path,
            model="test-model",
            dimension=2,
            batch_size=1,
        )
        with pytest.raises(RuntimeError, match="simulated provider failure"):
            await encoder.embed(("good", "bad"))

        second_runtime = FailingRuntime(None)
        resumed = TransitionEncoder(
            second_runtime,
            cache_path=cache_path,
            model="test-model",
            dimension=2,
            batch_size=1,
        )
        assert await resumed.embed(("good", "bad")) == ((4.0, 1.0), (3.0, 1.0))
        assert second_runtime.calls == [("bad",)]

    asyncio.run(run())
