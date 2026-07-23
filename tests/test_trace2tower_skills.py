from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from trace2tower.components.llm_runtime import ChatResult, CommonLLMRuntime, LLMUsage, ToolCall
from trace2tower.core.manifests import Benchmark
from trace2tower.methods.trace2tower.core.models import HighPath, MidCluster, PrimitiveAction
from trace2tower.methods.trace2tower.induction.high_paths import (
    compress_repeated_mid_ids,
    mine_high_paths,
)
from trace2tower.methods.trace2tower.induction.skills import (
    MidRenderInput,
    MidSkillCard,
    SegmentEvidence,
    build_mid_render_inputs,
)
from trace2tower.methods.trace2tower.rendering.renderer import render_high_card, render_mid_card
from scripts.experiments.build.build_trace2tower_skills import build_trajectory_render_contexts


def segment(segment_id: str, trajectory_id: str, step: int) -> dict:
    transition_id = f"{segment_id}:transition"
    return {
        "segment_id": segment_id,
        "trajectory_id": trajectory_id,
        "start_step": step,
        "end_step": step,
        "transition_ids": [transition_id],
        "embedding": [1.0, 0.0],
        "trajectory_score": 1.0,
        "event_type": None,
        "raw_actions": ["go to counter"],
        "observation_before": "before",
        "observation_after": "after",
    }


def record(trajectory_id: str, score: float, segment_ids: tuple[str, ...]) -> dict:
    segments = [
        segment(segment_id, trajectory_id, index) for index, segment_id in enumerate(segment_ids)
    ]
    transitions = [
        {
            "transition_id": item["transition_ids"][0],
            "trajectory_id": trajectory_id,
            "step_index": item["start_step"],
            "goal": "put an object away",
            "observation_before": "before",
            "raw_action": "go to counter",
            "primitive_action": "GOTO",
            "observation_after": "after",
            "trajectory_score": score,
        }
        for item in segments
    ]
    return {
        "trajectory_id": trajectory_id,
        "sample_id": trajectory_id.split("-")[0],
        "primary_score": score,
        "transitions": transitions,
        "segments": segments,
    }


def clusters() -> tuple[MidCluster, ...]:
    return (
        MidCluster("mid_a", ("p1a1", "p1a2", "p2a", "n1a"), ()),
        MidCluster("mid_b", ("p1b", "p2b"), ()),
        MidCluster("mid_c", ("p1c", "n1c"), ()),
    )


def records() -> tuple[dict, ...]:
    return (
        record("positive-1", 1.0, ("p1a1", "p1a2", "p1b", "p1c")),
        record("positive-2", 1.0, ("p2a", "p2b")),
        record("negative-1", 0.5, ("n1a", "n1c")),
    )


def test_high_path_compression_support_and_max_length_are_deterministic() -> None:
    assert compress_repeated_mid_ids(("a", "a", "b", "b", "c")) == ("a", "b", "c")
    paths = mine_high_paths(records(), clusters(), max_path_length=2)
    by_ids = {path.ordered_mid_ids: path for path in paths}
    assert set(by_ids) == {("mid_a", "mid_b"), ("mid_b", "mid_c")}
    assert by_ids[("mid_a", "mid_b")].positive_support == 1.0
    assert by_ids[("mid_a", "mid_b")].negative_support == 0.0
    assert by_ids[("mid_a", "mid_b")].supporting_trajectory_ids == (
        "positive-1",
        "positive-2",
    )
    reversed_paths = mine_high_paths(tuple(reversed(records())), clusters(), max_path_length=2)
    assert [path.to_record() for path in paths] == [path.to_record() for path in reversed_paths]


def test_high_paths_require_positive_contrastive_evidence() -> None:
    path_records = (
        record("positive-1", 1.0, ("p1a1", "p1b")),
        record("positive-2", 1.0, ("p2a", "p2b")),
        record("negative-1", 0.5, ("n1a", "n1c")),
    )
    path_clusters = (
        MidCluster("mid_a", ("p1a1", "p2a", "n1a"), ()),
        MidCluster("mid_b", ("p1b", "p2b", "n1c"), ()),
    )
    assert mine_high_paths(path_records, path_clusters, max_path_length=2) == ()


def test_high_render_context_preserves_compact_visible_state_transitions() -> None:
    source = record("positive-1", 1.0, ("p1a1", "p1b"))
    source["transitions"][0]["observation_before"] = "initial visible page"
    source["transitions"][0]["observation_after"] = "product page with Buy Now"

    context = build_trajectory_render_contexts([source], clusters())["positive-1"]

    assert context["state_transitions"] == [
        {
            "step_index": 0,
            "raw_action": "go to counter",
            "observation_before": "initial visible page",
            "observation_after": "product page with Buy Now",
        },
        {
            "step_index": 1,
            "raw_action": "go to counter",
            "observation_before": "before",
            "observation_after": "after",
        },
    ]


def test_mid_evidence_requires_a_cluster_partition() -> None:
    inputs = build_mid_render_inputs(records(), clusters())
    assert sum(item.support_count for item in inputs) == 8
    assert inputs[0].primitive_action_distribution == {"GOTO": 4}

    overlapping = clusters() + (MidCluster("mid_duplicate", ("p1a1",), ()),)
    with pytest.raises(ValueError, match="multiple Mid clusters"):
        build_mid_render_inputs(records(), overlapping)


class FakeRuntime:
    def __init__(self, name: str, payload: dict):
        self.name = name
        self.payload = payload
        self.calls = []

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.calls.append((role, messages, kwargs))
        return ChatResult(
            content=None,
            tool_calls=(ToolCall("call-1", self.name, json.dumps(self.payload)),),
            usage=LLMUsage(10, 5, None),
            latency_ms=12,
        )


def mid_input() -> MidRenderInput:
    return MidRenderInput(
        cluster_id="mid_fixed",
        member_segment_ids=("segment-fixed",),
        segment_evidence=(
            SegmentEvidence(
                segment_id="segment-fixed",
                trajectory_id="trajectory-fixed",
                goal="put an object away",
                raw_actions=("go to counter",),
                primitive_actions=(PrimitiveAction.GOTO,),
                observation_before="before",
                observation_after="after",
                trajectory_score=1.0,
                event_type=None,
            ),
        ),
        support_count=1,
        primitive_action_distribution={"GOTO": 1},
    )


def test_mid_renderer_preserves_builder_fields_and_rejects_illegal_grounding() -> None:
    payload = {
        "name": "Navigate to the target",
        "description": "Use when the target receptacle is known.",
        "procedure": ["Go to the target receptacle."],
        "constraints": ["Use an available action."],
        "grounding_actions": ["GOTO"],
    }
    runtime = FakeRuntime("render_mid_skill", payload)
    card, _ = asyncio.run(render_mid_card(runtime, Benchmark.ALFWORLD, mid_input()))
    assert card.skill_id == "mid_fixed"
    assert card.member_segment_ids == ("segment-fixed",)
    assert card.grounding_actions == (PrimitiveAction.GOTO,)
    assert runtime.calls[0][2]["tool_choice"] == "required"
    assert "ALFWorld execution semantics" in runtime.calls[0][1][0]["content"]
    assert "WebShop execution semantics" not in runtime.calls[0][1][0]["content"]
    assert runtime.calls[0][2]["prompt_cache_key"] == "trace2tower:mid:alfworld:v3"

    payload["grounding_actions"] = ["CLICK"]
    with pytest.raises(ValueError, match="outside the cluster"):
        asyncio.run(
            render_mid_card(
                FakeRuntime("render_mid_skill", payload),
                Benchmark.ALFWORLD,
                mid_input(),
            )
        )

    payload["grounding_actions"] = ["GOTO"]
    payload["skill_id"] = "model_owned_id"
    with pytest.raises(ValueError, match="unexpected fields"):
        asyncio.run(
            render_mid_card(
                FakeRuntime("render_mid_skill", payload),
                Benchmark.ALFWORLD,
                mid_input(),
            )
        )


def test_high_renderer_preserves_path_id_and_mid_order() -> None:
    child = MidSkillCard(
        "mid_fixed",
        ("segment-fixed",),
        "Navigate",
        "When a target is known.",
        ("Navigate.",),
        ("Use available actions.",),
        (PrimitiveAction.GOTO,),
    )
    path = HighPath("high_fixed", ("mid_fixed",), 1.0, 0.0, 1.0, ("trajectory-fixed",))
    payload = {
        "name": "Complete navigation",
        "description": "Use for a known destination.",
        "procedure": ["Execute the child skill in order."],
        "constraints": ["Confirm the destination before acting."],
    }
    runtime = FakeRuntime("render_high_skill", payload)
    card, _ = asyncio.run(render_high_card(runtime, Benchmark.ALFWORLD, path, {"mid_fixed": child}))
    assert card.skill_id == path.path_id
    assert card.ordered_mid_ids == path.ordered_mid_ids
    assert runtime.calls[0][2]["tool_choice"] == "required"
    assert runtime.calls[0][2]["prompt_cache_key"] == "trace2tower:high:alfworld:task-v7"
    assert "ALFWorld task-strategy semantics" in runtime.calls[0][1][0]["content"]
    assert "WebShop task-strategy semantics" not in runtime.calls[0][1][0]["content"]


def test_mid_renderer_keeps_a_stable_prefix_before_variable_evidence() -> None:
    payload = {
        "name": "Navigate",
        "description": "Use when a destination is known.",
        "procedure": ["Go to the destination."],
        "constraints": ["Use an available action."],
        "grounding_actions": ["GOTO"],
    }
    first = FakeRuntime("render_mid_skill", payload)
    second = FakeRuntime("render_mid_skill", payload)
    changed_input = MidRenderInput(
        cluster_id="mid_other",
        member_segment_ids=mid_input().member_segment_ids,
        segment_evidence=mid_input().segment_evidence,
        support_count=mid_input().support_count,
        primitive_action_distribution={"GOTO": 1, "PICK": 1},
    )
    asyncio.run(render_mid_card(first, Benchmark.ALFWORLD, mid_input()))
    asyncio.run(render_mid_card(second, Benchmark.ALFWORLD, changed_input))

    first_messages = first.calls[0][1]
    second_messages = second.calls[0][1]
    assert first_messages[0] == second_messages[0]
    first_actions = first.calls[0][2]["tools"][0]["function"]["parameters"]["properties"][
        "grounding_actions"
    ]["items"]["enum"]
    second_actions = second.calls[0][2]["tools"][0]["function"]["parameters"]["properties"][
        "grounding_actions"
    ]["items"]["enum"]
    assert first_actions == ["GOTO"]
    assert second_actions == ["GOTO", "PICK"]
    assert first.calls[0][2]["prompt_cache_key"] == second.calls[0][2]["prompt_cache_key"]
    assert first_messages[-1] != second_messages[-1]


def test_mid_renderer_uses_event_proportions_and_excludes_rare_grounding() -> None:
    dominant = tuple(
        SegmentEvidence(
            segment_id=f"goto-{index}",
            trajectory_id=f"trajectory-{index}",
            goal="move an object",
            raw_actions=("go to counter",),
            primitive_actions=(PrimitiveAction.GOTO,),
            observation_before="before",
            observation_after="after",
            trajectory_score=float(index % 2 == 0),
            event_type="GotoLocation",
        )
        for index in range(9)
    )
    rare = SegmentEvidence(
        segment_id="heat-rare",
        trajectory_id="trajectory-rare",
        goal="heat an object",
        raw_actions=("heat apple with microwave",),
        primitive_actions=(PrimitiveAction.HEAT,),
        observation_before="before",
        observation_after="after",
        trajectory_score=1.0,
        event_type="HeatObject",
    )
    render_input = MidRenderInput(
        cluster_id="mid_weighted",
        member_segment_ids=tuple(item.segment_id for item in (*dominant, rare)),
        segment_evidence=(*dominant, rare),
        support_count=10,
        primitive_action_distribution={"GOTO": 99, "HEAT": 1},
    )
    payload = {
        "name": "Navigate",
        "description": "Use when a destination is known.",
        "procedure": ["Go to the destination."],
        "constraints": ["Confirm the destination."],
        "grounding_actions": ["GOTO"],
    }
    runtime = FakeRuntime("render_mid_skill", payload)

    asyncio.run(render_mid_card(runtime, Benchmark.ALFWORLD, render_input))

    profile = json.loads(runtime.calls[0][1][-1]["content"])
    assert profile["event_type_distribution"] == {
        "GotoLocation": 9,
        "HeatObject": 1,
    }
    assert [item["raw_actions"] for item in profile["representative_examples"]].count(
        ["heat apple with microwave"]
    ) == 1
    legal_actions = runtime.calls[0][2]["tools"][0]["function"]["parameters"]["properties"][
        "grounding_actions"
    ]["items"]["enum"]
    assert legal_actions == ["GOTO"]


def test_mid_renderer_receives_complete_trajectory_context() -> None:
    payload = {
        "name": "Navigate",
        "description": "Use when a destination is known.",
        "procedure": ["Go to the destination."],
        "constraints": ["Use an available action."],
        "grounding_actions": ["GOTO"],
    }
    runtime = FakeRuntime("render_mid_skill", payload)
    context = {
        "goal": "put an object away",
        "trajectory_score": 1.0,
        "ordered_mid_ids": ["mid_fixed", "mid_other"],
        "ordered_events": ["GotoLocation", "PutObject"],
        "ordered_actions": [{"primitive_action": "GOTO", "raw_action": "go to counter"}],
    }

    asyncio.run(
        render_mid_card(
            runtime,
            Benchmark.ALFWORLD,
            mid_input(),
            trajectory_contexts={"trajectory-fixed": context},
        )
    )

    profile = json.loads(runtime.calls[0][1][-1]["content"])
    assert profile["successful_trajectory_contexts"] == [context]
    assert profile["unsuccessful_trajectory_contexts"] == []


def test_high_renderer_does_not_send_builder_membership_ids() -> None:
    child = MidSkillCard(
        "mid_fixed",
        tuple(f"segment-{index}" for index in range(100)),
        "Navigate",
        "Use when a target is known.",
        ("Navigate.",),
        ("Confirm the target.",),
        (PrimitiveAction.GOTO,),
    )
    path = HighPath(
        "high_fixed",
        ("mid_fixed",),
        1.0,
        0.0,
        1.0,
        tuple(f"trajectory-{index}" for index in range(100)),
    )
    payload = {
        "name": "Navigate",
        "description": "Use for a known destination.",
        "procedure": ["Go to the destination."],
        "constraints": ["Confirm the destination."],
    }
    runtime = FakeRuntime("render_high_skill", payload)

    asyncio.run(render_high_card(runtime, Benchmark.ALFWORLD, path, {"mid_fixed": child}))

    render_input = json.loads(runtime.calls[0][1][-1]["content"])
    assert "member_segment_ids" not in render_input["child_mid_cards"][0]
    assert "supporting_trajectory_ids" not in render_input
    assert render_input["supporting_trajectory_count"] == 100


def test_runtime_records_prompt_cache_usage() -> None:
    usage = CommonLLMRuntime._usage(
        SimpleNamespace(
            prompt_tokens=7354,
            completion_tokens=208,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=4992,
                cache_write_tokens=None,
            ),
        )
    )
    assert usage.cached_input_tokens == 4992
    assert usage.cache_write_input_tokens is None
