from __future__ import annotations

import asyncio
import json

import pytest

from trace2tower.llm_runtime import (
    ChatResult,
    EmbeddingResult,
    LLMUsage,
    ToolCall,
)
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.methods.flat_skill_summary.models import (
    FlatSkillCard,
    FlatSkillLibrary,
    build_flat_library,
)
from trace2tower.methods.flat_skill_summary.prompt import FLAT_SKILL_PROMPT
from trace2tower.methods.flat_skill_summary.provider import FlatSkillProvider
from trace2tower.methods.flat_skill_summary.renderer import render_flat_skill
from trace2tower.methods.flat_skill_summary.retrieval import retrieve_flat_skills
from trace2tower.results import FinishReason, MethodName
from trace2tower.semantic_index import SkillEmbeddingIndex
from trace2tower.trajectory import EpisodeTrajectory, StepRecord


def trajectory() -> EpisodeTrajectory:
    return EpisodeTrajectory(
        run_id="pilot",
        benchmark=Benchmark.WEBSHOP,
        split=ExperimentSplit.TRAIN,
        method=MethodName.NO_SKILL,
        sample_id="webshop:1000",
        repeat_id=0,
        task_goal="buy a red bottle",
        steps=(
            StepRecord(
                0,
                "search page",
                "search_action",
                {"keywords": "red bottle"},
                "results",
                0,
                False,
                True,
                (),
                {},
            ),
        ),
        primary_score=1.0,
        finish_reason=FinishReason.COMPLETED,
    )


class FakeChatRuntime:
    def __init__(self, payload: dict):
        self.payload = payload
        self.kwargs = None

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.kwargs = kwargs
        return ChatResult(
            None,
            (ToolCall("call-1", "render_flat_skill", json.dumps(self.payload)),),
            LLMUsage(20, 5, None),
            1,
        )


def card(index: int) -> FlatSkillCard:
    return FlatSkillCard(
        f"flat_{index}",
        f"trajectory_{index}",
        f"Skill {index}",
        "Use when applicable.",
        (f"Execute step {index}.",),
        ("Verify the state.",),
    )


def library() -> FlatSkillLibrary:
    cards = tuple(card(index) for index in range(3))
    return build_flat_library(
        Benchmark.WEBSHOP,
        "a" * 64,
        cards,
        SkillEmbeddingIndex(
            tuple(item.skill_id for item in cards),
            ((1.0, 0.0), (0.8, 0.2), (0.0, 1.0)),
            tuple(chr(98 + index) * 64 for index in range(3)),
        ),
    )


def test_renderer_preserves_builder_provenance_and_fixed_schema() -> None:
    payload = {
        "name": "Search for a matching product",
        "description": "Use for attribute-constrained shopping.",
        "procedure": ["Search with the requested attributes."],
        "constraints": ["Verify the product before purchase."],
    }
    runtime = FakeChatRuntime(payload)
    rendered, _ = asyncio.run(render_flat_skill(runtime, trajectory()))
    assert rendered.skill_id.startswith("flat_")
    assert rendered.source_trajectory_id == trajectory().trajectory_id
    assert runtime.kwargs["tool_choice"] == "required"
    assert runtime.kwargs["prompt_cache_key"] == "flat-skill:webshop:compact-v1"
    assert len(FLAT_SKILL_PROMPT) > 4096

    payload["skill_id"] = "model-owned"
    with pytest.raises(ValueError, match="outside the fixed schema"):
        asyncio.run(render_flat_skill(FakeChatRuntime(payload), trajectory()))


def test_library_round_trip_and_content_id_reject_tampering() -> None:
    current = library()
    assert FlatSkillLibrary.from_record(current.to_record()) == current
    record = current.to_record()
    record["cards"][0]["name"] = "tampered"
    with pytest.raises(ValueError, match="library ID"):
        FlatSkillLibrary.from_record(record)


def test_flat_retrieval_returns_fixed_top_three_with_stable_order() -> None:
    current = library()
    cards = {item.skill_id: item for item in current.cards}
    result = retrieve_flat_skills((1.0, 0.0), current.index, cards)
    assert result.skill_ids == ("flat_0", "flat_1", "flat_2")
    assert result.context.count("## Skill:") == 3


def test_flat_provider_reports_query_embedding_cost() -> None:
    class FakeEmbeddingRuntime:
        async def embed(self, texts) -> EmbeddingResult:
            assert texts == ["goal\ninitial"]
            return EmbeddingResult(((1.0, 0.0),), LLMUsage(11, None, None), 1)

    selection = asyncio.run(
        FlatSkillProvider(FakeEmbeddingRuntime(), library()).select("goal", "initial")
    )
    assert selection.skill_ids == ("flat_0", "flat_1", "flat_2")
    assert selection.model_input_tokens == 11
