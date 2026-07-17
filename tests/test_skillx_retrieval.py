from __future__ import annotations

import asyncio
from unittest.mock import ANY

import pytest

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import ChatResult, EmbeddingResult, LLMUsage, ModelRole
from trace2tower.manifests import Benchmark
from trace2tower.methods.skillx.models import (
    SkillXCard,
    SkillXExecutionLibrary,
    SkillXPlan,
    build_execution_library,
)
from trace2tower.methods.skillx.provider import SkillXProvider
from trace2tower.methods.skillx.native_inference import (
    SKILLX_COMMIT,
    NativeSkillCandidate,
    NativeSkillXInference,
)
from trace2tower.methods.skillx.retrieval import plan_steps
from trace2tower.semantic_index import SkillEmbeddingIndex


def library() -> SkillXExecutionLibrary:
    plan = SkillXPlan(
        "skillx_plan_one",
        "a" * 64,
        "buy a rug",
        "# step 1: search for a rug\n# step 2: inspect and buy the rug",
    )
    skills = (
        SkillXCard(
            "skillx_search",
            "b" * 64,
            "search",
            "Find candidates.",
            "Call search_action with useful keywords.",
            ("search_action",),
            "functional",
        ),
        SkillXCard(
            "skillx_buy",
            "c" * 64,
            "inspect and buy",
            "Verify and purchase.",
            "Use click_action to inspect options and buy.",
            ("click_action",),
            "functional",
        ),
    )
    return build_execution_library(
        Benchmark.WEBSHOP,
        "d" * 64,
        SKILLX_COMMIT,
        (plan,),
        skills,
        SkillEmbeddingIndex((plan.plan_id,), ((1.0, 0.0),), ("e" * 64,)),
        SkillEmbeddingIndex(
            tuple(skill.skill_id for skill in skills),
            ((1.0, 0.0), (0.0, 1.0)),
            ("f" * 64, "0" * 64),
        ),
    )


class FakeRuntime:
    def __init__(self):
        self.calls = []

    async def embed(self, texts) -> EmbeddingResult:
        self.calls.append(texts)
        if len(self.calls) == 1:
            return EmbeddingResult(((1.0, 0.0),), LLMUsage(3, None, None), 1)
        return EmbeddingResult(
            ((1.0, 0.0), (0.0, 1.0)),
            LLMUsage(7, None, None),
            1,
        )

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.calls.append((role, messages, kwargs))
        return ChatResult(
            "<plan># step 1: search for a rug\n# step 2: inspect and buy the rug</plan>",
            (),
            LLMUsage(5, 2, None),
            1,
        )


def test_execution_library_round_trip_and_rejects_tampering() -> None:
    current = library()
    assert SkillXExecutionLibrary.from_record(current.to_record()) == current
    record = current.to_record()
    record["skills"][0]["document"] = "tampered"
    with pytest.raises(ValueError, match="library ID"):
        SkillXExecutionLibrary.from_record(record)


def test_provider_reproduces_plan_then_per_step_skill_retrieval() -> None:
    runtime = FakeRuntime()
    provider = SkillXProvider(
        runtime,
        library(),
        allowed_tools={"search_action", "click_action"},
        similarity_threshold=0.45,
        plan_top_k=3,
        skills_per_step=4,
        max_skills=10,
        llm_max_output_tokens=10240,
        rewrite_plan=True,
    )
    selection = asyncio.run(
        provider.select(
            "buy a rug",
            EnvironmentState("search page", (), {}, False, 0.0, False, True),
        )
    )
    assert selection.skill_ids == (
        "skillx_plan_one",
        "skillx_search",
        "skillx_buy",
    )
    assert selection.model_input_tokens == 15
    assert selection.model_output_tokens == 2
    assert runtime.calls == [
        ["buy a rug"],
        (
            ModelRole.RENDERER,
            ANY,
            ANY,
        ),
        ("# step 1: search for a rug", "# step 2: inspect and buy the rug"),
    ]
    assert "# Reference Plan" in selection.context
    assert "# Skill 1: search" in selection.context
    assert "# Skill 2: inspect and buy" in selection.context


def test_provider_rejects_unavailable_skill_tools() -> None:
    with pytest.raises(ValueError, match="unavailable tools"):
        SkillXProvider(
            FakeRuntime(),
            library(),
            allowed_tools={"search_action"},
            similarity_threshold=0.45,
            plan_top_k=3,
            skills_per_step=4,
            max_skills=10,
            llm_max_output_tokens=10240,
            rewrite_plan=True,
        )


def test_provider_can_inject_original_plan_without_rewrite() -> None:
    runtime = FakeRuntime()
    provider = SkillXProvider(
        runtime,
        library(),
        allowed_tools={"search_action", "click_action"},
        similarity_threshold=0.45,
        plan_top_k=3,
        skills_per_step=4,
        max_skills=10,
        llm_max_output_tokens=10240,
        rewrite_plan=False,
    )
    selection = asyncio.run(
        provider.select(
            "buy a rug",
            EnvironmentState("search page", (), {}, False, 0.0, False, True),
        )
    )
    assert selection.skill_ids == (
        "skillx_plan_one",
        "skillx_search",
        "skillx_buy",
    )
    assert selection.model_output_tokens == 0
    assert runtime.calls == [
        ["buy a rug"],
        ("# step 1: search for a rug", "# step 2: inspect and buy the rug"),
    ]


def test_plan_step_parser_matches_upstream_rules() -> None:
    assert plan_steps("short\na sufficiently long instruction") == (
        "a sufficiently long instruction",
    )


class SelectorRuntime:
    def __init__(self):
        self.calls = []

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.calls.append((role, messages, kwargs))
        return ChatResult(
            '```python\n["skill_8", "skill_1"]\n```',
            (),
            LLMUsage(11, 3, None),
            1,
        )


def test_native_selector_runs_only_when_candidates_exceed_limit() -> None:
    runtime = SelectorRuntime()
    inference = NativeSkillXInference(runtime, max_output_tokens=10240)
    candidates = tuple(
        NativeSkillCandidate(str(index), f"skill_{index}", "description", "content")
        for index in range(9)
    )
    direct = asyncio.run(
        inference.select_skills(
            task="task",
            plan="plan",
            skills=candidates[:8],
            max_skills=8,
        )
    )
    selected = asyncio.run(
        inference.select_skills(
            task="task",
            plan="plan",
            skills=candidates,
            max_skills=8,
        )
    )
    assert direct.skills == candidates[:8]
    assert tuple(skill.name for skill in selected.skills) == ("skill_1", "skill_8")
    assert selected.input_tokens == 11
    assert selected.output_tokens == 3
    assert len(runtime.calls) == 1
    assert runtime.calls[0][0] is ModelRole.RENDERER
