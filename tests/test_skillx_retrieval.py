from __future__ import annotations

import asyncio

import pytest

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import EmbeddingResult, LLMUsage
from trace2tower.manifests import Benchmark
from trace2tower.methods.skillx.models import (
    SkillXCard,
    SkillXExecutionLibrary,
    SkillXPlan,
    build_execution_library,
)
from trace2tower.methods.skillx.provider import SkillXProvider
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
        "commit",
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
    assert selection.model_input_tokens == 10
    assert runtime.calls == [
        ["buy a rug"],
        ["# step 1: search for a rug", "# step 2: inspect and buy the rug"],
    ]
    assert "## Reference Plan" in selection.context
    assert selection.context.count("## Skill:") == 2


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
        )


def test_plan_step_parser_matches_upstream_rules() -> None:
    assert plan_steps("short\na sufficiently long instruction") == (
        "a sufficiently long instruction",
    )
