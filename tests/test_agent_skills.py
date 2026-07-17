from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from trace2tower.benchmarks.models import EnvironmentState, EpisodeStart
from trace2tower.components.agent import AgentEvaluator, SkillSelection
from trace2tower.components.llm_runtime import ChatResult, LLMUsage, ModelRole, ToolCall
from trace2tower.core.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.core.results import MethodName
from trace2tower.core.trajectory import TrajectoryWriter


class FakeRuntime:
    def __init__(self):
        self.messages = None
        self.role = None

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.role = role
        self.messages = messages
        return ChatResult(
            content=None,
            tool_calls=(ToolCall("call-1", "click_action", json.dumps({"value": "Buy Now"})),),
            usage=LLMUsage(10, 2, None),
            latency_ms=1,
        )


class FakeEnvironment:
    tool_schemas = ()

    def __init__(self):
        self.closed = False

    async def reset(self, entry: ManifestEntry) -> EpisodeStart:
        return EpisodeStart(
            "buy the matching item",
            EnvironmentState("initial page", ("Buy Now",), {}, False, 0, False, True),
        )

    async def execute(self, tool_name: str, arguments: dict) -> EnvironmentState:
        return EnvironmentState("done", (), {}, False, 0.75, True, True)

    async def close(self) -> None:
        self.closed = True


def test_legacy_context_without_structured_ids_remains_valid() -> None:
    selection = SkillSelection((), "legacy flat context")
    assert selection.context == "legacy flat context"


def test_agent_selects_skills_after_reset_and_records_selection_cost(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    environment = FakeEnvironment()
    evaluator = AgentEvaluator(
        runtime,
        TrajectoryWriter(tmp_path / "episodes"),
        temperature=0,
        max_output_tokens=128,
    )
    entry = ManifestEntry(
        Benchmark.WEBSHOP,
        ExperimentSplit.TEST,
        "webshop:0",
        0,
        "goals",
        0,
    )

    async def select(task_goal: str, state: EnvironmentState) -> SkillSelection:
        assert task_goal == "buy the matching item"
        assert state.observation == "initial page"
        assert state.admissible_actions == ("Buy Now",)
        return SkillSelection(
            ("high_a", "mid_a"),
            "retrieved context",
            7,
            0,
            ("high_a",),
        )

    result = asyncio.run(
        evaluator.run_episode(
            entry=entry,
            environment=environment,
            run_id="skill-smoke",
            method=MethodName.TRACE2TOWER,
            skill_context=None,
            shard_id=0,
            max_steps=1,
            skill_provider=select,
        )
    )
    assert result.skill_ids == ("high_a", "mid_a")
    assert result.context_skill_ids == ("high_a",)
    assert result.skill_context_chars == len("retrieved context")
    assert result.skill_context_sha256 == hashlib.sha256(b"retrieved context").hexdigest()
    assert result.input_tokens == 17
    assert result.output_tokens == 2
    assert result.chat_input_tokens == 10
    assert result.chat_output_tokens == 2
    assert "retrieved context" in runtime.messages[1]["content"]
    assert environment.closed


def test_agent_chat_cost_does_not_include_retrieval_embedding_tokens(tmp_path: Path) -> None:
    evaluator = AgentEvaluator(
        FakeRuntime(),
        TrajectoryWriter(tmp_path / "episodes"),
        temperature=0,
        max_output_tokens=128,
    )
    entry = ManifestEntry(
        Benchmark.WEBSHOP,
        ExperimentSplit.TRAIN,
        "webshop:0",
        0,
        "goals",
        0,
    )

    async def select(task_goal: str, state: EnvironmentState) -> SkillSelection:
        return SkillSelection(("mid_a",), "retrieved context", 7, 0)

    result = asyncio.run(
        evaluator.run_episode(
            entry=entry,
            environment=FakeEnvironment(),
            run_id="chat-cost-smoke",
            method=MethodName.TRACE2TOWER,
            skill_context=None,
            shard_id=0,
            max_steps=1,
            skill_provider=select,
        )
    )
    assert result.input_tokens == 17
    assert result.chat_input_tokens + result.chat_output_tokens == 12


def test_agent_injects_task_skill_once_and_refreshes_only_state_skills(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    evaluator = AgentEvaluator(
        runtime,
        TrajectoryWriter(tmp_path / "episodes"),
        temperature=0,
        max_output_tokens=128,
    )
    entry = ManifestEntry(
        Benchmark.WEBSHOP,
        ExperimentSplit.TEST,
        "webshop:0",
        0,
        "goals",
        0,
    )
    task_calls = 0
    state_calls = 0

    async def select_task(task_goal: str, state: EnvironmentState) -> SkillSelection:
        nonlocal task_calls
        task_calls += 1
        return SkillSelection(("high_a",), "end-to-end task strategy", 3, 0)

    async def select_state(task_goal: str, state: EnvironmentState) -> SkillSelection:
        nonlocal state_calls
        state_calls += 1
        return SkillSelection(
            ("mid_a", "low:CLICK"),
            "current stage and action template",
            5,
            0,
        )

    result = asyncio.run(
        evaluator.run_episode(
            entry=entry,
            environment=FakeEnvironment(),
            run_id="split-lifecycle-smoke",
            method=MethodName.TRACE2TOWER,
            skill_context=None,
            shard_id=0,
            max_steps=1,
            skill_provider=select_task,
            state_skill_provider=select_state,
        )
    )

    assert task_calls == 1
    assert state_calls == 1
    assert result.skill_ids == ("high_a", "mid_a", "low:CLICK")
    assert result.skill_context_chars == len("end-to-end task strategy") + len(
        "current stage and action template"
    )
    assert "end-to-end task strategy" in runtime.messages[1]["content"]
    assert "current stage and action template" in runtime.messages[2]["content"]


def test_agent_can_use_explicit_renderer_endpoint_role(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    evaluator = AgentEvaluator(
        runtime,
        TrajectoryWriter(tmp_path / "episodes"),
        temperature=0,
        max_output_tokens=128,
        endpoint_role=ModelRole.RENDERER,
    )
    entry = ManifestEntry(
        Benchmark.WEBSHOP,
        ExperimentSplit.TEST,
        "webshop:0",
        0,
        "goals",
        0,
    )
    asyncio.run(
        evaluator.run_episode(
            entry=entry,
            environment=FakeEnvironment(),
            run_id="renderer-agent-smoke",
            method=MethodName.NO_SKILL,
            skill_context=None,
            shard_id=0,
            max_steps=1,
        )
    )
    assert runtime.role is ModelRole.RENDERER


def test_agent_closes_environment_when_skill_selection_fails(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    evaluator = AgentEvaluator(
        FakeRuntime(),
        TrajectoryWriter(tmp_path / "episodes"),
        temperature=0,
        max_output_tokens=128,
    )
    entry = ManifestEntry(
        Benchmark.WEBSHOP,
        ExperimentSplit.TEST,
        "webshop:0",
        0,
        "goals",
        0,
    )

    async def fail_selection(task_goal: str, state: EnvironmentState) -> SkillSelection:
        raise RuntimeError("selection failed")

    with pytest.raises(RuntimeError, match="selection failed"):
        asyncio.run(
            evaluator.run_episode(
                entry=entry,
                environment=environment,
                run_id="skill-smoke",
                method=MethodName.TRACE2TOWER,
                skill_context=None,
                shard_id=0,
                max_steps=1,
                skill_provider=fail_selection,
            )
        )
    assert environment.closed
