from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from trace2tower.agent import AgentEvaluator, SkillSelection
from trace2tower.benchmarks.models import EnvironmentState, EpisodeStart
from trace2tower.llm_runtime import ChatResult, LLMUsage, ToolCall
from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.results import MethodName
from trace2tower.trajectory import TrajectoryWriter


class FakeRuntime:
    def __init__(self):
        self.messages = None

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.messages = messages
        return ChatResult(
            content=None,
            tool_calls=(
                ToolCall("call-1", "click_action", json.dumps({"value": "Buy Now"})),
            ),
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

    async def select(task_goal: str, initial_observation: str) -> SkillSelection:
        assert task_goal == "buy the matching item"
        assert initial_observation == "initial page"
        return SkillSelection(("high_a", "mid_a"), "retrieved context", 7, 0)

    result = asyncio.run(
        evaluator.run_episode(
            entry=entry,
            environment=environment,
            run_id="skill-smoke",
            method=MethodName.TRACE2TOWER_STATIC,
            skill_context=None,
            shard_id=0,
            max_steps=1,
            skill_provider=select,
        )
    )
    assert result.skill_ids == ("high_a", "mid_a")
    assert result.skill_context_chars == len("retrieved context")
    assert result.input_tokens == 17
    assert result.output_tokens == 2
    assert "retrieved context" in runtime.messages[1]["content"]
    assert environment.closed


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

    async def fail_selection(task_goal: str, initial_observation: str) -> SkillSelection:
        raise RuntimeError("selection failed")

    with pytest.raises(RuntimeError, match="selection failed"):
        asyncio.run(
            evaluator.run_episode(
                entry=entry,
                environment=environment,
                run_id="skill-smoke",
                method=MethodName.TRACE2TOWER_STATIC,
                skill_context=None,
                shard_id=0,
                max_steps=1,
                skill_provider=fail_selection,
            )
        )
    assert environment.closed
