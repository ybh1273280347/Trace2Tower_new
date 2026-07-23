from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from trace2tower.benchmarks.alfworld import AlfworldEnvironment
from trace2tower.benchmarks.webshop import WebShopEnvironment
from trace2tower.components.llm_runtime import ChatResult, EmbeddingResult, LLMUsage
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory, StepRecord
from trace2tower.methods.skillx.embedding_adapter import SkillXEmbeddingAdapter
from trace2tower.methods.skillx.llm_adapter import SkillXLLMAdapter
from trace2tower.methods.skillx.trajectory_adapter import (
    adapt_tool_schemas,
    adapt_trajectory,
)


def episode(benchmark: Benchmark, action_name: str, arguments: dict) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        "pilot",
        benchmark,
        ExperimentSplit.TRAIN,
        MethodName.NO_SKILL,
        f"{benchmark}:sample",
        0,
        "complete the task",
        (
            StepRecord(
                0,
                "before",
                action_name,
                arguments,
                "after",
                1,
                True,
                True,
                (),
                {},
            ),
        ),
        1.0,
        FinishReason.COMPLETED,
    )


def test_trajectory_adapter_preserves_real_benchmark_tool_calls() -> None:
    alfworld = adapt_trajectory(
        episode(Benchmark.ALFWORLD, "take_action", {"action": "go to fridge 1"})
    )
    webshop = adapt_trajectory(
        episode(Benchmark.WEBSHOP, "search_action", {"keywords": "red bottle"})
    )
    alfworld_call = alfworld["task_history"][2]["tool_calls"][0]
    webshop_call = webshop["task_history"][2]["tool_calls"][0]
    assert alfworld_call["name"] == "take_action"
    assert alfworld_call["arguments"] == {"action": "go to fridge 1"}
    assert webshop_call["name"] == "search_action"
    assert webshop_call["arguments"] == {"keywords": "red bottle"}
    assert alfworld["reward"] == 1.0


def test_tool_schema_adapter_uses_upstream_name_mapping() -> None:
    alfworld = adapt_tool_schemas(AlfworldEnvironment.tool_schemas)
    webshop = adapt_tool_schemas(WebShopEnvironment.tool_schemas)
    assert set(alfworld) == {"take_action"}
    assert set(webshop) == {"search_action", "click_action"}
    assert webshop["click_action"]["parameters"]["required"] == ["value"]


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.responses = ("invalid", '<skill>{"name":"valid"}</skill>')

    async def chat(self, role, messages, **kwargs) -> ChatResult:
        self.calls.append((role, messages, kwargs))
        content = self.responses[len(self.calls) - 1]
        return ChatResult(content, (), LLMUsage(10, 2, None), 1)


def test_llm_adapter_preserves_messages_and_defers_validation_to_upstream_parser() -> None:
    runtime = FakeRuntime()
    adapter = SkillXLLMAdapter(
        runtime,
        max_output_tokens=100,
        temperature=0,
        max_validation_attempts=2,
        retry_delay_seconds=0,
    )

    def extract(text: str):
        return json.loads(text[7:-8]) if text.startswith("<skill>") else None

    result = asyncio.run(
        adapter.ainvoke(
            [("system", "official prompt"), ("human", "unaltered input")],
            regex_extractor=extract,
        )
    )
    assert result == '<skill>{"name":"valid"}</skill>'
    assert len(runtime.calls) == 2
    assert runtime.calls[0][1] == [
        {"role": "system", "content": "official prompt"},
        {"role": "user", "content": "unaltered input"},
    ]
    assert adapter.usage.calls == 2
    assert adapter.usage.input_tokens == 20


def test_llm_adapter_rejects_unsupported_message_types() -> None:
    adapter = SkillXLLMAdapter(
        FakeRuntime(),
        max_output_tokens=100,
        temperature=0,
        max_validation_attempts=1,
        retry_delay_seconds=0,
    )
    with pytest.raises(ValueError, match="unsupported SkillX message"):
        asyncio.run(adapter.ainvoke([("tool", "not accepted")]))


def test_llm_adapter_rejects_negative_retry_delay() -> None:
    with pytest.raises(ValueError, match="retry delay"):
        SkillXLLMAdapter(
            FakeRuntime(),
            max_output_tokens=100,
            temperature=0,
            max_validation_attempts=1,
            retry_delay_seconds=-1,
        )


class FakeEmbeddingRuntime:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    async def embed(self, texts):
        self.calls.append(texts)
        return EmbeddingResult(self.vectors, LLMUsage(7, None, None), 1)


def test_embedding_adapter_normalizes_and_tracks_usage() -> None:
    runtime = FakeEmbeddingRuntime(((3.0, 4.0), (0.0, 0.0)))
    adapter = SkillXEmbeddingAdapter(runtime)
    vectors = asyncio.run(adapter.embed_batch(["one", "two"]))
    assert vectors[0].tolist() == pytest.approx([0.6, 0.8])
    assert vectors[1].tolist() == [0.0, 0.0]
    assert runtime.calls == [["one", "two"]]
    assert adapter.input_tokens == 7


def test_embedding_adapter_skips_empty_batches_and_checks_cardinality() -> None:
    runtime = FakeEmbeddingRuntime(((1.0, 0.0),))
    adapter = SkillXEmbeddingAdapter(runtime)
    assert asyncio.run(adapter.embed_batch([])).shape == (0, 0)
    assert runtime.calls == []
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(adapter.embed_batch(["one", "two"]))


def test_embedding_adapter_drives_pinned_upstream_dbscan() -> None:
    skillx_parent = Path("third_party").resolve()
    if not (skillx_parent / "SkillX" / "clustering" / "dbscan.py").exists():
        pytest.skip("pinned SkillX checkout is not available")
    if str(skillx_parent) not in sys.path:
        sys.path.insert(0, str(skillx_parent))
    from SkillX.clustering.dbscan import DBSCANClusterer

    runtime = FakeEmbeddingRuntime(((1.0, 0.0), (0.999, 0.001), (0.0, 1.0)))
    adapter = SkillXEmbeddingAdapter(runtime)
    clusterer = DBSCANClusterer(
        eps=0.1,
        min_samples=1,
        metric="cosine",
        embedding_service=adapter,
    )
    clusters = asyncio.run(
        clusterer.cluster_async(
            [
                {"embedding_text": "near one"},
                {"embedding_text": "near two"},
                {"embedding_text": "separate"},
            ]
        )
    )
    assert sorted(sorted(indices) for indices in clusters.values()) == [[0, 1], [2]]
    assert runtime.calls == [["near one", "near two", "separate"]]
    assert adapter.input_tokens == 7
