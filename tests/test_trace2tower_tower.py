from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.llm_runtime import (
    ChatResult,
    EmbeddingResult,
    LLMUsage,
    ModelRole,
    ToolCall,
)
from trace2tower.core.manifests import Benchmark
from trace2tower.methods.trace2tower.adapters.alfworld.plan_rewrite import (
    AlfworldPlanRewriteAdapter,
)
from trace2tower.methods.trace2tower.artifacts.tower import (
    TowerSnapshot,
    TowerSourceHashes,
    TowerVersion,
    build_tower_snapshot,
)
from trace2tower.methods.trace2tower.core.config import TowerBuildMethod, Trace2TowerConfig
from trace2tower.methods.trace2tower.core.models import (
    HighPath,
    MidCluster,
    PrimitiveAction,
)
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard, LowSkill, MidSkillCard
from trace2tower.methods.trace2tower.inference.plan_rewrite import PlanRewriteTrace2TowerProvider
from trace2tower.methods.trace2tower.inference.provider import HighToMidSkillProvider


def config() -> Trace2TowerConfig:
    return Trace2TowerConfig(
        method=TowerBuildMethod.TRACE2TOWER,
        semantic_only=False,
        use_transition_edge=True,
        use_outcome_edge=True,
        use_contrastive_decomposition=True,
        failure_penalty=1.0,
        min_mid_clusters=2,
        max_mid_clusters=20,
        random_state=42,
    )


def mid_card(skill_id: str, segment_id: str) -> MidSkillCard:
    return MidSkillCard(
        skill_id,
        (segment_id,),
        f"Name {skill_id}",
        "Use when applicable.",
        ("Execute the action.",),
        ("Check the state.",),
        (PrimitiveAction.GOTO,),
    )


def complete_snapshot() -> TowerSnapshot:
    trajectory_id = "alfworld:train:no_skill:sample:0"
    segment_a = f"{trajectory_id}:segment:0-0"
    segment_b = f"{trajectory_id}:segment:1-1"
    mids = (mid_card("mid_a", segment_a), mid_card("mid_b", segment_b))
    high = HighSkillCard(
        "high_ab",
        ("mid_a", "mid_b"),
        "Combined",
        "Use for the sequence.",
        ("Execute both skills.",),
    )
    return build_tower_snapshot(
        version=TowerVersion.V0,
        benchmark=Benchmark.ALFWORLD,
        config=config(),
        training_trajectory_ids=(trajectory_id,),
        source_hashes=TowerSourceHashes(*("a" * 64,) * 5),
        low_skills=(LowSkill(PrimitiveAction.GOTO, "go to {receptacle}"),),
        mid_clusters=(
            MidCluster("mid_a", (segment_a,), (1.0, 0.0)),
            MidCluster("mid_b", (segment_b,), (0.0, 1.0)),
        ),
        high_paths=(
            HighPath(
                "high_ab",
                ("mid_a", "mid_b"),
                1.0,
                0.0,
                1.0,
                (trajectory_id,),
            ),
        ),
        mid_cards=mids,
        high_cards=(high,),
        mid_index=SkillEmbeddingIndex(
            ("mid_a", "mid_b"), ((1.0, 0.0), (0.0, 1.0)), ("b" * 64, "c" * 64)
        ),
        high_index=SkillEmbeddingIndex(("high_ab",), ((1.0, 1.0),), ("d" * 64,)),
    )


def test_complete_snapshot_round_trip_and_content_id() -> None:
    snapshot = complete_snapshot()
    assert snapshot.is_complete
    snapshot.require_complete()
    assert TowerSnapshot.from_record(snapshot.to_record()) == snapshot

    record = snapshot.to_record()
    record["mid_cards"][0]["name"] = "tampered"
    with pytest.raises(ValueError, match="snapshot ID"):
        TowerSnapshot.from_record(record)


def test_incomplete_snapshot_is_representable_but_not_formally_executable() -> None:
    snapshot = complete_snapshot()
    partial = build_tower_snapshot(
        version=snapshot.version,
        benchmark=snapshot.benchmark,
        config=snapshot.config,
        training_trajectory_ids=snapshot.training_trajectory_ids,
        source_hashes=snapshot.source_hashes,
        low_skills=snapshot.low_skills,
        mid_clusters=snapshot.mid_clusters,
        high_paths=snapshot.high_paths,
        mid_cards=snapshot.mid_cards,
        high_cards=(),
        mid_index=snapshot.mid_index,
        high_index=SkillEmbeddingIndex((), (), ()),
    )
    assert not partial.high_coverage_complete
    with pytest.raises(ValueError, match="complete Mid and High"):
        partial.require_complete()


def test_snapshot_rejects_support_outside_training_provenance() -> None:
    snapshot = complete_snapshot()
    invalid_path = replace(snapshot.high_paths[0], supporting_trajectory_ids=("other-trajectory",))
    with pytest.raises(ValueError, match="support outside training"):
        replace(snapshot, snapshot_id="", high_paths=(invalid_path,))


def test_high_to_mid_provider_can_skip_rewrite_without_dropping_high() -> None:
    class FakeRuntime:
        def __init__(self):
            self.calls = []

        async def embed(self, texts) -> EmbeddingResult:
            self.calls.append(texts)
            vectors = ((1.0, 1.0),) if texts == ["goal"] else ((1.0, 0.0),)
            return EmbeddingResult(vectors, LLMUsage(7, None, None), 1)

    runtime = FakeRuntime()
    provider = HighToMidSkillProvider(
        runtime,
        complete_snapshot(),
        reference_high_top_k=1,
        skills_per_step=2,
        max_mid_skills=2,
        mid_similarity_threshold=-1.0,
        rewrite_model_role=ModelRole.RENDERER,
        rewrite_max_output_tokens=1200,
        rewrite_plan=False,
    )
    selection = asyncio.run(
        provider.select_task(
            "goal",
            EnvironmentState("initial observation", (), {}, False, 0.0, False, True),
        )
    )
    assert selection.skill_ids == ("high_ab", "mid_a", "mid_b")
    assert selection.context_skill_ids == selection.skill_ids
    assert selection.model_output_tokens == 0
    assert "# Reference Plan" in selection.context
    assert runtime.calls == [["goal"], ("# step 1: Execute both skills.",)]


def test_plan_rewrite_provider_preserves_original_two_stage_contract() -> None:
    class FakeRuntime:
        def __init__(self):
            self.embed_calls = []
            self.chat_calls = []

        async def embed(self, texts) -> EmbeddingResult:
            self.embed_calls.append(texts)
            vectors = ((1.0, 1.0),) if len(texts) == 1 else tuple((1.0, 0.0) for _ in texts)
            return EmbeddingResult(vectors, LLMUsage(7, 3, None), 1)

        async def chat(self, role, messages, **kwargs) -> ChatResult:
            self.chat_calls.append((role, messages, kwargs))
            if len(self.chat_calls) == 1:
                payload = {
                    "name": "Rewritten goal",
                    "description": "Use the concrete task plan.",
                    "procedure": ["Find the object.", "Place the object."],
                    "constraints": ["Use the exact object."],
                }
                call = ToolCall("rewrite", "submit_task_plan", json.dumps(payload))
            else:
                call = ToolCall("select", "select_supporting_skills", '{"skill_ids":["mid_a"]}')
            return ChatResult(None, (call,), LLMUsage(11, 5, None), 1)

    runtime = FakeRuntime()
    provider = PlanRewriteTrace2TowerProvider(
        runtime,
        complete_snapshot(),
        adapter=AlfworldPlanRewriteAdapter(),
        reference_high_top_k=1,
        high_similarity_threshold=-1.0,
        skills_per_step=2,
        max_mid_skills=2,
        mid_similarity_threshold=-1.0,
        expose_reference_mid_evidence=False,
        rewrite_model_role=ModelRole.RENDERER,
        rewrite_max_output_tokens=1200,
    )

    selection = asyncio.run(
        provider.select_task(
            "goal",
            EnvironmentState("Your task is to: goal", (), {}, False, 0.0, False, True),
        )
    )

    assert selection.skill_ids == ("high_ab", "mid_a")
    assert selection.context_skill_ids == selection.skill_ids
    assert "## Strategy: Rewritten goal" in selection.context
    assert "## Skill: Name mid_a" in selection.context
    assert [call[2]["tools"][0]["function"]["name"] for call in runtime.chat_calls] == [
        "submit_task_plan",
        "select_supporting_skills",
    ]


def test_plan_rewrite_provider_can_use_threshold_mid_candidates() -> None:
    class FakeRuntime:
        def __init__(self):
            self.chat_calls = []

        async def embed(self, texts) -> EmbeddingResult:
            vectors = ((1.0, 1.0),) if len(texts) == 1 else tuple((1.0, 0.0) for _ in texts)
            return EmbeddingResult(vectors, LLMUsage(7, 3, None), 1)

        async def chat(self, role, messages, **kwargs) -> ChatResult:
            self.chat_calls.append((role, messages, kwargs))
            payload = {
                "name": "Rewritten goal",
                "description": "Use the concrete task plan.",
                "procedure": ["Find the object.", "Place the object."],
                "constraints": ["Use the exact object."],
            }
            call = ToolCall("rewrite", "submit_task_plan", json.dumps(payload))
            return ChatResult(None, (call,), LLMUsage(11, 5, None), 1)

    runtime = FakeRuntime()
    provider = PlanRewriteTrace2TowerProvider(
        runtime,
        complete_snapshot(),
        adapter=AlfworldPlanRewriteAdapter(),
        reference_high_top_k=1,
        high_similarity_threshold=-1.0,
        skills_per_step=2,
        max_mid_skills=2,
        mid_similarity_threshold=-1.0,
        expose_reference_mid_evidence=False,
        rewrite_model_role=ModelRole.RENDERER,
        rewrite_max_output_tokens=1200,
        mid_selection_strategy="threshold",
    )

    selection = asyncio.run(
        provider.select_task(
            "goal",
            EnvironmentState("Your task is to: goal", (), {}, False, 0.0, False, True),
        )
    )

    assert selection.skill_ids == ("high_ab", "mid_a", "mid_b")
    assert len(runtime.chat_calls) == 1


def test_alfworld_plan_rewrite_preserves_observation_goal_case() -> None:
    state = EnvironmentState(
        "Your task is to: Put a CLEAN apple in cabinet.\nYou are in a room.",
        (),
        {},
        False,
        0.0,
        False,
        True,
    )

    assert (
        AlfworldPlanRewriteAdapter().task_text("fallback", state)
        == "Put a CLEAN apple in cabinet."
    )
