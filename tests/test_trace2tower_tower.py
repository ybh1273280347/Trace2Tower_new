from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from trace2tower.llm_runtime import EmbeddingResult, LLMUsage
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.models import HighPath, MidCluster, PrimitiveAction
from trace2tower.methods.trace2tower.provider import Trace2TowerSkillProvider
from trace2tower.methods.trace2tower.retrieval import SkillEmbeddingIndex
from trace2tower.methods.trace2tower.skills import HighSkillCard, LowSkill, MidSkillCard
from trace2tower.methods.trace2tower.tower import (
    TowerSnapshot,
    TowerSourceHashes,
    TowerVersion,
    build_tower_snapshot,
)
from trace2tower.results import MethodName


def config() -> Trace2TowerConfig:
    return Trace2TowerConfig(
        method=MethodName.TRACE2TOWER_FULL,
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
        high_index=SkillEmbeddingIndex(
            ("high_ab",), ((1.0, 1.0),), ("d" * 64,)
        ),
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
    invalid_path = replace(
        snapshot.high_paths[0], supporting_trajectory_ids=("other-trajectory",)
    )
    with pytest.raises(ValueError, match="support outside training"):
        replace(snapshot, snapshot_id="", high_paths=(invalid_path,))


def test_provider_selects_from_complete_snapshot_and_reports_embedding_cost() -> None:
    class FakeRuntime:
        async def embed(self, texts) -> EmbeddingResult:
            assert texts == ["goal", "goal\ninitial observation"]
            return EmbeddingResult(
                ((1.0, 0.0), (1.0, 0.0)),
                LLMUsage(23, None, None),
                1,
            )

    provider = Trace2TowerSkillProvider(FakeRuntime(), complete_snapshot())
    selection = asyncio.run(provider.select("goal", "initial observation"))
    assert selection.skill_ids == ("high_ab", "mid_a", "mid_b")
    assert selection.model_input_tokens == 23
    assert "Combined" in selection.context
