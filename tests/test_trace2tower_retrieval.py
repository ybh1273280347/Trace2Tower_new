from __future__ import annotations

import pytest

from trace2tower.methods.trace2tower.models import PrimitiveAction
from trace2tower.methods.trace2tower.retrieval import (
    SkillEmbeddingIndex,
    retrieve_tower,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard


def mid_card(skill_id: str) -> MidSkillCard:
    return MidSkillCard(
        skill_id=skill_id,
        member_segment_ids=(f"segment-{skill_id}",),
        name=f"Name {skill_id}",
        description=f"Description {skill_id}",
        procedure=(f"Execute {skill_id}.",),
        constraints=(f"Check {skill_id}.",),
        grounding_actions=(PrimitiveAction.GOTO,),
    )


def test_retrieval_expands_high_then_deduplicates_direct_mid() -> None:
    mids = {skill_id: mid_card(skill_id) for skill_id in ("mid_a", "mid_b", "mid_c")}
    high = HighSkillCard(
        "high_a",
        ("mid_b", "mid_c"),
        "Combined strategy",
        "Use for the combined task.",
        ("Execute children in order.",),
    )
    result = retrieve_tower(
        (1.0, 0.0),
        (1.0, 0.0),
        SkillEmbeddingIndex(("high_a",), ((1.0, 0.0),)),
        SkillEmbeddingIndex(
            ("mid_a", "mid_b", "mid_c"),
            ((1.0, 0.0), (0.9, 0.1), (0.0, 1.0)),
        ),
        {"high_a": high},
        mids,
    )
    assert result.skill_ids == ("high_a", "mid_b", "mid_c", "mid_a")
    assert result.high_match.skill_id == "high_a"
    assert tuple(match.skill_id for match in result.direct_mid_matches) == (
        "mid_a",
        "mid_b",
    )
    assert tuple(card.skill_id for card in result.mid_cards) == (
        "mid_b",
        "mid_c",
        "mid_a",
    )
    assert result.context.index("Name mid_b") < result.context.index("Name mid_c")
    assert "high_a" not in result.context


def test_retrieval_without_high_uses_direct_mid_only() -> None:
    mids = {skill_id: mid_card(skill_id) for skill_id in ("mid_a", "mid_b")}
    result = retrieve_tower(
        (1.0, 0.0),
        (1.0, 0.0),
        SkillEmbeddingIndex((), ()),
        SkillEmbeddingIndex(("mid_b", "mid_a"), ((1.0, 0.0), (1.0, 0.0))),
        {},
        mids,
    )
    assert result.high_card is None
    assert result.high_match is None
    assert result.skill_ids == ("mid_a", "mid_b")


def test_embedding_index_rejects_dimension_mismatch() -> None:
    index = SkillEmbeddingIndex(("mid_a",), ((1.0, 0.0),))
    with pytest.raises(ValueError, match="dimension"):
        index.top_k((1.0,), 1)


def test_embedding_index_requires_hash_alignment_when_hashes_are_present() -> None:
    with pytest.raises(ValueError, match="text hashes"):
        SkillEmbeddingIndex(("mid_a", "mid_b"), ((1.0,), (2.0,)), ("hash-a",))
