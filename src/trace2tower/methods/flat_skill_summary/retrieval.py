from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trace2tower.methods.flat_skill_summary.models import FlatSkillCard
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch, diverse_search


@dataclass(frozen=True, slots=True)
class FlatRetrieval:
    cards: tuple[FlatSkillCard, ...]
    candidate_matches: tuple[SkillMatch, ...]
    filtered_matches: tuple[SkillMatch, ...]
    deduplicated_matches: tuple[SkillMatch, ...]
    matches: tuple[SkillMatch, ...]
    context: str

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(card.skill_id for card in self.cards)


def retrieve_flat_skills(
    query_vector: tuple[float, ...],
    index: SkillEmbeddingIndex,
    cards: Mapping[str, FlatSkillCard],
    *,
    candidate_top_k: int = 100,
    similarity_threshold: float = 0.45,
    relative_margin: float = 0.08,
    dedup_similarity_threshold: float = 0.95,
    mmr_lambda: float = 0.75,
    max_skills: int = 8,
) -> FlatRetrieval:
    if set(index.skill_ids) != set(cards):
        raise ValueError("Flat index and card library differ")
    stages = diverse_search(
        index,
        query_vector,
        candidate_count=candidate_top_k,
        similarity_threshold=similarity_threshold,
        relative_margin=relative_margin,
        dedup_similarity_threshold=dedup_similarity_threshold,
        relevance_weight=mmr_lambda,
        max_count=max_skills,
    )
    candidate_matches = stages.candidates
    filtered_matches = stages.filtered
    deduplicated_matches = stages.deduplicated
    matches = stages.selected
    selected = tuple(cards[match.skill_id] for match in matches)
    return FlatRetrieval(
        cards=selected,
        candidate_matches=candidate_matches,
        filtered_matches=filtered_matches,
        deduplicated_matches=deduplicated_matches,
        matches=matches,
        context="\n\n".join(format_flat_card(card) for card in selected),
    )


def retrieve_flat_skills_legacy(
    query_vector: tuple[float, ...],
    index: SkillEmbeddingIndex,
    cards: Mapping[str, FlatSkillCard],
    top_k: int = 3,
) -> FlatRetrieval:
    if set(index.skill_ids) != set(cards):
        raise ValueError("Flat index and card library differ")
    matches = index.search(query_vector, top_k)
    selected = tuple(cards[match.skill_id] for match in matches)
    return FlatRetrieval(
        cards=selected,
        candidate_matches=matches,
        filtered_matches=matches,
        deduplicated_matches=matches,
        matches=matches,
        context="\n\n".join(format_flat_card(card) for card in selected),
    )


def format_flat_card(card: FlatSkillCard) -> str:
    lines = [f"## Skill: {card.name}", f"Use when: {card.description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(card.procedure, 1))
    lines.append("Constraints:")
    lines.extend(f"- {constraint}" for constraint in card.constraints)
    return "\n".join(lines)
