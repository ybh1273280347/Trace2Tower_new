from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch, diverse_search


@dataclass(frozen=True, slots=True)
class TowerRetrieval:
    high_card: HighSkillCard | None
    mid_cards: tuple[MidSkillCard, ...]
    high_candidate: SkillMatch | None
    high_match: SkillMatch | None
    direct_mid_candidates: tuple[SkillMatch, ...]
    direct_mid_filtered: tuple[SkillMatch, ...]
    direct_mid_deduplicated: tuple[SkillMatch, ...]
    direct_mid_matches: tuple[SkillMatch, ...]
    skill_ids: tuple[str, ...]
    context: str


def retrieve_tower(
    high_query_vector: Sequence[float],
    mid_query_vector: Sequence[float],
    high_index: SkillEmbeddingIndex,
    mid_index: SkillEmbeddingIndex,
    high_cards: Mapping[str, HighSkillCard],
    mid_cards: Mapping[str, MidSkillCard],
    *,
    high_top_k: int = 1,
    direct_mid_top_k: int = 2,
    high_similarity_threshold: float = -1.0,
    include_high_child_context: bool = True,
    direct_mid_candidate_top_k: int | None = None,
    direct_mid_similarity_threshold: float = 0.45,
    direct_mid_relative_margin: float = 0.08,
    direct_mid_dedup_similarity_threshold: float = 0.95,
    direct_mid_mmr_lambda: float = 0.75,
) -> TowerRetrieval:
    if not -1 <= high_similarity_threshold <= 1:
        raise ValueError("High similarity threshold must be in [-1, 1]")
    if not isinstance(include_high_child_context, bool):
        raise ValueError("High child context switch must be boolean")
    missing_high_cards = set(high_index.skill_ids) - set(high_cards)
    missing_mid_cards = set(mid_index.skill_ids) - set(mid_cards)
    if missing_high_cards or missing_mid_cards:
        raise ValueError("skill embedding index references missing cards")
    high_matches = high_index.search(high_query_vector, high_top_k)
    high_candidate = high_matches[0] if high_matches else None
    high_match = (
        high_candidate
        if high_candidate
        and high_candidate.cosine_similarity >= high_similarity_threshold
        else None
    )
    high_card = high_cards[high_match.skill_id] if high_match else None
    if high_card and not set(high_card.ordered_mid_ids) <= set(mid_cards):
        raise ValueError("retrieved High card references missing child Mid cards")
    if direct_mid_candidate_top_k is None:
        direct_mid_matches = mid_index.search(mid_query_vector, direct_mid_top_k)
        direct_mid_candidates = direct_mid_matches
        direct_mid_filtered = direct_mid_matches
        direct_mid_deduplicated = direct_mid_matches
    else:
        stages = diverse_search(
            mid_index,
            mid_query_vector,
            candidate_count=direct_mid_candidate_top_k,
            similarity_threshold=direct_mid_similarity_threshold,
            relative_margin=direct_mid_relative_margin,
            dedup_similarity_threshold=direct_mid_dedup_similarity_threshold,
            relevance_weight=direct_mid_mmr_lambda,
            max_count=direct_mid_top_k,
        )
        direct_mid_candidates = stages.candidates
        direct_mid_filtered = stages.filtered
        direct_mid_deduplicated = stages.deduplicated
        direct_mid_matches = stages.selected
    direct_mid_ids = tuple(match.skill_id for match in direct_mid_matches)
    ordered_mid_ids = (*high_card.ordered_mid_ids, *direct_mid_ids) if high_card else direct_mid_ids
    unique_mid_ids = tuple(dict.fromkeys(ordered_mid_ids))
    selected_mid_cards = tuple(mid_cards[mid_id] for mid_id in unique_mid_ids)
    context_mid_ids = (
        unique_mid_ids
        if include_high_child_context or not high_card
        else tuple(dict.fromkeys(direct_mid_ids))
    )
    context_mid_cards = tuple(mid_cards[mid_id] for mid_id in context_mid_ids)
    skill_ids = (
        ((high_card.skill_id,) if high_card else ())
        + tuple(card.skill_id for card in selected_mid_cards)
    )
    return TowerRetrieval(
        high_card=high_card,
        mid_cards=selected_mid_cards,
        high_candidate=high_candidate,
        high_match=high_match,
        direct_mid_candidates=direct_mid_candidates,
        direct_mid_filtered=direct_mid_filtered,
        direct_mid_deduplicated=direct_mid_deduplicated,
        direct_mid_matches=direct_mid_matches,
        skill_ids=skill_ids,
        context=format_tower_context(high_card, context_mid_cards),
    )


def format_tower_context(
    high_card: HighSkillCard | None, mid_cards: Sequence[MidSkillCard]
) -> str:
    sections = []
    if high_card:
        sections.append(
            _format_card(
                "Strategy",
                high_card.name,
                high_card.description,
                high_card.procedure,
            )
        )
    sections.extend(
        _format_card(
            "Skill",
            card.name,
            card.description,
            card.procedure,
            card.constraints,
        )
        for card in mid_cards
    )
    return "\n\n".join(sections)


def mid_card_text(card: MidSkillCard) -> str:
    return "\n".join((card.name, card.description, *card.procedure, *card.constraints))


def high_card_text(card: HighSkillCard) -> str:
    return "\n".join((card.name, card.description, *card.procedure))


def _format_card(
    kind: str,
    name: str,
    description: str,
    procedure: Sequence[str],
    constraints: Sequence[str] = (),
) -> str:
    lines = [f"## {kind}: {name}", f"Use when: {description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(procedure, 1))
    if constraints:
        lines.append("Constraints:")
        lines.extend(f"- {constraint}" for constraint in constraints)
    return "\n".join(lines)
