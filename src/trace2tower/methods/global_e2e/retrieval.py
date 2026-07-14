from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.methods.global_e2e.models import GlobalE2ESkillCard
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch


@dataclass(frozen=True, slots=True)
class GlobalE2ERetrieval:
    card: GlobalE2ESkillCard
    match: SkillMatch
    context: str


def retrieve_global_e2e_skill(
    query_vector: Sequence[float],
    index: SkillEmbeddingIndex,
    cards: Mapping[str, GlobalE2ESkillCard],
) -> GlobalE2ERetrieval:
    if set(index.skill_ids) != set(cards):
        raise ValueError("Global E2E index and card library differ")
    matches = index.search(query_vector, 1)
    if not matches:
        raise ValueError("Global E2E library is empty")
    match = matches[0]
    card = cards[match.skill_id]
    return GlobalE2ERetrieval(card, match, format_global_e2e_card(card))


def format_global_e2e_card(card: GlobalE2ESkillCard) -> str:
    lines = [f"## Skill: {card.name}", f"Use when: {card.description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(card.procedure, 1))
    lines.append("Constraints:")
    lines.extend(f"- {constraint}" for constraint in card.constraints)
    return "\n".join(lines)
