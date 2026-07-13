from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from trace2tower.methods.flat_skill_summary.models import FlatSkillCard
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch


@dataclass(frozen=True, slots=True)
class FlatRetrieval:
    cards: tuple[FlatSkillCard, ...]
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
    top_k: int = 3,
) -> FlatRetrieval:
    if set(index.skill_ids) != set(cards):
        raise ValueError("Flat index and card library differ")
    matches = index.search(query_vector, top_k)
    selected = tuple(cards[match.skill_id] for match in matches)
    return FlatRetrieval(
        cards=selected,
        matches=matches,
        context="\n\n".join(format_flat_card(card) for card in selected),
    )


def format_flat_card(card: FlatSkillCard) -> str:
    lines = [f"## Skill: {card.name}", f"Use when: {card.description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(card.procedure, 1))
    lines.append("Constraints:")
    lines.extend(f"- {constraint}" for constraint in card.constraints)
    return "\n".join(lines)
