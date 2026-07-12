from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard


@dataclass(frozen=True, slots=True)
class SkillMatch:
    skill_id: str
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class SkillEmbeddingIndex:
    skill_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.skill_ids) != len(self.vectors):
            raise ValueError("skill IDs and embedding vectors must align")
        if len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("skill embedding index contains duplicate IDs")
        dimensions = {len(vector) for vector in self.vectors}
        if self.vectors and (len(dimensions) != 1 or 0 in dimensions):
            raise ValueError("skill embedding vectors must have one nonzero dimension")

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> SkillEmbeddingIndex:
        return cls(
            skill_ids=tuple(record["skill_ids"]),
            vectors=tuple(
                tuple(float(value) for value in vector) for vector in record["vectors"]
            ),
        )

    def top_k(self, query_vector: Sequence[float], count: int) -> tuple[str, ...]:
        return tuple(match.skill_id for match in self.search(query_vector, count))

    def search(
        self, query_vector: Sequence[float], count: int
    ) -> tuple[SkillMatch, ...]:
        if count < 0:
            raise ValueError("retrieval count must be non-negative")
        if not self.vectors or count == 0:
            return ()
        query = np.asarray(query_vector, dtype=np.float64)
        vectors = np.asarray(self.vectors, dtype=np.float64)
        if query.shape != (vectors.shape[1],):
            raise ValueError("query embedding dimension does not match the skill index")
        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(vectors, axis=1)
        scores = np.divide(
            vectors @ query,
            vector_norms * query_norm,
            out=np.full(len(vectors), -np.inf),
            where=(vector_norms > 0) & (query_norm > 0),
        )
        ranked = sorted(
            zip(self.skill_ids, scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(
            SkillMatch(skill_id, float(score)) for skill_id, score in ranked[:count]
        )


@dataclass(frozen=True, slots=True)
class TowerRetrieval:
    high_card: HighSkillCard | None
    mid_cards: tuple[MidSkillCard, ...]
    high_match: SkillMatch | None
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
) -> TowerRetrieval:
    missing_high_cards = set(high_index.skill_ids) - set(high_cards)
    missing_mid_cards = set(mid_index.skill_ids) - set(mid_cards)
    if missing_high_cards or missing_mid_cards:
        raise ValueError("skill embedding index references missing cards")
    high_matches = high_index.search(high_query_vector, high_top_k)
    high_match = high_matches[0] if high_matches else None
    high_card = high_cards[high_match.skill_id] if high_match else None
    if high_card and not set(high_card.ordered_mid_ids) <= set(mid_cards):
        raise ValueError("retrieved High card references missing child Mid cards")
    direct_mid_matches = mid_index.search(mid_query_vector, direct_mid_top_k)
    direct_mid_ids = tuple(match.skill_id for match in direct_mid_matches)
    ordered_mid_ids = (*high_card.ordered_mid_ids, *direct_mid_ids) if high_card else direct_mid_ids
    unique_mid_ids = tuple(dict.fromkeys(ordered_mid_ids))
    selected_mid_cards = tuple(mid_cards[mid_id] for mid_id in unique_mid_ids)
    skill_ids = (
        ((high_card.skill_id,) if high_card else ())
        + tuple(card.skill_id for card in selected_mid_cards)
    )
    return TowerRetrieval(
        high_card=high_card,
        mid_cards=selected_mid_cards,
        high_match=high_match,
        direct_mid_matches=direct_mid_matches,
        skill_ids=skill_ids,
        context=format_tower_context(high_card, selected_mid_cards),
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
