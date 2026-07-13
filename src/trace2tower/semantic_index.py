from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SkillMatch:
    skill_id: str
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class DiverseSkillMatches:
    candidates: tuple[SkillMatch, ...]
    filtered: tuple[SkillMatch, ...]
    deduplicated: tuple[SkillMatch, ...]
    selected: tuple[SkillMatch, ...]


@dataclass(frozen=True, slots=True)
class SkillEmbeddingIndex:
    skill_ids: tuple[str, ...]
    vectors: tuple[tuple[float, ...], ...]
    text_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.skill_ids) != len(self.vectors):
            raise ValueError("skill IDs and embedding vectors must align")
        if len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("skill embedding index contains duplicate IDs")
        if self.text_hashes and len(self.text_hashes) != len(self.skill_ids):
            raise ValueError("skill IDs and text hashes must align")
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
            text_hashes=tuple(record.get("text_hashes", ())),
        )

    def top_k(self, query_vector: Sequence[float], count: int) -> tuple[str, ...]:
        return tuple(match.skill_id for match in self.search(query_vector, count))

    def search(
        self,
        query_vector: Sequence[float],
        count: int,
        *,
        score_penalties: Mapping[str, float] | None = None,
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
        penalties = score_penalties or {}
        if set(penalties) - set(self.skill_ids) or any(
            penalty < 0 for penalty in penalties.values()
        ):
            raise ValueError("skill score penalties are invalid")
        ranked = sorted(
            zip(self.skill_ids, scores, strict=True),
            key=lambda item: (
                -(item[1] - penalties.get(item[0], 0.0)),
                -item[1],
                item[0],
            ),
        )
        return tuple(
            SkillMatch(skill_id, float(score)) for skill_id, score in ranked[:count]
        )


def diverse_search(
    index: SkillEmbeddingIndex,
    query_vector: Sequence[float],
    *,
    candidate_count: int,
    similarity_threshold: float,
    relative_margin: float,
    dedup_similarity_threshold: float,
    relevance_weight: float,
    max_count: int,
    score_penalties: Mapping[str, float] | None = None,
) -> DiverseSkillMatches:
    if candidate_count <= 0 or max_count <= 0:
        raise ValueError("diverse retrieval counts must be positive")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity threshold must be between zero and one")
    if not 0 <= relative_margin <= 2:
        raise ValueError("relative margin must be between zero and two")
    if not 0 <= dedup_similarity_threshold <= 1:
        raise ValueError("deduplication threshold must be between zero and one")
    if not 0 <= relevance_weight <= 1:
        raise ValueError("MMR relevance weight must be between zero and one")

    penalties = score_penalties or {}
    candidates = index.search(
        query_vector,
        candidate_count,
        score_penalties=penalties,
    )
    best_similarity = max(
        (match.cosine_similarity for match in candidates),
        default=-1.0,
    )
    filtered = tuple(
        match
        for match in candidates
        if match.cosine_similarity >= similarity_threshold
        and match.cosine_similarity >= best_similarity - relative_margin
    )
    vectors = dict(zip(index.skill_ids, index.vectors, strict=True))
    deduplicated = []
    for match in filtered:
        if any(
            _cosine(vectors[match.skill_id], vectors[item.skill_id])
            > dedup_similarity_threshold
            for item in deduplicated
        ):
            continue
        deduplicated.append(match)

    remaining = list(deduplicated)
    selected = []
    while remaining and len(selected) < max_count:
        ranked = sorted(
            remaining,
            key=lambda match: (
                -(
                    _mmr_score(match, selected, vectors, relevance_weight)
                    - penalties.get(match.skill_id, 0.0)
                ),
                -match.cosine_similarity,
                match.skill_id,
            ),
        )
        selected.append(ranked[0])
        remaining.remove(ranked[0])
    return DiverseSkillMatches(candidates, filtered, tuple(deduplicated), tuple(selected))


def _mmr_score(
    match: SkillMatch,
    selected: list[SkillMatch],
    vectors: Mapping[str, tuple[float, ...]],
    relevance_weight: float,
) -> float:
    redundancy = max(
        (
            _cosine(vectors[match.skill_id], vectors[item.skill_id])
            for item in selected
        ),
        default=0.0,
    )
    return relevance_weight * match.cosine_similarity - (1 - relevance_weight) * redundancy


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float64)
    right_vector = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    return float(left_vector @ right_vector / denominator) if denominator else -1.0
