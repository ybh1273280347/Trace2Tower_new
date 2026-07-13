from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SkillMatch:
    skill_id: str
    cosine_similarity: float


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
