from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch


@dataclass(frozen=True, slots=True)
class ObjectConditionedHighProfile:
    target_object: str
    transformation: str
    destination_receptacles: tuple[str, ...]
    transformation_devices: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record: Mapping) -> ObjectConditionedHighProfile:
        return cls(
            target_object=str(record["target_object"]),
            transformation=str(record["transformation"]),
            destination_receptacles=tuple(record["destination_receptacles"]),
            transformation_devices=tuple(record.get("transformation_devices", ())),
        )


def retrieve_object_conditioned_high(
    query_vector: Sequence[float],
    index: SkillEmbeddingIndex,
    profiles: Mapping[str, ObjectConditionedHighProfile],
    *,
    target_object: str,
    transformation: str,
    destination: str,
    similarity_threshold: float = -1.0,
) -> SkillMatch | None:
    if set(index.skill_ids) != set(profiles):
        raise ValueError("object-conditioned profiles must cover the High index")
    if not target_object or not transformation or not destination:
        return None
    for match in index.search(query_vector, len(index.skill_ids)):
        profile = profiles[match.skill_id]
        if (
            profile.target_object == target_object
            and profile.transformation == transformation
            and destination in profile.destination_receptacles
        ):
            return match if match.cosine_similarity >= similarity_threshold else None
    return None
