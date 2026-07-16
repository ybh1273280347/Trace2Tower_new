from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.methods.trace2tower.skills import HighSkillCard
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch


class TaskCompatibility(StrEnum):
    INCOMPATIBLE = "incompatible"
    WORKFLOW = "workflow"
    PARTIAL = "partial"
    EXACT = "exact"

    @property
    def rank(self) -> int:
        return tuple(TaskCompatibility).index(self)


@dataclass(frozen=True, slots=True)
class TaskProperty:
    name: str
    values: tuple[str, ...]

    def to_record(self) -> dict:
        return {"name": self.name, "values": self.values}

    @classmethod
    def from_record(cls, record: Mapping) -> TaskProperty:
        return cls(str(record["name"]), tuple(str(value) for value in record["values"]))


@dataclass(frozen=True, slots=True)
class TaskCondition:
    task_text: str
    retrieval_text: str
    required_events: tuple[str, ...]
    properties: tuple[TaskProperty, ...]

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.properties)
        if len(names) != len(set(names)):
            raise ValueError("task condition contains duplicate property names")

    def values(self, name: str) -> tuple[str, ...]:
        return next(
            (item.values for item in self.properties if item.name == name),
            (),
        )

    def to_record(self) -> dict:
        return {
            "task_text": self.task_text,
            "retrieval_text": self.retrieval_text,
            "required_events": self.required_events,
            "properties": [item.to_record() for item in self.properties],
        }

    @classmethod
    def from_record(cls, record: Mapping) -> TaskCondition:
        return cls(
            task_text=str(record["task_text"]),
            retrieval_text=str(record["retrieval_text"]),
            required_events=tuple(str(value) for value in record["required_events"]),
            properties=tuple(
                TaskProperty.from_record(item) for item in record["properties"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SkillTaskCondition:
    skill_id: str
    condition: TaskCondition

    def to_record(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "condition": self.condition.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping) -> SkillTaskCondition:
        return cls(
            skill_id=str(record["skill_id"]),
            condition=TaskCondition.from_record(record["condition"]),
        )


@dataclass(frozen=True, slots=True)
class TaskConditionProfile:
    domain: str
    skills: tuple[SkillTaskCondition, ...]

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("task-condition profile requires a domain")
        skill_ids = tuple(item.skill_id for item in self.skills)
        if len(skill_ids) != len(set(skill_ids)):
            raise ValueError("task-condition profile contains duplicate skill IDs")

    @property
    def by_skill_id(self) -> dict[str, TaskCondition]:
        return {item.skill_id: item.condition for item in self.skills}

    def to_record(self) -> dict:
        return {
            "domain": self.domain,
            "skills": [item.to_record() for item in self.skills],
        }

    @classmethod
    def from_record(cls, record: Mapping) -> TaskConditionProfile:
        return cls(
            domain=str(record["domain"]),
            skills=tuple(
                SkillTaskCondition.from_record(item) for item in record["skills"]
            ),
        )


class DomainTaskAdapter(Protocol):
    domain: str

    def extract_query(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> TaskCondition: ...

    def profile_condition(self, record: Mapping) -> TaskCondition: ...

    def compatibility(
        self,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> TaskCompatibility: ...

    def bind(
        self,
        source_card: HighSkillCard,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> HighSkillCard: ...


@dataclass(frozen=True, slots=True)
class ConditionedSkillMatch:
    semantic_match: SkillMatch
    compatibility: TaskCompatibility


def retrieve_task_conditioned_high(
    query_vector: Sequence[float],
    index: SkillEmbeddingIndex,
    query: TaskCondition,
    conditions: Mapping[str, TaskCondition],
    adapter: DomainTaskAdapter,
    *,
    minimum_compatibility: TaskCompatibility,
    similarity_threshold: float = -1.0,
) -> ConditionedSkillMatch | None:
    if set(index.skill_ids) != set(conditions):
        raise ValueError("task conditions must cover the High index")

    eligible = []
    for semantic_match in index.search(query_vector, len(index.skill_ids)):
        if semantic_match.cosine_similarity < similarity_threshold:
            continue
        compatibility = adapter.compatibility(
            query,
            conditions[semantic_match.skill_id],
        )
        if compatibility.rank >= minimum_compatibility.rank:
            eligible.append(ConditionedSkillMatch(semantic_match, compatibility))
    return max(
        eligible,
        key=lambda item: (
            item.compatibility.rank,
            item.semantic_match.cosine_similarity,
        ),
        default=None,
    )
