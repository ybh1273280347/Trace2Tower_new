from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.core.manifests import Benchmark


@dataclass(frozen=True, slots=True)
class SkillXPlan:
    plan_id: str
    source_sha256: str
    task: str
    plan: str

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> SkillXPlan:
        return cls(
            plan_id=str(record["plan_id"]),
            source_sha256=str(record["source_sha256"]),
            task=str(record["task"]),
            plan=str(record["plan"]),
        )


@dataclass(frozen=True, slots=True)
class SkillXCard:
    skill_id: str
    source_sha256: str
    name: str
    document: str
    content: str
    tools: tuple[str, ...]
    skill_type: str

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> SkillXCard:
        return cls(
            skill_id=str(record["skill_id"]),
            source_sha256=str(record["source_sha256"]),
            name=str(record["name"]),
            document=str(record["document"]),
            content=str(record["content"]),
            tools=tuple(record["tools"]),
            skill_type=str(record["skill_type"]),
        )


@dataclass(frozen=True, slots=True)
class SkillXExecutionLibrary:
    library_id: str
    benchmark: Benchmark
    source_library_sha256: str
    skillx_commit: str
    plans: tuple[SkillXPlan, ...]
    skills: tuple[SkillXCard, ...]
    plan_index: SkillEmbeddingIndex
    skill_index: SkillEmbeddingIndex

    def __post_init__(self) -> None:
        hashes = (
            self.source_library_sha256,
            *(plan.source_sha256 for plan in self.plans),
            *(skill.source_sha256 for skill in self.skills),
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("SkillX provenance hashes must be SHA-256")
        plan_ids = {plan.plan_id for plan in self.plans}
        skill_ids = {skill.skill_id for skill in self.skills}
        if len(plan_ids) != len(self.plans) or len(skill_ids) != len(self.skills):
            raise ValueError("SkillX execution library contains duplicate IDs")
        if set(self.plan_index.skill_ids) != plan_ids or (
            plan_ids and not self.plan_index.text_hashes
        ):
            raise ValueError("SkillX plan index must cover every plan with text hashes")
        if set(self.skill_index.skill_ids) != skill_ids or (
            skill_ids and not self.skill_index.text_hashes
        ):
            raise ValueError("SkillX skill index must cover every skill with text hashes")
        if any(skill.skill_type not in {"functional", "atomic"} for skill in self.skills):
            raise ValueError("SkillX execution library has an unknown skill type")
        if self.library_id:
            expected = f"skillxlib_{digest(self.content_record())[:16]}"
            if self.library_id != expected:
                raise ValueError("SkillX execution library ID does not match its contents")

    def content_record(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "source_library_sha256": self.source_library_sha256,
            "skillx_commit": self.skillx_commit,
            "plans": [plan.to_record() for plan in self.plans],
            "skills": [skill.to_record() for skill in self.skills],
            "plan_index": self.plan_index.to_record(),
            "skill_index": self.skill_index.to_record(),
        }

    def to_record(self) -> dict:
        return {"library_id": self.library_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: Mapping) -> SkillXExecutionLibrary:
        return cls(
            library_id=str(record["library_id"]),
            benchmark=Benchmark(record["benchmark"]),
            source_library_sha256=str(record["source_library_sha256"]),
            skillx_commit=str(record["skillx_commit"]),
            plans=tuple(SkillXPlan.from_record(item) for item in record["plans"]),
            skills=tuple(SkillXCard.from_record(item) for item in record["skills"]),
            plan_index=SkillEmbeddingIndex.from_record(record["plan_index"]),
            skill_index=SkillEmbeddingIndex.from_record(record["skill_index"]),
        )


def build_execution_library(
    benchmark: Benchmark,
    source_library_sha256: str,
    skillx_commit: str,
    plans: tuple[SkillXPlan, ...],
    skills: tuple[SkillXCard, ...],
    plan_index: SkillEmbeddingIndex,
    skill_index: SkillEmbeddingIndex,
) -> SkillXExecutionLibrary:
    library = SkillXExecutionLibrary(
        library_id="",
        benchmark=benchmark,
        source_library_sha256=source_library_sha256,
        skillx_commit=skillx_commit,
        plans=tuple(sorted(plans, key=lambda plan: plan.plan_id)),
        skills=tuple(sorted(skills, key=lambda skill: skill.skill_id)),
        plan_index=plan_index,
        skill_index=skill_index,
    )
    return replace(
        library,
        library_id=f"skillxlib_{digest(library.content_record())[:16]}",
    )


def digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_text(plan: SkillXPlan) -> str:
    return f"{plan.task}\n{plan.plan}"


def skill_text(skill: SkillXCard) -> str:
    return f"{skill.name}\n{skill.document}\n{skill.content}"
