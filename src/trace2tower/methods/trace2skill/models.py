from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from trace2tower.core.manifests import Benchmark


TRACE2SKILL_COMMIT = "3d0b52a140f002a512930252b613c49048f7d5ac"


@dataclass(frozen=True, slots=True)
class Trace2SkillArtifact:
    artifact_id: str
    benchmark: Benchmark
    source_pool_sha256: str
    upstream_commit: str
    author_model: str
    trajectory_count: int
    success_count: int
    failure_count: int
    skill_markdown: str
    evolution_signal: str = "combined"

    def __post_init__(self) -> None:
        if len(self.source_pool_sha256) != 64:
            raise ValueError("Trace2Skill source pool hash is invalid")
        if self.upstream_commit != TRACE2SKILL_COMMIT:
            raise ValueError("Trace2Skill artifact uses an unsupported upstream commit")
        if not self.author_model.strip() or not self.skill_markdown.strip():
            raise ValueError("Trace2Skill author model and skill must be non-empty")
        if self.trajectory_count <= 0:
            raise ValueError("Trace2Skill requires trajectory evidence")
        if self.success_count + self.failure_count != self.trajectory_count:
            raise ValueError("Trace2Skill outcome counts do not cover the trajectory pool")
        if self.evolution_signal not in {"combined", "error", "success"}:
            raise ValueError("Trace2Skill evolution signal is invalid")
        expected_id = f"trace2skill_{digest(self.content_record())[:16]}"
        if self.artifact_id != expected_id:
            raise ValueError("Trace2Skill artifact ID does not match its contents")

    def content_record(self) -> dict:
        record = asdict(self)
        record.pop("artifact_id")
        record["benchmark"] = self.benchmark.value
        if self.evolution_signal == "combined":
            record.pop("evolution_signal")
        return record

    def to_record(self) -> dict:
        return {"artifact_id": self.artifact_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: Mapping) -> Trace2SkillArtifact:
        return cls(
            artifact_id=str(record["artifact_id"]),
            benchmark=Benchmark(record["benchmark"]),
            source_pool_sha256=str(record["source_pool_sha256"]),
            upstream_commit=str(record["upstream_commit"]),
            author_model=str(record["author_model"]),
            trajectory_count=int(record["trajectory_count"]),
            success_count=int(record["success_count"]),
            failure_count=int(record["failure_count"]),
            skill_markdown=str(record["skill_markdown"]),
            evolution_signal=str(record.get("evolution_signal", "combined")),
        )


def build_artifact(
    *,
    benchmark: Benchmark,
    source_pool_sha256: str,
    author_model: str,
    trajectory_count: int,
    success_count: int,
    failure_count: int,
    skill_markdown: str,
    evolution_signal: str = "combined",
) -> Trace2SkillArtifact:
    content = {
        "benchmark": benchmark.value,
        "source_pool_sha256": source_pool_sha256,
        "upstream_commit": TRACE2SKILL_COMMIT,
        "author_model": author_model,
        "trajectory_count": trajectory_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "skill_markdown": skill_markdown,
    }
    if evolution_signal != "combined":
        content["evolution_signal"] = evolution_signal
    return Trace2SkillArtifact(
        artifact_id=f"trace2skill_{digest(content)[:16]}",
        benchmark=benchmark,
        source_pool_sha256=source_pool_sha256,
        upstream_commit=TRACE2SKILL_COMMIT,
        author_model=author_model,
        trajectory_count=trajectory_count,
        success_count=success_count,
        failure_count=failure_count,
        skill_markdown=skill_markdown,
        evolution_signal=evolution_signal,
    )


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
