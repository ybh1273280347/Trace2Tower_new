"""Official per-episode result protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from trace2tower.checkpoint import EpisodeCheckpoint, EpisodeKey
from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry


class MethodName(StrEnum):
    NO_SKILL = "no_skill"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    FLAT_SKILL_SUMMARY = "flat_skill_summary"
    SKILLX = "skillx"
    TRACE2TOWER_STATIC = "trace2tower_static"
    TRACE2TOWER_FULL = "trace2tower_full"
    TRACE2TOWER_NO_TRANSITION = "trace2tower_no_transition"
    TRACE2TOWER_NO_OUTCOME = "trace2tower_no_outcome"
    TRACE2TOWER_NO_CONTRASTIVE = "trace2tower_no_contrastive"


class FinishReason(StrEnum):
    COMPLETED = "completed"
    TASK_LIMIT_REACHED = "task_limit_reached"
    AGENT_VALIDATION_FAILED = "agent_validation_failed"
    AGENT_INVALID_ACTION = "agent_invalid_action"
    TASK_ERROR = "task_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    run_id: str
    benchmark: Benchmark
    split: ExperimentSplit
    method: MethodName
    sample_id: str
    repeat_id: int
    shard_id: int
    primary_score: float
    success: bool | None
    steps: int
    invalid_actions: int
    finish_reason: FinishReason
    input_tokens: int | None
    output_tokens: int | None
    billable_tokens: int | None
    latency_ms: int
    skill_ids: tuple[str, ...]
    skill_context_chars: int
    error: None = None

    def __post_init__(self) -> None:
        if self.steps < 0 or not 0 <= self.invalid_actions <= self.steps:
            raise ValueError("invalid episode step counts")
        if self.latency_ms < 0 or self.skill_context_chars < 0:
            raise ValueError("latency and skill context size must be non-negative")
        token_counts = (self.input_tokens, self.output_tokens, self.billable_tokens)
        if any(value is not None and value < 0 for value in token_counts):
            raise ValueError("token counts must be non-negative")
        if not 0 <= self.primary_score <= 1:
            raise ValueError("primary score must be in [0, 1]")
        if self.benchmark is Benchmark.ALFWORLD:
            if self.success is None or self.primary_score not in (0.0, 1.0):
                raise ValueError("ALFWorld requires a binary score and success")
            if self.success is not bool(self.primary_score):
                raise ValueError("ALFWorld score and success disagree")
        if self.benchmark is Benchmark.WEBSHOP and self.success is not None:
            raise ValueError("WebShop result does not define binary success")
        if self.error is not None:
            raise ValueError("official episode results cannot contain errors")

    @property
    def episode_key(self) -> EpisodeKey:
        return EpisodeKey(
            benchmark=self.benchmark,
            split=self.split,
            method=self.method,
            sample_id=self.sample_id,
            repeat_id=self.repeat_id,
        )

    def to_record(self) -> dict:
        record = asdict(self)
        record["skill_ids"] = list(self.skill_ids)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EpisodeResult:
        success = record.get("success")
        return cls(
            run_id=str(record["run_id"]),
            benchmark=Benchmark(record["benchmark"]),
            split=ExperimentSplit(record["split"]),
            method=MethodName(record["method"]),
            sample_id=str(record["sample_id"]),
            repeat_id=int(record["repeat_id"]),
            shard_id=int(record["shard_id"]),
            primary_score=float(record["primary_score"]),
            success=bool(success) if success is not None else None,
            steps=int(record["steps"]),
            invalid_actions=int(record["invalid_actions"]),
            finish_reason=FinishReason(record["finish_reason"]),
            input_tokens=(
                int(record["input_tokens"])
                if record.get("input_tokens") is not None
                else None
            ),
            output_tokens=(
                int(record["output_tokens"])
                if record.get("output_tokens") is not None
                else None
            ),
            billable_tokens=(
                int(record["billable_tokens"])
                if record.get("billable_tokens") is not None
                else None
            ),
            latency_ms=int(record["latency_ms"]),
            skill_ids=tuple(record.get("skill_ids", ())),
            skill_context_chars=int(record["skill_context_chars"]),
            error=record.get("error"),
        )


class EpisodeResultWriter:
    def __init__(self, checkpoint: EpisodeCheckpoint):
        self.checkpoint = checkpoint

    def is_completed(self, entry: ManifestEntry, method: MethodName) -> bool:
        return self.checkpoint.is_completed(
            EpisodeKey(
                benchmark=entry.benchmark,
                split=entry.split,
                method=method,
                sample_id=entry.sample_id,
                repeat_id=entry.repeat_id,
            )
        )

    def write(self, result: EpisodeResult) -> bool:
        return self.checkpoint.write_result(result.to_record())

    def write_error(self, entry: ManifestEntry, method: MethodName, error: str) -> None:
        self.checkpoint.write_error(
            EpisodeKey(
                benchmark=entry.benchmark,
                split=entry.split,
                method=method,
                sample_id=entry.sample_id,
                repeat_id=entry.repeat_id,
            ),
            error,
        )
