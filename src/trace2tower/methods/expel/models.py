from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.core.manifests import Benchmark


EXPEL_COMMIT = "e41ec9a24823e7b560c561ab191441b56d9bcefc"


@dataclass(frozen=True, slots=True)
class ExpeLEpisode:
    episode_id: str
    sample_id: str
    task_goal: str
    task_scope: str
    trajectory: str

    def __post_init__(self) -> None:
        values = (
            self.episode_id,
            self.sample_id,
            self.task_goal,
            self.task_scope,
            self.trajectory,
        )
        if any(not value.strip() for value in values):
            raise ValueError("ExpeL episode fields must be non-empty")

    @classmethod
    def from_record(cls, record: Mapping) -> ExpeLEpisode:
        return cls(
            episode_id=str(record["episode_id"]),
            sample_id=str(record["sample_id"]),
            task_goal=str(record["task_goal"]),
            task_scope=str(record["task_scope"]),
            trajectory=str(record["trajectory"]),
        )


@dataclass(frozen=True, slots=True)
class ExpeLExecutionLibrary:
    library_id: str
    benchmark: Benchmark
    source_pool_sha256: str
    expel_commit: str
    rules: tuple[str, ...]
    episodes: tuple[ExpeLEpisode, ...]
    episode_index: SkillEmbeddingIndex

    def __post_init__(self) -> None:
        if len(self.source_pool_sha256) != 64 or len(self.expel_commit) != 40:
            raise ValueError("ExpeL provenance hashes are invalid")
        if not self.rules or len(self.rules) > 20 or any(not rule.strip() for rule in self.rules):
            raise ValueError("ExpeL requires one to twenty non-empty rules")
        episode_ids = tuple(episode.episode_id for episode in self.episodes)
        if not episode_ids or len(episode_ids) != len(set(episode_ids)):
            raise ValueError("ExpeL episodes must be non-empty and uniquely identified")
        if set(self.episode_index.skill_ids) != set(episode_ids):
            raise ValueError("ExpeL episode index does not cover the execution episodes")
        expected_hashes = {
            episode.episode_id: hashlib.sha256(episode.task_goal.encode()).hexdigest()
            for episode in self.episodes
        }
        if self.episode_index.text_hashes and any(
            expected_hashes[episode_id] != text_hash
            for episode_id, text_hash in zip(
                self.episode_index.skill_ids,
                self.episode_index.text_hashes,
                strict=True,
            )
        ):
            raise ValueError("ExpeL episode index text hashes do not match task goals")
        expected_id = f"expellib_{digest(self.content_record())[:16]}"
        if self.library_id != expected_id:
            raise ValueError("ExpeL library ID does not match its contents")

    def content_record(self) -> dict:
        return {
            "benchmark": self.benchmark.value,
            "source_pool_sha256": self.source_pool_sha256,
            "expel_commit": self.expel_commit,
            "rules": list(self.rules),
            "episodes": [asdict(episode) for episode in self.episodes],
            "episode_index": self.episode_index.to_record(),
        }

    def to_record(self) -> dict:
        return {"library_id": self.library_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: Mapping) -> ExpeLExecutionLibrary:
        return cls(
            library_id=str(record["library_id"]),
            benchmark=Benchmark(record["benchmark"]),
            source_pool_sha256=str(record["source_pool_sha256"]),
            expel_commit=str(record["expel_commit"]),
            rules=tuple(str(rule) for rule in record["rules"]),
            episodes=tuple(ExpeLEpisode.from_record(item) for item in record["episodes"]),
            episode_index=SkillEmbeddingIndex.from_record(record["episode_index"]),
        )


def build_execution_library(
    benchmark: Benchmark,
    source_pool_sha256: str,
    rules: tuple[str, ...],
    episodes: tuple[ExpeLEpisode, ...],
    episode_index: SkillEmbeddingIndex,
) -> ExpeLExecutionLibrary:
    content = {
        "benchmark": benchmark.value,
        "source_pool_sha256": source_pool_sha256,
        "expel_commit": EXPEL_COMMIT,
        "rules": list(rules),
        "episodes": [asdict(episode) for episode in episodes],
        "episode_index": episode_index.to_record(),
    }
    return ExpeLExecutionLibrary(
        library_id=f"expellib_{digest(content)[:16]}",
        benchmark=benchmark,
        source_pool_sha256=source_pool_sha256,
        expel_commit=EXPEL_COMMIT,
        rules=rules,
        episodes=episodes,
        episode_index=episode_index,
    )


def digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
