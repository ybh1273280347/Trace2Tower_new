from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from trace2tower.checkpoint import EpisodeKey
from trace2tower.manifests import Benchmark, ManifestEntry, select_shard
from trace2tower.results import MethodName
from trace2tower.trajectory import TrajectoryReader


@dataclass(frozen=True, slots=True)
class ShardAudit:
    benchmark: Benchmark
    method: MethodName
    shard_id: int
    expected_count: int
    result_count: int
    trajectory_count: int
    missing_results: tuple[str, ...]
    unexpected_results: tuple[str, ...]
    missing_trajectories: tuple[str, ...]
    unexpected_trajectories: tuple[str, ...]
    unresolved_errors: tuple[str, ...]
    mismatched_records: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not any(
            (
                self.missing_results,
                self.unexpected_results,
                self.missing_trajectories,
                self.unexpected_trajectories,
                self.unresolved_errors,
                self.mismatched_records,
            )
        )

    def to_record(self) -> dict:
        return {
            "benchmark": self.benchmark.value,
            "method": self.method.value,
            "shard_id": self.shard_id,
            "expected_count": self.expected_count,
            "result_count": self.result_count,
            "trajectory_count": self.trajectory_count,
            "missing_results": list(self.missing_results),
            "unexpected_results": list(self.unexpected_results),
            "missing_trajectories": list(self.missing_trajectories),
            "unexpected_trajectories": list(self.unexpected_trajectories),
            "unresolved_errors": list(self.unresolved_errors),
            "mismatched_records": list(self.mismatched_records),
            "complete": self.complete,
        }


def audit_training_shard(
    entries: Iterable[ManifestEntry],
    *,
    run_id: str,
    benchmark: Benchmark,
    method: MethodName,
    shard_id: int,
    num_shards: int,
    run_dir: Path,
    pool_path: Path,
    max_episodes: int | None = None,
) -> ShardAudit:
    selected = select_shard(entries, shard_id, num_shards)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        selected = selected[:max_episodes]
    expected_keys = {
        EpisodeKey(
            entry.benchmark,
            entry.split,
            method,
            entry.sample_id,
            entry.repeat_id,
        )
        for entry in selected
    }

    result_records = _read_records(run_dir / "results.jsonl")
    results = {}
    mismatched = set()
    for record in result_records:
        key = EpisodeKey.from_record(record)
        if key in results:
            raise ValueError(f"duplicate result key: {_display_key(key)}")
        results[key] = record
        if (
            record.get("run_id") != run_id
            or record.get("primary_score") is None
            or record.get("error") is not None
        ):
            mismatched.add(_display_key(key))

    trajectories = () if not pool_path.exists() else TrajectoryReader.read_jsonl(pool_path)
    trajectory_records = {}
    for trajectory in trajectories:
        key = EpisodeKey(
            trajectory.benchmark,
            trajectory.split,
            trajectory.method,
            trajectory.sample_id,
            trajectory.repeat_id,
        )
        trajectory_records[key] = trajectory
        if trajectory.run_id != run_id:
            mismatched.add(_display_key(key))

    for key in results.keys() & trajectory_records.keys():
        result = results[key]
        trajectory = trajectory_records[key]
        if (
            float(result["primary_score"]) != trajectory.primary_score
            or result["finish_reason"] != trajectory.finish_reason
            or int(result["steps"]) != len(trajectory.steps)
        ):
            mismatched.add(_display_key(key))

    error_keys = {
        EpisodeKey.from_record(record)
        for record in _read_records(run_dir / "errors.jsonl")
    }
    result_keys = set(results)
    trajectory_keys = set(trajectory_records)
    return ShardAudit(
        benchmark=benchmark,
        method=method,
        shard_id=shard_id,
        expected_count=len(expected_keys),
        result_count=len(result_keys),
        trajectory_count=len(trajectory_keys),
        missing_results=_display_keys(expected_keys - result_keys),
        unexpected_results=_display_keys(result_keys - expected_keys),
        missing_trajectories=_display_keys(expected_keys - trajectory_keys),
        unexpected_trajectories=_display_keys(trajectory_keys - expected_keys),
        unresolved_errors=_display_keys(error_keys - result_keys),
        mismatched_records=tuple(sorted(mismatched)),
    )


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _display_keys(keys: set[EpisodeKey]) -> tuple[str, ...]:
    return tuple(sorted(_display_key(key) for key in keys))


def _display_key(key: EpisodeKey) -> str:
    return f"{key.sample_id}#{key.repeat_id}"
