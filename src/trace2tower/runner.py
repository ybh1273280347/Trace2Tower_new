"""Resumable asynchronous execution over deterministic manifest shards."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from trace2tower.checkpoint import EpisodeKey
from trace2tower.manifests import ManifestEntry, select_shard
from trace2tower.results import EpisodeResult, EpisodeResultWriter, MethodName


EpisodeExecutor = Callable[[ManifestEntry, int], Awaitable[EpisodeResult]]


@dataclass(frozen=True, slots=True)
class RunSummary:
    selected: int
    skipped: int
    completed: int
    failed: int


async def run_shard(
    entries: Iterable[ManifestEntry],
    *,
    method: MethodName,
    shard_id: int,
    num_shards: int,
    writer: EpisodeResultWriter,
    executor: EpisodeExecutor,
    max_concurrency: int,
    max_episodes: int | None = None,
    dry_run: bool = False,
) -> RunSummary:
    selected = select_shard(entries, shard_id, num_shards)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        selected = selected[:max_episodes]
    pending = [entry for entry in selected if not writer.is_completed(entry, method)]
    skipped = len(selected) - len(pending)
    if dry_run:
        return RunSummary(len(selected), skipped, 0, 0)
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(entry: ManifestEntry) -> bool:
        async with semaphore:
            try:
                result = await executor(entry, shard_id)
                if result.episode_key != _episode_key(entry, method):
                    raise ValueError(f"executor returned a mismatched result for {entry.sample_id}")
                if result.shard_id != shard_id:
                    raise ValueError(f"executor returned the wrong shard for {entry.sample_id}")
                writer.write(result)
                return True
            except Exception as exc:
                writer.write_error(entry, method, f"{type(exc).__name__}: {exc}")
                return False

    outcomes = await asyncio.gather(*(execute(entry) for entry in pending))
    completed = sum(outcomes)
    return RunSummary(len(selected), skipped, completed, len(outcomes) - completed)


def _episode_key(entry: ManifestEntry, method: MethodName) -> EpisodeKey:
    return EpisodeKey(
        benchmark=entry.benchmark,
        split=entry.split,
        method=method,
        sample_id=entry.sample_id,
        repeat_id=entry.repeat_id,
    )
