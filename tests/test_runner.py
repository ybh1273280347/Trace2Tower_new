from __future__ import annotations

import asyncio
from pathlib import Path

from trace2tower.checkpoint import EpisodeCheckpoint
from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry
from trace2tower.results import (
    EpisodeResult,
    EpisodeResultWriter,
    FinishReason,
    MethodName,
)
from trace2tower.runner import run_shard


def make_result(entry: ManifestEntry, shard_id: int) -> EpisodeResult:
    return EpisodeResult(
        run_id="test-run",
        benchmark=entry.benchmark,
        split=entry.split,
        method=MethodName.NO_SKILL,
        sample_id=entry.sample_id,
        repeat_id=entry.repeat_id,
        shard_id=shard_id,
        primary_score=0.5,
        success=None,
        steps=2,
        invalid_actions=0,
        finish_reason=FinishReason.COMPLETED,
        input_tokens=10,
        output_tokens=2,
        billable_tokens=None,
        latency_ms=5,
        skill_ids=(),
        skill_context_chars=0,
    )


def test_runner_resumes_only_failed_episode(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(
            benchmark=Benchmark.WEBSHOP,
            split=ExperimentSplit.TEST,
            sample_id=f"webshop:{index}",
            dataset_index=index,
            source_split="goals",
            repeat_id=0,
        )
        for index in range(6)
    ]
    results_path = tmp_path / "episodes.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    writer = EpisodeResultWriter(EpisodeCheckpoint(results_path, errors_path))

    async def interrupted(entry: ManifestEntry, shard_id: int) -> EpisodeResult:
        if entry.dataset_index == 3:
            raise ConnectionError("simulated interruption")
        return make_result(entry, shard_id)

    dry_run = asyncio.run(
        run_shard(
            entries,
            method=MethodName.NO_SKILL,
            shard_id=0,
            num_shards=1,
            writer=writer,
            executor=interrupted,
            max_concurrency=3,
            dry_run=True,
        )
    )
    assert (dry_run.selected, dry_run.completed, dry_run.failed) == (6, 0, 0)
    assert not results_path.exists()

    first = asyncio.run(
        run_shard(
            entries,
            method=MethodName.NO_SKILL,
            shard_id=0,
            num_shards=1,
            writer=writer,
            executor=interrupted,
            max_concurrency=3,
        )
    )
    assert (first.completed, first.failed, first.skipped) == (5, 1, 0)

    resumed_writer = EpisodeResultWriter(EpisodeCheckpoint(results_path, errors_path))

    async def succeeds(entry: ManifestEntry, shard_id: int) -> EpisodeResult:
        return make_result(entry, shard_id)

    second = asyncio.run(
        run_shard(
            entries,
            method=MethodName.NO_SKILL,
            shard_id=0,
            num_shards=1,
            writer=resumed_writer,
            executor=succeeds,
            max_concurrency=3,
        )
    )
    assert (second.completed, second.failed, second.skipped) == (1, 0, 5)
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 6
    assert "simulated interruption" in errors_path.read_text(encoding="utf-8")


def test_runner_shares_episode_semaphore_across_shards(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(
            benchmark=Benchmark.WEBSHOP,
            split=ExperimentSplit.TRAIN,
            sample_id=f"webshop:{index}",
            dataset_index=index,
            source_split="goals",
            repeat_id=0,
        )
        for index in range(6)
    ]
    active = 0
    maximum_active = 0

    async def execute(entry: ManifestEntry, shard_id: int) -> EpisodeResult:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return make_result(entry, shard_id)

    async def run() -> None:
        shared_semaphore = asyncio.Semaphore(2)
        await asyncio.gather(
            *(
                run_shard(
                    entries,
                    method=MethodName.NO_SKILL,
                    shard_id=shard_id,
                    num_shards=2,
                    writer=EpisodeResultWriter(
                        EpisodeCheckpoint(
                            tmp_path / f"results-{shard_id}.jsonl",
                            tmp_path / f"errors-{shard_id}.jsonl",
                        )
                    ),
                    executor=execute,
                    max_concurrency=3,
                    episode_semaphore=shared_semaphore,
                )
                for shard_id in range(2)
            )
        )

    asyncio.run(run())
    assert maximum_active == 2
