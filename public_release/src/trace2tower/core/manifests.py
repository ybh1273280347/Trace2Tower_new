"""Fixed task manifests shared by every experiment method."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

import pyarrow.parquet as parquet


class Benchmark(StrEnum):
    ALFWORLD = "alfworld"
    WEBSHOP = "webshop"


class ExperimentSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    ABLATION = "ablation"


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    benchmark: Benchmark
    split: ExperimentSplit
    sample_id: str
    dataset_index: int
    source_split: str
    repeat_id: int

    @classmethod
    def from_record(cls, record: dict) -> ManifestEntry:
        return cls(
            benchmark=Benchmark(record["benchmark"]),
            split=ExperimentSplit(record["split"]),
            sample_id=str(record["sample_id"]),
            dataset_index=int(record["dataset_index"]),
            source_split=str(record["source_split"]),
            repeat_id=int(record["repeat_id"]),
        )

    @property
    def manifest_key(self) -> tuple[str, str, str, int]:
        return self.benchmark, self.split, self.sample_id, self.repeat_id


def build_alfworld_manifests(
    dataset_root: Path,
    split_sources: dict[ExperimentSplit, str],
    repeat_ids: Iterable[int],
) -> dict[ExperimentSplit, list[ManifestEntry]]:
    manifests = {}
    for split, source_split in split_sources.items():
        table = parquet.read_table(dataset_root / f"{source_split}.parquet", columns=["extra_info"])
        extra_info = table.column("extra_info").combine_chunks().to_pylist()
        entries = []
        for dataset_index, item in enumerate(extra_info):
            task_id = item["task_id"]
            for repeat_id in repeat_ids:
                entries.append(
                    ManifestEntry(
                        benchmark=Benchmark.ALFWORLD,
                        split=split,
                        sample_id=f"alfworld:{source_split}:{task_id}",
                        dataset_index=dataset_index,
                        source_split=source_split,
                        repeat_id=repeat_id,
                    )
                )
        validate_manifest(entries)
        manifests[split] = entries
    return manifests


def build_webshop_manifests(
    goals_path: Path,
    split_ranges: dict[ExperimentSplit, tuple[int, int]],
    repeat_ids: Iterable[int],
) -> dict[ExperimentSplit, list[ManifestEntry]]:
    goal_count = len(json.loads(goals_path.read_text(encoding="utf-8")))
    manifests = {}
    for split, (start, end) in split_ranges.items():
        if start < 0 or end > goal_count or start >= end:
            raise ValueError(f"invalid WebShop goal range for {split}: [{start}, {end})")
        entries = [
            ManifestEntry(
                benchmark=Benchmark.WEBSHOP,
                split=split,
                sample_id=f"webshop:{goal_index}",
                dataset_index=goal_index,
                source_split="goals",
                repeat_id=repeat_id,
            )
            for goal_index in range(start, end)
            for repeat_id in repeat_ids
        ]
        validate_manifest(entries)
        manifests[split] = entries
    return manifests


def validate_manifest(entries: Iterable[ManifestEntry]) -> None:
    seen = set()
    for entry in entries:
        if entry.manifest_key in seen:
            raise ValueError(f"duplicate manifest entry: {entry.manifest_key}")
        seen.add(entry.manifest_key)


def expand_manifest_repeats(
    entries: Iterable[ManifestEntry], repeat_ids: Iterable[int]
) -> list[ManifestEntry]:
    selected = list(entries)
    repeats = tuple(repeat_ids)
    if not repeats:
        return selected
    if any(repeat_id < 0 for repeat_id in repeats) or len(set(repeats)) != len(repeats):
        raise ValueError("repeat IDs must be unique non-negative integers")

    templates = {}
    signatures = {}
    for entry in selected:
        key = (entry.benchmark, entry.split, entry.sample_id)
        signature = (entry.dataset_index, entry.source_split)
        if key in signatures and signatures[key] != signature:
            raise ValueError("sample repeat entries disagree on dataset provenance")
        templates.setdefault(key, entry)
        signatures[key] = signature
    expanded = [
        replace(template, repeat_id=repeat_id)
        for template in templates.values()
        for repeat_id in sorted(repeats)
    ]
    validate_manifest(expanded)
    return expanded


def shard_counts(entries: Iterable[ManifestEntry], num_shards: int) -> list[int]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    counts = [0] * num_shards
    ordered = sorted(entries, key=lambda entry: (entry.sample_id, entry.repeat_id))
    for index, _ in enumerate(ordered):
        counts[index % num_shards] += 1
    return counts


def select_shard(
    entries: Iterable[ManifestEntry], shard_id: int, num_shards: int
) -> list[ManifestEntry]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards})")
    ordered = sorted(entries, key=lambda entry: (entry.sample_id, entry.repeat_id))
    return [entry for index, entry in enumerate(ordered) if index % num_shards == shard_id]


def read_manifest(path: Path) -> list[ManifestEntry]:
    entries = [
        ManifestEntry.from_record(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    validate_manifest(entries)
    return entries


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        temporary_path = Path(output_file.name)
        for entry in entries:
            json.dump(asdict(entry), output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)
