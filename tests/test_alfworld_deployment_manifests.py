from __future__ import annotations

from scripts.experiments.data.prepare_alfworld_deployment_manifests import (
    PARTITION_NAMES,
    partition_train_tasks,
    random_subset,
)
from trace2tower.core.manifests import Benchmark, ExperimentSplit, ManifestEntry


def entry(index: int) -> ManifestEntry:
    return ManifestEntry(
        benchmark=Benchmark.ALFWORLD,
        split=ExperimentSplit.TRAIN,
        sample_id=f"alfworld:train:task-{index:02d}",
        dataset_index=index,
        source_split="train",
        repeat_id=0,
    )


def test_deployment_partition_is_random_complete_and_deterministic() -> None:
    candidates = [(entry(index), "family-a" if index < 6 else "family-b") for index in range(12)]

    first = partition_train_tasks(candidates, (6, 3, 3), seed=17)
    second = partition_train_tasks(candidates, (6, 3, 3), seed=17)

    assert first == second
    assert [len(first[name]) for name in PARTITION_NAMES] == [6, 3, 3]
    selected_ids = [item.sample_id for name in PARTITION_NAMES for item, _ in first[name]]
    assert len(selected_ids) == len(set(selected_ids)) == 12


def test_feedback_pilot_is_a_deterministic_random_subset() -> None:
    candidates = [(entry(index), "family-a" if index < 6 else "family-b") for index in range(12)]

    first = random_subset(candidates, 4, seed=19)
    second = random_subset(candidates, 4, seed=19)

    assert first == second
    assert len(first) == 4
    assert {item.sample_id for item, _ in first} <= {item.sample_id for item, _ in candidates}
