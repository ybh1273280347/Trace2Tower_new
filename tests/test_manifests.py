import pytest

from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    ManifestEntry,
    expand_manifest_repeats,
    shard_counts,
)


def test_sharding_is_deterministic_and_balanced() -> None:
    entries = [
        ManifestEntry(
            benchmark=Benchmark.WEBSHOP,
            split=ExperimentSplit.TEST,
            sample_id=f"webshop:{index}",
            dataset_index=index,
            source_split="goals",
            repeat_id=0,
        )
        for index in reversed(range(23))
    ]

    assert shard_counts(entries, 10) == [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]


def test_explicit_repeats_expand_episode_keys_without_changing_provenance() -> None:
    source = ManifestEntry(
        Benchmark.WEBSHOP,
        ExperimentSplit.TRAIN,
        "webshop:7",
        7,
        "goals",
        0,
    )
    expanded = expand_manifest_repeats((source,), (2, 0, 1))
    assert [entry.repeat_id for entry in expanded] == [0, 1, 2]
    assert {entry.dataset_index for entry in expanded} == {7}
    with pytest.raises(ValueError, match="unique non-negative"):
        expand_manifest_repeats((source,), (0, 0))
