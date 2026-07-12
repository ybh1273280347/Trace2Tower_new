from trace2tower.manifests import Benchmark, ExperimentSplit, ManifestEntry, shard_counts


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
