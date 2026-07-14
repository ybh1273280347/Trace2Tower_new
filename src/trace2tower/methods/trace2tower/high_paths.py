from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from trace2tower.methods.trace2tower.models import HighPath, MidCluster


def compress_repeated_mid_ids(mid_ids: Iterable[str]) -> tuple[str, ...]:
    compressed = []
    for mid_id in mid_ids:
        if not compressed or compressed[-1] != mid_id:
            compressed.append(mid_id)
    return tuple(compressed)


def trajectory_mid_sequences(
    records: Sequence[Mapping], clusters: Iterable[MidCluster]
) -> dict[str, tuple[str, ...]]:
    segment_to_mid = {
        segment_id: cluster.cluster_id
        for cluster in clusters
        for segment_id in cluster.member_segment_ids
    }
    sequences = {}
    for record in sorted(records, key=lambda item: item["trajectory_id"]):
        ordered_segments = sorted(record["segments"], key=lambda item: item["start_step"])
        try:
            mid_ids = (segment_to_mid[item["segment_id"]] for item in ordered_segments)
            sequences[str(record["trajectory_id"])] = compress_repeated_mid_ids(mid_ids)
        except KeyError as exc:
            raise ValueError(f"segment has no Mid cluster: {exc.args[0]}") from exc
    return sequences


def mine_high_paths(
    records: Sequence[Mapping],
    clusters: Iterable[MidCluster],
    *,
    max_path_length: int = 4,
    min_support_ratio: float = 0.02,
    epsilon: float = 1e-6,
    success_threshold: float = 0.999,
) -> tuple[HighPath, ...]:
    if max_path_length < 2 or not 0 <= min_support_ratio <= 1 or epsilon <= 0:
        raise ValueError("invalid High path mining configuration")
    sequences = trajectory_mid_sequences(records, clusters)
    records_by_id = {str(record["trajectory_id"]): record for record in records}
    sample_by_trajectory = {
        trajectory_id: str(record.get("sample_id", trajectory_id))
        for trajectory_id, record in records_by_id.items()
    }
    positive_ids = {
        trajectory_id
        for trajectory_id, record in records_by_id.items()
        if float(record["primary_score"]) >= success_threshold
    }
    negative_ids = set(records_by_id) - positive_ids
    positive_samples = {
        sample_by_trajectory[trajectory_id]
        for trajectory_id in positive_ids
    }
    negative_samples = {
        sample_by_trajectory[trajectory_id]
        for trajectory_id in negative_ids
    }
    supporting_ids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for trajectory_id, sequence in sequences.items():
        paths_in_trajectory = set()
        for length in range(2, min(max_path_length, len(sequence)) + 1):
            for start in range(len(sequence) - length + 1):
                path = sequence[start : start + length]
                if len(set(path)) >= 2:
                    paths_in_trajectory.add(path)
        for path in paths_in_trajectory:
            supporting_ids[path].add(trajectory_id)

    paths = []
    for ordered_mid_ids, trajectory_ids in supporting_ids.items():
        supporting_positive_samples = {
            sample_by_trajectory[trajectory_id] for trajectory_id in trajectory_ids
            if trajectory_id in positive_ids
        }
        supporting_negative_samples = {
            sample_by_trajectory[trajectory_id] for trajectory_id in trajectory_ids
            if trajectory_id in negative_ids
        }
        positive_count = len(supporting_positive_samples)
        negative_count = len(supporting_negative_samples)
        positive_support = (
            positive_count / len(positive_samples) if positive_samples else 0.0
        )
        negative_support = (
            negative_count / len(negative_samples) if negative_samples else 0.0
        )
        if positive_support < min_support_ratio:
            continue
        contrastive_score = positive_support * math.log(
            (positive_support + epsilon) / (negative_support + epsilon)
        )
        if contrastive_score <= 0:
            continue
        path_key = "\x1f".join(ordered_mid_ids).encode("utf-8")
        paths.append(
            HighPath(
                path_id=f"high_{hashlib.sha256(path_key).hexdigest()[:12]}",
                ordered_mid_ids=ordered_mid_ids,
                positive_support=positive_support,
                negative_support=negative_support,
                contrastive_score=contrastive_score,
                supporting_trajectory_ids=tuple(sorted(trajectory_ids)),
            )
        )
    return tuple(
        sorted(
            paths,
            key=lambda item: (
                -item.contrastive_score,
                len(item.ordered_mid_ids),
                len(item.ordered_mid_ids) - len(set(item.ordered_mid_ids)),
                item.path_id,
            ),
        )
    )
