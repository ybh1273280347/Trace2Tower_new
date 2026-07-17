from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trace2tower.methods.trace2tower.high_paths import trajectory_mid_sequences
from trace2tower.methods.trace2tower.models import HighCommunity, HighPath, MidCluster


@dataclass(frozen=True, slots=True)
class HighCommunityDiscovery:
    communities: tuple[HighCommunity, ...]
    trajectory_ids: tuple[str, ...]
    labels: tuple[int, ...]
    feature_count: int
    graph_weight: float
    modularity: float


def discover_high_communities(
    records: Sequence[Mapping],
    clusters: Iterable[MidCluster],
    paths: Iterable[HighPath],
    *,
    success_threshold: float,
) -> HighCommunityDiscovery:
    sequences = trajectory_mid_sequences(records, clusters)
    successful_ids = tuple(
        sorted(
            str(record["trajectory_id"])
            for record in records
            if float(record["primary_score"]) >= success_threshold
        )
    )
    if not successful_ids:
        raise ValueError("High community discovery requires successful trajectories")
    feature_rows = tuple(_sequence_features(sequences[trajectory_id]) for trajectory_id in successful_ids)
    vocabulary = tuple(sorted({feature for row in feature_rows for feature in row}))
    feature_indices = {feature: index for index, feature in enumerate(vocabulary)}
    document_frequency = Counter(feature for row in feature_rows for feature in row)
    matrix = np.zeros((len(feature_rows), len(vocabulary)), dtype=np.float64)
    for row_index, features in enumerate(feature_rows):
        counts = Counter(features)
        for feature, count in counts.items():
            # 跨全部成功轨迹都出现的流程骨架不应支配 High 社区边界。
            inverse_frequency = math.log(len(feature_rows) / document_frequency[feature])
            matrix[row_index, feature_indices[feature]] = count * inverse_frequency
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    adjacency = normalized @ normalized.T
    np.fill_diagonal(adjacency, 0.0)
    adjacency[adjacency < 1e-12] = 0.0
    groups, modularity = _leading_eigenvector_groups(adjacency)
    paths = tuple(paths)
    communities = []
    labels = np.empty(len(successful_ids), dtype=np.int64)
    for label, indices in enumerate(
        sorted(groups, key=lambda group: min(successful_ids[index] for index in group))
    ):
        member_trajectory_ids = tuple(successful_ids[index] for index in indices)
        member_set = set(member_trajectory_ids)
        member_mid_ids = tuple(
            sorted(
                {
                    mid_id
                    for trajectory_id in member_trajectory_ids
                    for mid_id in sequences[trajectory_id]
                }
            )
        )
        member_path_ids = tuple(
            sorted(
                path.path_id
                for path in paths
                if member_set.intersection(path.supporting_trajectory_ids)
            )
        )
        digest = hashlib.sha256("\x1f".join(member_trajectory_ids).encode()).hexdigest()[:12]
        communities.append(
            HighCommunity(
                community_id=f"high_community_{digest}",
                member_mid_ids=member_mid_ids,
                member_path_ids=member_path_ids,
                supporting_trajectory_ids=member_trajectory_ids,
            )
        )
        labels[list(indices)] = label
    return HighCommunityDiscovery(
        communities=tuple(communities),
        trajectory_ids=successful_ids,
        labels=tuple(int(value) for value in labels),
        feature_count=len(vocabulary),
        graph_weight=float(adjacency.sum() / 2),
        modularity=modularity,
    )


def _sequence_features(sequence: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        [f"mid:{mid_id}" for mid_id in sequence]
        + [f"transition:{source}->{target}" for source, target in zip(sequence, sequence[1:])]
    )


def _leading_eigenvector_groups(adjacency: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], float]:
    node_count = len(adjacency)
    total_degree = float(adjacency.sum())
    if node_count == 1 or total_degree <= 1e-12:
        return (tuple(range(node_count)),), 0.0
    degrees = adjacency.sum(axis=1)
    modularity_matrix = adjacency - np.outer(degrees, degrees) / total_degree

    def split(indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        if len(indices) < 2:
            return (indices,)
        local = modularity_matrix[np.ix_(indices, indices)]
        local = local - np.diag(local.sum(axis=1))
        eigenvalues, eigenvectors = np.linalg.eigh((local + local.T) * 0.5)
        if eigenvalues[-1] <= 1e-10:
            return (indices,)
        signs = np.where(eigenvectors[:, -1] >= 0, 1.0, -1.0)
        left = tuple(index for index, sign in zip(indices, signs, strict=True) if sign > 0)
        right = tuple(index for index, sign in zip(indices, signs, strict=True) if sign < 0)
        gain = float(signs @ local @ signs / (2 * total_degree))
        if not left or not right or gain <= 1e-10:
            return (indices,)
        return split(left) + split(right)

    groups = split(tuple(range(node_count)))
    membership = np.empty(node_count, dtype=np.int64)
    for label, group in enumerate(groups):
        membership[list(group)] = label
    modularity = sum(
        modularity_matrix[left, right]
        for left in range(node_count)
        for right in range(node_count)
        if membership[left] == membership[right]
    ) / total_degree
    return groups, float(modularity)
