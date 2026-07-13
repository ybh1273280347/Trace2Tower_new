from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.graph import GraphComponents
from trace2tower.methods.trace2tower.models import MidCluster


@dataclass(frozen=True, slots=True)
class ClusteringResult:
    cluster_count: int
    labels: tuple[int, ...]
    eigenvalues: tuple[float, ...]
    representation: np.ndarray
    clusters: tuple[MidCluster, ...]


def spectral_clustering(
    graph: GraphComponents,
    config: Trace2TowerConfig,
) -> ClusteringResult:
    node_count = len(graph.segment_ids)
    if node_count == 1:
        return cluster_representation(
            graph.segment_ids,
            graph.segment_embeddings,
            np.ones((1, 1)),
            1,
            config.random_state,
            (float(graph.laplacian[0, 0]),),
        )

    requested = min(node_count, config.max_mid_clusters + 2)
    if requested == node_count:
        eigenvalues, eigenvectors = np.linalg.eigh(graph.laplacian.toarray())
    else:
        generator = np.random.default_rng(config.random_state)
        eigenvalues, eigenvectors = eigsh(
            graph.laplacian,
            k=requested,
            which="SM",
            v0=generator.standard_normal(node_count),
            tol=1e-8,
        )
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    valid_columns = np.isfinite(eigenvectors).all(axis=0) & (
        np.std(eigenvectors, axis=0) > 1e-10
    )
    valid_eigenvalues = eigenvalues[valid_columns]
    valid_eigenvectors = eigenvectors[:, valid_columns]
    if valid_eigenvectors.shape[1] == 0:
        valid_eigenvalues = np.asarray((0.0,))
        valid_eigenvectors = np.ones((node_count, 1), dtype=np.float64)

    maximum = min(
        config.max_mid_clusters,
        node_count,
        max(1, len(valid_eigenvalues) - 1),
    )
    group_ids = ()
    if config.event_type_stratification:
        if any(event_type is None for event_type in graph.event_types):
            raise ValueError("event-type stratification requires typed segments")
        group_ids = graph.event_types
        group_count = len(set(group_ids))
        if group_count > maximum:
            raise ValueError("Mid cluster limit cannot cover every event type")
        minimum = max(config.min_mid_clusters, group_count)
    else:
        minimum = min(config.min_mid_clusters, maximum)
    if minimum == maximum:
        cluster_count = maximum
    else:
        candidates = range(minimum, maximum + 1)
        cluster_count = max(
            candidates,
            key=lambda count: (
                valid_eigenvalues[count] - valid_eigenvalues[count - 1],
                -count,
            ),
        )
    representation = valid_eigenvectors[:, :cluster_count]
    representation = _normalize_rows(representation)
    return cluster_representation(
        graph.segment_ids,
        graph.segment_embeddings,
        representation,
        cluster_count,
        config.random_state,
        tuple(float(value) for value in valid_eigenvalues),
        group_ids=group_ids,
    )


def semantic_only_clustering(
    segment_ids: tuple[str, ...],
    segment_embeddings: np.ndarray,
    *,
    cluster_count: int,
    random_state: int,
) -> ClusteringResult:
    if not 1 <= cluster_count <= len(segment_ids):
        raise ValueError("cluster count must fit the segment set")
    return cluster_representation(
        segment_ids,
        segment_embeddings,
        segment_embeddings,
        cluster_count,
        random_state,
        (),
    )


def cluster_representation(
    segment_ids: tuple[str, ...],
    segment_embeddings: np.ndarray,
    representation: np.ndarray,
    cluster_count: int,
    random_state: int,
    eigenvalues: tuple[float, ...],
    *,
    group_ids: tuple[object, ...] = (),
) -> ClusteringResult:
    if group_ids:
        labels = _stratified_kmeans(
            representation,
            group_ids,
            cluster_count,
            random_state,
        )
    elif cluster_count == 1:
        labels = np.zeros(len(segment_ids), dtype=np.int64)
    else:
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            n_init=20,
            max_iter=300,
        ).fit_predict(representation)

    members = {
        label: tuple(
            index for index, current_label in enumerate(labels) if current_label == label
        )
        for label in range(cluster_count)
    }
    ordered_labels = sorted(
        members,
        key=lambda label: min(segment_ids[index] for index in members[label]),
    )
    canonical = {label: index for index, label in enumerate(ordered_labels)}
    canonical_labels = tuple(canonical[int(label)] for label in labels)
    clusters = []
    for cluster_index in range(cluster_count):
        indices = [
            index for index, label in enumerate(canonical_labels) if label == cluster_index
        ]
        centroid = tuple(np.mean(segment_embeddings[indices], axis=0).tolist())
        clusters.append(
            MidCluster(
                cluster_id=f"mid_{cluster_index:04d}",
                member_segment_ids=tuple(sorted(segment_ids[index] for index in indices)),
                centroid=centroid,
            )
        )
    return ClusteringResult(
        cluster_count=cluster_count,
        labels=canonical_labels,
        eigenvalues=eigenvalues,
        representation=representation,
        clusters=tuple(clusters),
    )


def _stratified_kmeans(
    representation: np.ndarray,
    group_ids: tuple[object, ...],
    cluster_count: int,
    random_state: int,
) -> np.ndarray:
    if len(group_ids) != len(representation):
        raise ValueError("stratification groups and representation must align")
    groups = {
        group_id: np.asarray(
            [index for index, current in enumerate(group_ids) if current == group_id],
            dtype=np.int64,
        )
        for group_id in sorted(set(group_ids), key=str)
    }
    if not len(groups) <= cluster_count <= len(representation):
        raise ValueError("cluster count must cover stratification groups")
    allocations = {group_id: 1 for group_id in groups}
    for _ in range(cluster_count - len(groups)):
        eligible = [
            group_id
            for group_id, indices in groups.items()
            if allocations[group_id] < len(indices)
        ]
        selected = min(
            eligible,
            key=lambda group_id: (
                -len(groups[group_id]) / allocations[group_id],
                str(group_id),
            ),
        )
        allocations[selected] += 1

    labels = np.empty(len(representation), dtype=np.int64)
    next_label = 0
    for group_id, indices in groups.items():
        count = allocations[group_id]
        if count == 1:
            local_labels = np.zeros(len(indices), dtype=np.int64)
        else:
            local_labels = KMeans(
                n_clusters=count,
                random_state=random_state,
                n_init=20,
                max_iter=300,
            ).fit_predict(representation[indices])
        labels[indices] = local_labels + next_label
        next_label += count
    return labels


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
