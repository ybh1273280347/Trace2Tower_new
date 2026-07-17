from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.graph import GraphComponents
from trace2tower.methods.trace2tower.models import EventType, MidCluster


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
            node_member_segment_ids=_node_members(graph),
        )

    requested = min(
        node_count,
        config.max_mid_clusters + 2
        if config.max_mid_clusters is not None
        else max(34, config.min_mid_clusters + 2),
    )
    previous_cluster_count = None
    while True:
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
            config.max_mid_clusters
            if config.max_mid_clusters is not None
            else node_count,
            node_count,
            max(1, len(valid_eigenvalues) - 1),
        )
        minimum = min(config.min_mid_clusters, maximum)
        if minimum == maximum:
            cluster_count = maximum
        else:
            cluster_count = max(
                range(minimum, maximum + 1),
                key=lambda count: (
                    valid_eigenvalues[count] - valid_eigenvalues[count - 1],
                    -count,
                ),
            )
        if config.max_mid_clusters is not None or requested == node_count:
            break
        if cluster_count == previous_cluster_count and cluster_count < maximum:
            break
        previous_cluster_count = cluster_count
        requested = min(node_count, requested * 2)
    representation = valid_eigenvectors[:, :cluster_count]
    representation = _normalize_rows(representation)
    return cluster_representation(
        graph.segment_ids,
        graph.segment_embeddings,
        representation,
        cluster_count,
        config.random_state,
        tuple(float(value) for value in valid_eigenvalues),
        node_member_segment_ids=_node_members(graph),
    )


def semantic_only_clustering(
    segment_ids: tuple[str, ...],
    segment_embeddings: np.ndarray,
    *,
    cluster_count: int,
    random_state: int,
    node_member_segment_ids: tuple[tuple[str, ...], ...] | None = None,
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
        node_member_segment_ids=node_member_segment_ids,
    )


def separate_exclusive_event_clusters(
    clustering: ClusteringResult,
    graph: GraphComponents,
    exclusive_events: frozenset[EventType],
) -> ClusteringResult:
    if not exclusive_events:
        return clustering

    group_labels = {}
    labels = []
    for spectral_label, event in zip(clustering.labels, graph.event_types, strict=True):
        # 排他目标 operator 必须拥有独立正文；其余节点继续保留谱图学到的角色。
        group = (
            ("event", event.value)
            if event in exclusive_events
            else ("spectral", str(spectral_label))
        )
        if group not in group_labels:
            group_labels[group] = len(group_labels)
        labels.append(group_labels[group])

    return _clustering_result(
        graph.segment_ids,
        graph.segment_embeddings,
        labels,
        clustering.eigenvalues,
        clustering.representation,
        node_member_segment_ids=_node_members(graph),
    )


def cluster_representation(
    segment_ids: tuple[str, ...],
    segment_embeddings: np.ndarray,
    representation: np.ndarray,
    cluster_count: int,
    random_state: int,
    eigenvalues: tuple[float, ...],
    node_member_segment_ids: tuple[tuple[str, ...], ...] | None = None,
) -> ClusteringResult:
    if cluster_count == 1:
        labels = np.zeros(len(segment_ids), dtype=np.int64)
    else:
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=random_state,
            n_init=20,
            max_iter=300,
        ).fit_predict(representation)

    return _clustering_result(
        segment_ids,
        segment_embeddings,
        labels,
        eigenvalues,
        representation,
        node_member_segment_ids=node_member_segment_ids,
    )


def _clustering_result(
    segment_ids: tuple[str, ...],
    segment_embeddings: np.ndarray,
    labels,
    eigenvalues: tuple[float, ...],
    representation: np.ndarray,
    node_member_segment_ids: tuple[tuple[str, ...], ...] | None = None,
) -> ClusteringResult:
    node_members = node_member_segment_ids or tuple(
        (segment_id,) for segment_id in segment_ids
    )
    if len(node_members) != len(segment_ids):
        raise ValueError("spectral node membership must align with graph nodes")
    unique_labels = sorted(set(int(label) for label in labels))
    members = {
        label: tuple(index for index, current_label in enumerate(labels) if current_label == label)
        for label in unique_labels
    }
    ordered_labels = sorted(
        members,
        key=lambda label: min(segment_ids[index] for index in members[label]),
    )
    canonical = {label: index for index, label in enumerate(ordered_labels)}
    canonical_labels = tuple(canonical[int(label)] for label in labels)
    clusters = []
    for cluster_index in range(len(ordered_labels)):
        indices = [index for index, label in enumerate(canonical_labels) if label == cluster_index]
        weights = np.asarray([len(node_members[index]) for index in indices], dtype=np.float64)
        centroid = tuple(
            np.average(segment_embeddings[indices], axis=0, weights=weights).tolist()
        )
        clusters.append(
            MidCluster(
                cluster_id=f"mid_{cluster_index:04d}",
                member_segment_ids=tuple(
                    sorted(
                        segment_id
                        for index in indices
                        for segment_id in node_members[index]
                    )
                ),
                centroid=centroid,
            )
        )
    return ClusteringResult(
        cluster_count=len(ordered_labels),
        labels=canonical_labels,
        eigenvalues=eigenvalues,
        representation=representation,
        clusters=tuple(clusters),
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _node_members(graph: GraphComponents) -> tuple[tuple[str, ...], ...]:
    return graph.node_member_segment_ids or tuple(
        (segment_id,) for segment_id in graph.segment_ids
    )
