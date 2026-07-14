from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.models import SegmentInstance, WebShopEventType


@dataclass(frozen=True, slots=True)
class GraphComponents:
    segment_ids: tuple[str, ...]
    event_types: tuple[WebShopEventType | None, ...]
    segment_embeddings: np.ndarray
    rho: np.ndarray
    semantic: sparse.csr_matrix
    transition: sparse.csr_matrix
    outcome: sparse.csr_matrix
    base: sparse.csr_matrix
    adjacency: sparse.csr_matrix
    laplacian: sparse.csr_matrix
    neighbor_count: int
    edge_count: int


def build_graph(
    trajectory_segments: Sequence[Sequence[SegmentInstance]],
    config: Trace2TowerConfig,
) -> GraphComponents:
    if config.semantic_only:
        raise ValueError("semantic-only clustering skips graph construction")
    groups = ordered_segment_groups(trajectory_segments)
    segments = tuple(segment for group in groups for segment in group)
    if not segments:
        raise ValueError("graph construction requires segments")
    if any(segment.event_type is not None for segment in segments):
        config.validate_for_benchmark(Benchmark.WEBSHOP)
    dimension = len(segments[0].embedding)
    if dimension == 0 or any(len(segment.embedding) != dimension for segment in segments):
        raise ValueError("all segment embeddings must have one fixed nonzero dimension")

    embeddings = np.asarray([segment.embedding for segment in segments], dtype=np.float64)
    normalized_embeddings = _normalize_rows(embeddings)
    contexts = np.zeros((len(segments), dimension * 2), dtype=np.float64)
    offset = 0
    for group in groups:
        for local_index in range(len(group)):
            if local_index:
                contexts[offset + local_index, :dimension] = normalized_embeddings[
                    offset + local_index - 1
                ]
            if local_index + 1 < len(group):
                contexts[offset + local_index, dimension:] = normalized_embeddings[
                    offset + local_index + 1
                ]
        offset += len(group)
    normalized_contexts = _normalize_rows(contexts)

    requested_neighbors = min(30, max(10, math.ceil(math.log2(max(len(segments), 2)))))
    neighbor_count = min(requested_neighbors, len(segments) - 1)
    if config.event_type_stratification:
        event_types = tuple(segment.event_type for segment in segments)
        semantic_neighbors = _nearest_neighbors_by_group(
            normalized_embeddings, event_types, neighbor_count
        )
        transition_neighbors = _nearest_neighbors_by_group(
            normalized_contexts, event_types, neighbor_count
        )
    else:
        semantic_neighbors = _nearest_neighbors(normalized_embeddings, neighbor_count)
        transition_neighbors = _nearest_neighbors(normalized_contexts, neighbor_count)
    candidate_edges = {
        tuple(sorted((source, target)))
        for neighbors in (semantic_neighbors, transition_neighbors)
        for source, targets in enumerate(neighbors)
        for target in targets
        if source != target
        and (
            not config.event_type_stratification
            or segments[source].event_type is None
            or segments[target].event_type is None
            or segments[source].event_type == segments[target].event_type
        )
    }

    scores = np.clip(
        np.asarray([segment.trajectory_score for segment in segments], dtype=np.float64),
        0,
        1,
    )
    rho = np.empty(len(segments), dtype=np.float64)
    for source, targets in enumerate(semantic_neighbors):
        weights = np.maximum(
            0,
            normalized_embeddings[targets] @ normalized_embeddings[source],
        )
        rho[source] = (
            scores[source] + float(weights @ scores[targets])
        ) / (1 + float(weights.sum()))
    rho = np.clip(rho, 0, 1)

    rows = []
    columns = []
    semantic_values = []
    transition_values = []
    outcome_values = []
    base_values = []
    adjacency_values = []
    enabled_count = 1 + config.use_transition_edge + config.use_outcome_edge
    for left, right in sorted(candidate_edges):
        semantic_value = max(
            0.0,
            float(normalized_embeddings[left] @ normalized_embeddings[right]),
        )
        transition_value = max(
            0.0,
            float(normalized_contexts[left] @ normalized_contexts[right]),
        )
        outcome_value = min(1.0, max(0.0, 1 - abs(rho[left] - rho[right])))
        base_value = (
            semantic_value
            + (transition_value if config.use_transition_edge else 0)
            + (outcome_value if config.use_outcome_edge else 0)
        ) / enabled_count
        if config.use_contrastive_decomposition:
            positive = math.sqrt(rho[left] * rho[right])
            negative = math.sqrt((1 - rho[left]) * (1 - rho[right]))
            adjacency_value = base_value * (
                positive - config.failure_penalty * negative
            )
        else:
            adjacency_value = base_value

        rows.extend((left, right))
        columns.extend((right, left))
        semantic_values.extend((semantic_value, semantic_value))
        transition_values.extend((transition_value, transition_value))
        outcome_values.extend((outcome_value, outcome_value))
        base_values.extend((base_value, base_value))
        adjacency_values.extend((adjacency_value, adjacency_value))

    shape = (len(segments), len(segments))
    matrices = [
        sparse.csr_matrix((values, (rows, columns)), shape=shape)
        for values in (
            semantic_values,
            transition_values,
            outcome_values,
            base_values,
            adjacency_values,
        )
    ]
    for matrix in matrices:
        matrix.eliminate_zeros()
    semantic, transition, outcome, base, adjacency = matrices
    adjacency = ((adjacency + adjacency.T) * 0.5).tocsr()
    adjacency.eliminate_zeros()
    absolute_degree = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    inverse_sqrt_degree = np.zeros_like(absolute_degree)
    nonzero = absolute_degree > 0
    inverse_sqrt_degree[nonzero] = 1 / np.sqrt(absolute_degree[nonzero])
    scaling = sparse.diags(inverse_sqrt_degree)
    laplacian = (
        sparse.eye(len(segments), format="csr")
        - scaling @ adjacency @ scaling
    ).tocsr()
    return GraphComponents(
        segment_ids=tuple(segment.segment_id for segment in segments),
        event_types=tuple(segment.event_type for segment in segments),
        segment_embeddings=embeddings,
        rho=rho,
        semantic=semantic,
        transition=transition,
        outcome=outcome,
        base=base,
        adjacency=adjacency,
        laplacian=laplacian,
        neighbor_count=neighbor_count,
        edge_count=len(candidate_edges),
    )


def ordered_segment_groups(
    trajectory_segments: Sequence[Sequence[SegmentInstance]],
) -> tuple[tuple[SegmentInstance, ...], ...]:
    groups = tuple(
        tuple(sorted(group, key=lambda segment: segment.start_step))
        for group in trajectory_segments
        if group
    )
    groups = tuple(sorted(groups, key=lambda group: group[0].trajectory_id))
    if any(
        segment.trajectory_id != group[0].trajectory_id
        for group in groups
        for segment in group
    ):
        raise ValueError("each segment group must contain one trajectory")
    return groups


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _nearest_neighbors(values: np.ndarray, neighbor_count: int) -> tuple[np.ndarray, ...]:
    if neighbor_count == 0:
        return tuple(np.empty(0, dtype=np.int64) for _ in range(len(values)))
    model = NearestNeighbors(
        n_neighbors=min(neighbor_count + 1, len(values)),
        metric="cosine",
        algorithm="brute",
    ).fit(values)
    raw_neighbors = model.kneighbors(values, return_distance=False)
    return tuple(
        np.asarray([target for target in targets if target != source][:neighbor_count])
        for source, targets in enumerate(raw_neighbors)
    )


def _nearest_neighbors_by_group(
    values: np.ndarray,
    group_ids: Sequence[object],
    neighbor_count: int,
) -> tuple[np.ndarray, ...]:
    if len(values) != len(group_ids):
        raise ValueError("neighbor values and group IDs must align")
    groups: dict[object, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    neighbors = [np.empty(0, dtype=np.int64) for _ in group_ids]
    for indices in groups.values():
        local_neighbors = _nearest_neighbors(
            values[indices], min(neighbor_count, len(indices) - 1)
        )
        for local_source, local_targets in enumerate(local_neighbors):
            neighbors[indices[local_source]] = np.asarray(
                [indices[target] for target in local_targets], dtype=np.int64
            )
    return tuple(neighbors)
