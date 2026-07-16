from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.models import (
    EventType,
    GraphOutcomeMode,
    SegmentInstance,
    SemanticNeighborScope,
)


@dataclass(frozen=True, slots=True)
class GraphComponents:
    segment_ids: tuple[str, ...]
    event_types: tuple[EventType | None, ...]
    segment_embeddings: np.ndarray
    rho: np.ndarray
    semantic: sparse.csr_matrix
    transition: sparse.csr_matrix
    outcome: sparse.csr_matrix
    base: sparse.csr_matrix
    positive: sparse.csr_matrix
    negative: sparse.csr_matrix
    adjacency: sparse.csr_matrix
    laplacian: sparse.csr_matrix
    neighbor_count: int
    edge_count: int
    transition_edge_count: int
    cross_event_edge_count: int
    node_member_segment_ids: tuple[tuple[str, ...], ...] = ()


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
    if any(segment.event_type is None for segment in segments):
        raise ValueError("transition-aware graph requires domain event labels")
    dimension = len(segments[0].embedding)
    if dimension == 0 or any(len(segment.embedding) != dimension for segment in segments):
        raise ValueError("all segment embeddings must have one fixed nonzero dimension")

    node_groups = _node_groups(segments, config.collapse_duplicate_embeddings)
    node_segments = tuple(group[0] for group in node_groups)
    segment_to_node = {
        segment.segment_id: node_index
        for node_index, group in enumerate(node_groups)
        for segment in group
    }
    embeddings = np.asarray(
        [group[0].embedding for group in node_groups],
        dtype=np.float64,
    )
    normalized_embeddings = _normalize_rows(embeddings)
    requested_neighbors = min(30, max(10, math.ceil(math.log2(max(len(node_groups), 2)))))
    neighbor_count = min(requested_neighbors, len(node_groups) - 1)
    semantic_neighbors = (
        _nearest_neighbors_by_event(
            normalized_embeddings,
            tuple(segment.event_type for segment in node_segments),
            neighbor_count,
        )
        if config.semantic_neighbor_scope is SemanticNeighborScope.SAME_EVENT
        else _nearest_neighbors(normalized_embeddings, neighbor_count)
    )
    candidate_edges = {
        tuple(sorted((source, target)))
        for source, targets in enumerate(semantic_neighbors)
        for target in targets
        if source != target
    }

    observed_transitions = set()
    transition_counts = Counter()
    source_counts = Counter()
    for group in groups:
        for source, target in zip(group, group[1:]):
            source_index = segment_to_node[source.segment_id]
            target_index = segment_to_node[target.segment_id]
            source_key = (
                source_index
                if config.collapse_duplicate_embeddings
                else _transition_key(source)
            )
            target_key = (
                target_index
                if config.collapse_duplicate_embeddings
                else _transition_key(target)
            )
            transition_counts[(source_key, target_key)] += 1
            source_counts[source_key] += 1
            if source_index == target_index:
                continue
            observed_transitions.add((source_index, target_index))
            candidate_edges.add(tuple(sorted((source_index, target_index))))

    if config.graph_outcome_mode is not GraphOutcomeMode.BINARY_CONTRASTIVE:
        node_outcomes = np.asarray(
            [
                np.mean(
                    [min(1.0, max(0.0, segment.trajectory_score)) for segment in group]
                )
                for group in node_groups
            ],
            dtype=np.float64,
        )
    else:
        node_outcomes = np.asarray(
            [
                np.mean(
                    [
                        float(segment.trajectory_score >= config.success_threshold)
                        for segment in group
                    ]
                )
                for group in node_groups
            ],
            dtype=np.float64,
        )
    if config.collapse_duplicate_embeddings:
        rho = node_outcomes
    else:
        rho = np.empty(len(node_groups), dtype=np.float64)
        for source, targets in enumerate(semantic_neighbors):
            weights = np.maximum(
                0,
                normalized_embeddings[targets] @ normalized_embeddings[source],
            )
            rho[source] = (
                node_outcomes[source] + float(weights @ node_outcomes[targets])
            ) / (1 + float(weights.sum()))
        rho = np.clip(rho, 0, 1)

    rows = []
    columns = []
    semantic_values = []
    transition_values = []
    outcome_values = []
    base_values = []
    positive_values = []
    negative_values = []
    adjacency_values = []
    local_outcome_calibration = (
        _local_edge_outcome_calibration(candidate_edges, rho)
        if config.graph_outcome_mode is GraphOutcomeMode.CONTINUOUS_RESIDUAL
        else {}
    )
    enabled_count = 1 + config.use_transition_edge + config.use_outcome_edge
    for left, right in sorted(candidate_edges):
        semantic_value = max(
            0.0,
            float(normalized_embeddings[left] @ normalized_embeddings[right]),
        )
        transition_value = 0.0
        for source, target in ((left, right), (right, left)):
            if (source, target) not in observed_transitions:
                continue
            source_key = (
                source
                if config.collapse_duplicate_embeddings
                else _transition_key(node_segments[source])
            )
            target_key = (
                target
                if config.collapse_duplicate_embeddings
                else _transition_key(node_segments[target])
            )
            transition_value = max(
                transition_value,
                transition_counts[(source_key, target_key)] / source_counts[source_key],
            )
        outcome_value = min(1.0, max(0.0, 1 - abs(rho[left] - rho[right])))
        base_value = (
            semantic_value
            + (transition_value if config.use_transition_edge else 0)
            + (outcome_value if config.use_outcome_edge else 0)
        ) / enabled_count
        if config.use_contrastive_decomposition:
            if config.graph_outcome_mode is GraphOutcomeMode.CONTINUOUS_RESIDUAL:
                correction = (
                    base_value
                    * config.continuous_residual_weight
                    * local_outcome_calibration[(left, right)]
                )
                positive_value = max(0.0, correction)
                negative_value = max(0.0, -correction)
                adjacency_value = base_value + correction
            else:
                positive_value = base_value * math.sqrt(rho[left] * rho[right])
                negative_value = base_value * math.sqrt(
                    (1 - rho[left]) * (1 - rho[right])
                )
                adjacency_value = positive_value - config.failure_penalty * negative_value
        else:
            positive_value = base_value
            negative_value = 0.0
            adjacency_value = base_value

        rows.extend((left, right))
        columns.extend((right, left))
        semantic_values.extend((semantic_value, semantic_value))
        transition_values.extend((transition_value, transition_value))
        outcome_values.extend((outcome_value, outcome_value))
        base_values.extend((base_value, base_value))
        positive_values.extend((positive_value, positive_value))
        negative_values.extend((negative_value, negative_value))
        adjacency_values.extend((adjacency_value, adjacency_value))

    shape = (len(node_groups), len(node_groups))
    matrices = [
        sparse.csr_matrix((values, (rows, columns)), shape=shape)
        for values in (
            semantic_values,
            transition_values,
            outcome_values,
            base_values,
            positive_values,
            negative_values,
            adjacency_values,
        )
    ]
    for matrix in matrices:
        matrix.eliminate_zeros()
    semantic, transition, outcome, base, positive, negative, adjacency = matrices
    adjacency = ((adjacency + adjacency.T) * 0.5).tocsr()
    adjacency.eliminate_zeros()
    absolute_degree = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    inverse_sqrt_degree = np.zeros_like(absolute_degree)
    nonzero = absolute_degree > 0
    inverse_sqrt_degree[nonzero] = 1 / np.sqrt(absolute_degree[nonzero])
    scaling = sparse.diags(inverse_sqrt_degree)
    laplacian = (
        sparse.eye(len(node_groups), format="csr") - scaling @ adjacency @ scaling
    ).tocsr()
    return GraphComponents(
        segment_ids=tuple(segment.segment_id for segment in node_segments),
        event_types=tuple(segment.event_type for segment in node_segments),
        segment_embeddings=embeddings,
        rho=rho,
        semantic=semantic,
        transition=transition,
        outcome=outcome,
        base=base,
        positive=positive,
        negative=negative,
        adjacency=adjacency,
        laplacian=laplacian,
        neighbor_count=neighbor_count,
        edge_count=len(candidate_edges),
        transition_edge_count=len({tuple(sorted(edge)) for edge in observed_transitions}),
        cross_event_edge_count=sum(
            node_segments[left].event_type != node_segments[right].event_type
            for left, right in candidate_edges
        ),
        node_member_segment_ids=tuple(
            tuple(segment.segment_id for segment in group) for group in node_groups
        ),
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
        segment.trajectory_id != group[0].trajectory_id for group in groups for segment in group
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
        # 输入已单位归一化，两种距离给出相同近邻顺序；欧氏距离避免
        # sklearn 为每批查询复制完整参考矩阵再次归一化。
        metric="euclidean",
        algorithm="brute",
    ).fit(values)
    neighbors = []
    query_batch_size = 128
    for start in range(0, len(values), query_batch_size):
        batch = values[start : start + query_batch_size]
        raw_neighbors = model.kneighbors(batch, return_distance=False)
        neighbors.extend(
            np.asarray([target for target in targets if target != start + offset][:neighbor_count])
            for offset, targets in enumerate(raw_neighbors)
        )
    return tuple(neighbors)


def _nearest_neighbors_by_event(
    values: np.ndarray,
    event_types: tuple[EventType | None, ...],
    neighbor_count: int,
) -> tuple[np.ndarray, ...]:
    neighbors = [np.empty(0, dtype=np.int64) for _ in range(len(values))]
    event_indices: dict[EventType | None, list[int]] = {}
    for index, event_type in enumerate(event_types):
        event_indices.setdefault(event_type, []).append(index)

    for indices in event_indices.values():
        if len(indices) < 2:
            continue
        local_neighbors = _nearest_neighbors(
            values[indices],
            min(neighbor_count, len(indices) - 1),
        )
        for local_source, local_targets in enumerate(local_neighbors):
            neighbors[indices[local_source]] = np.asarray(
                [indices[int(target)] for target in local_targets],
                dtype=np.int64,
            )
    return tuple(neighbors)


def _transition_key(segment: SegmentInstance) -> str:
    if segment.event_type is None:
        raise ValueError("transition-aware graph requires domain event labels")
    return segment.event_type.value


def _local_edge_outcome_calibration(
    candidate_edges: set[tuple[int, int]],
    outcomes: np.ndarray,
) -> dict[tuple[int, int], float]:
    affinities = {
        edge: math.sqrt(outcomes[edge[0]] * outcomes[edge[1]])
        for edge in candidate_edges
    }
    incident: dict[int, list[float]] = {}
    for (left, right), affinity in affinities.items():
        incident.setdefault(left, []).append(affinity)
        incident.setdefault(right, []).append(affinity)
    local_moments = {
        node: (float(np.mean(values)), float(np.std(values)))
        for node, values in incident.items()
    }

    calibrated = {}
    for edge, affinity in affinities.items():
        endpoint_scores = []
        for node in edge:
            mean, deviation = local_moments[node]
            endpoint_scores.append(
                0.0 if deviation <= 1e-12 else (affinity - mean) / deviation
            )
        calibrated[edge] = math.tanh(sum(endpoint_scores) / 2)
    return calibrated


def _node_groups(
    segments: tuple[SegmentInstance, ...], collapse: bool
) -> tuple[tuple[SegmentInstance, ...], ...]:
    if not collapse:
        return tuple((segment,) for segment in segments)
    grouped: dict[tuple[EventType | None, bytes], list[SegmentInstance]] = {}
    for segment in segments:
        key = (
            segment.event_type,
            np.asarray(segment.embedding, dtype=np.float32).tobytes(),
        )
        grouped.setdefault(key, []).append(segment)
    return tuple(
        tuple(grouped[key])
        for key in sorted(
            grouped,
            key=lambda item: min(segment.segment_id for segment in grouped[item]),
        )
    )
