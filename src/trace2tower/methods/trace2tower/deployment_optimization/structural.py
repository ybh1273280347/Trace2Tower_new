from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trace2tower.methods.trace2tower.core.models import MidCluster
from trace2tower.methods.trace2tower.deployment_optimization.models import (
    LineageComponent,
    LineageKind,
    RefinementAction,
    StructuralObjectives,
    StructuralProposal,
)
from trace2tower.methods.trace2tower.deployment_optimization.pareto import rank_fronts


@dataclass(frozen=True, slots=True)
class SegmentStructuralEvidence:
    successful: bool
    event_type: str
    embedding: np.ndarray


def build_structural_proposals(
    lineage: Sequence[LineageComponent],
    old_clusters: Sequence[MidCluster],
    new_clusters: Sequence[MidCluster],
    evidence: Mapping[str, SegmentStructuralEvidence],
    *,
    tolerance: float = 1e-9,
) -> tuple[StructuralProposal, ...]:
    old_by_id = {cluster.cluster_id: cluster for cluster in old_clusters}
    new_by_id = {cluster.cluster_id: cluster for cluster in new_clusters}
    candidates = []
    for component in lineage:
        action = _action(component.kind)
        if action is None:
            continue
        old_partition = tuple(old_by_id[mid_id] for mid_id in component.old_mid_ids)
        new_partition = tuple(new_by_id[mid_id] for mid_id in component.new_mid_ids)
        historical_ids = frozenset(
            segment_id
            for cluster in old_partition
            for segment_id in cluster.member_segment_ids
        )
        new_member_ids = {
            segment_id
            for cluster in new_partition
            for segment_id in cluster.member_segment_ids
        }
        evaluation_ids = historical_ids & new_member_ids
        old_scores = _partition_scores(old_partition, evaluation_ids, evidence)
        new_scores = _partition_scores(new_partition, evaluation_ids, evidence)
        objectives = StructuralObjectives(
            *(new_value - old_value for old_value, new_value in zip(old_scores, new_scores))
        )
        if all(value >= -tolerance for value in objectives.values) and any(
            value > tolerance for value in objectives.values
        ):
            candidates.append((action, component, objectives))

    ranks = rank_fronts(
        {component.component_id: objectives.values for _, component, objectives in candidates}
    )
    return tuple(
        StructuralProposal(action, component, objectives, ranks[component.component_id])
        for action, component, objectives in candidates
    )


def _action(kind: LineageKind) -> RefinementAction | None:
    if kind is LineageKind.SPLIT:
        return RefinementAction.SPLIT
    if kind is LineageKind.MERGE:
        return RefinementAction.MERGE
    return None


def _partition_scores(
    clusters: Sequence[MidCluster],
    historical_ids: frozenset[str],
    evidence: Mapping[str, SegmentStructuralEvidence],
) -> tuple[float, float, float]:
    groups = tuple(
        tuple(segment_id for segment_id in cluster.member_segment_ids if segment_id in historical_ids)
        for cluster in clusters
    )
    groups = tuple(group for group in groups if group)
    assigned_ids = {segment_id for group in groups for segment_id in group}
    if assigned_ids != set(historical_ids):
        missing = len(historical_ids - assigned_ids)
        raise ValueError(f"structural partition drops {missing} historical segments")
    if not assigned_ids <= set(evidence):
        raise ValueError("structural evidence does not cover the historical partition")

    total = len(assigned_ids)
    outcome = sum(_purity(group, evidence, "outcome") * len(group) for group in groups) / total
    role = sum(_purity(group, evidence, "event") * len(group) for group in groups) / total
    compactness = sum(_compactness(group, evidence) * len(group) for group in groups) / total
    return outcome, role, compactness


def _purity(
    segment_ids: Sequence[str],
    evidence: Mapping[str, SegmentStructuralEvidence],
    kind: str,
) -> float:
    if kind == "outcome":
        values = [evidence[segment_id].successful for segment_id in segment_ids]
    else:
        values = [evidence[segment_id].event_type for segment_id in segment_ids]
    return max(Counter(values).values()) / len(values)


def _compactness(
    segment_ids: Sequence[str],
    evidence: Mapping[str, SegmentStructuralEvidence],
) -> float:
    vectors = np.asarray([evidence[segment_id].embedding for segment_id in segment_ids])
    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominators = vector_norms * centroid_norm
    similarities = np.divide(
        vectors @ centroid,
        denominators,
        out=np.zeros(len(vectors), dtype=np.float64),
        where=denominators > 0,
    )
    return float(similarities.mean())
