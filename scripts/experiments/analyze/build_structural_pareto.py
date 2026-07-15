from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy import sparse

from trace2tower.methods.trace2tower.lineage import (
    build_mid_lineage,
    decompose_mid_lineage,
)
from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.tower import TowerSnapshot


@dataclass(frozen=True, slots=True)
class StructuralMetrics:
    component_id: str
    source_old_skill_ids: tuple[str, ...]
    candidate_cluster_ids: tuple[str, ...]
    core_member_count: int
    source_member_coverage: float
    minimum_target_core_count: int
    outcome_consistency_gain: float
    transition_role_coherence_gain: float
    spectral_compactness_gain: float
    pareto_front_rank: int = 0
    dominates_noop: bool = False


def within_dense_dispersion(
    values: np.ndarray, labels: tuple[str, ...]
) -> float:
    total = 0.0
    for label in sorted(set(labels)):
        group = values[np.asarray([current == label for current in labels])]
        total += float(np.square(group - group.mean(axis=0)).sum())
    return total / len(labels)


def within_sparse_dispersion(
    values: sparse.csr_matrix, labels: tuple[str, ...]
) -> float:
    total = 0.0
    for label in sorted(set(labels)):
        indices = np.asarray(
            [index for index, current in enumerate(labels) if current == label]
        )
        group = values[indices]
        squared_norm = float(group.multiply(group).sum())
        mean = np.asarray(group.mean(axis=0)).ravel()
        total += max(0.0, squared_norm - len(indices) * float(mean @ mean))
    return total / len(labels)


def dominates(left: StructuralMetrics, right: StructuralMetrics) -> bool:
    left_values = (
        left.outcome_consistency_gain,
        left.transition_role_coherence_gain,
        left.spectral_compactness_gain,
    )
    right_values = (
        right.outcome_consistency_gain,
        right.transition_role_coherence_gain,
        right.spectral_compactness_gain,
    )
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


def rank_fronts(items: tuple[StructuralMetrics, ...]) -> tuple[StructuralMetrics, ...]:
    remaining = list(items)
    ranked = []
    rank = 1
    while remaining:
        front = [
            item
            for item in remaining
            if not any(dominates(other, item) for other in remaining if other != item)
        ]
        ranked.extend(replace(item, pareto_front_rank=rank) for item in front)
        remaining = [item for item in remaining if item not in front]
        rank += 1
    return tuple(sorted(ranked, key=lambda item: item.component_id))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--candidate-clusters", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    tower = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    candidate_clusters = tuple(
        MidCluster.from_record(record)
        for record in json.loads(
            options.candidate_clusters.read_text(encoding="utf-8")
        )["clusters"]
    )
    lineage = build_mid_lineage(tower.mid_clusters, candidate_clusters)
    decomposition = decompose_mid_lineage(lineage)

    spectral = np.load(options.spectral)
    segment_ids = tuple(str(value) for value in spectral["segment_ids"])
    index_by_id = {segment_id: index for index, segment_id in enumerate(segment_ids)}
    rho = np.asarray(spectral["rho"], dtype=np.float64).reshape(-1, 1)
    representation = np.asarray(spectral["representation"], dtype=np.float64)
    transition = sparse.load_npz(options.transition).tocsr()
    row_norms = np.sqrt(np.asarray(transition.multiply(transition).sum(axis=1)).ravel())
    inverse_norms = np.zeros_like(row_norms)
    inverse_norms[row_norms > 0] = 1 / row_norms[row_norms > 0]
    transition_profiles = sparse.diags(inverse_norms) @ transition

    old_members = {
        cluster.cluster_id: set(cluster.member_segment_ids)
        for cluster in tower.mid_clusters
    }
    candidate_members = {
        cluster.cluster_id: set(cluster.member_segment_ids)
        for cluster in candidate_clusters
    }
    old_by_member = {
        segment_id: cluster_id
        for cluster_id, members in old_members.items()
        for segment_id in members
    }
    candidate_by_member = {
        segment_id: cluster_id
        for cluster_id, members in candidate_members.items()
        for segment_id in members
    }

    metrics = []
    for component_index, (old_ids, candidate_ids) in enumerate(
        decomposition.components
    ):
        if len(old_ids) == 1 and len(candidate_ids) == 1:
            continue
        source_members = set().union(*(old_members[skill_id] for skill_id in old_ids))
        target_members = set().union(
            *(candidate_members[cluster_id] for cluster_id in candidate_ids)
        )
        core_ids = tuple(sorted(source_members & target_members))
        indices = np.asarray([index_by_id[segment_id] for segment_id in core_ids])
        old_labels = tuple(old_by_member[segment_id] for segment_id in core_ids)
        candidate_labels = tuple(
            candidate_by_member[segment_id] for segment_id in core_ids
        )
        old_outcome = within_dense_dispersion(rho[indices], old_labels)
        new_outcome = within_dense_dispersion(rho[indices], candidate_labels)
        old_transition = within_sparse_dispersion(
            transition_profiles[indices], old_labels
        )
        new_transition = within_sparse_dispersion(
            transition_profiles[indices], candidate_labels
        )
        old_spectral = within_dense_dispersion(
            representation[indices], old_labels
        )
        new_spectral = within_dense_dispersion(
            representation[indices], candidate_labels
        )
        target_counts = [candidate_labels.count(cluster_id) for cluster_id in candidate_ids]
        metrics.append(
            StructuralMetrics(
                component_id=f"lineage_component_{component_index:02d}",
                source_old_skill_ids=old_ids,
                candidate_cluster_ids=candidate_ids,
                core_member_count=len(core_ids),
                source_member_coverage=len(core_ids) / len(source_members),
                minimum_target_core_count=min(target_counts),
                outcome_consistency_gain=old_outcome - new_outcome,
                transition_role_coherence_gain=old_transition - new_transition,
                spectral_compactness_gain=old_spectral - new_spectral,
            )
        )

    noop = StructuralMetrics(
        component_id="noop",
        source_old_skill_ids=(),
        candidate_cluster_ids=(),
        core_member_count=0,
        source_member_coverage=1.0,
        minimum_target_core_count=0,
        outcome_consistency_gain=0.0,
        transition_role_coherence_gain=0.0,
        spectral_compactness_gain=0.0,
    )
    ranked = rank_fronts((noop, *metrics))
    ranked_noop = next(item for item in ranked if item.component_id == "noop")
    ranked = tuple(
        replace(item, dominates_noop=dominates(item, ranked_noop))
        for item in ranked
    )
    eligible = [item for item in ranked if item.dominates_noop]
    selected = min(
        eligible,
        key=lambda item: (
            item.pareto_front_rank,
            -item.outcome_consistency_gain,
            -item.transition_role_coherence_gain,
            -item.spectral_compactness_gain,
            item.component_id,
        ),
        default=None,
    )
    payload = {
        "protocol_id": "webshop-train-refinement-v1-structural-pareto",
        "tower_snapshot_id": tower.snapshot_id,
        "comparison_scope": "shared historical core members per local lineage component",
        "objectives": [
            "outcome_consistency_gain",
            "transition_role_coherence_gain",
            "spectral_compactness_gain",
        ],
        "selection_gate": "candidate must Pareto-dominate the no-op vector (0, 0, 0)",
        "candidates": [asdict(item) for item in ranked],
        "selected_component_id": selected.component_id if selected else None,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
