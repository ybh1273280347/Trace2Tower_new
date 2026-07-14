from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.refinement import (
    LegalMergeProposal,
    LegalSplitProposal,
)


@dataclass(frozen=True, slots=True)
class MidLineageEdge:
    old_skill_id: str
    candidate_cluster_id: str
    shared_member_count: int
    old_retention: float
    new_historical_purity: float
    centroid_similarity: float

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComplexRepartitionProposal:
    proposal_id: str
    source_old_skill_ids: tuple[str, ...]
    candidate_cluster_ids: tuple[str, ...]
    shared_member_count: int
    mean_centroid_drift: float

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MidLineage:
    edges: tuple[MidLineageEdge, ...]
    continuations: tuple[tuple[str, str], ...]
    splits: tuple[tuple[str, tuple[str, ...]], ...]
    merges: tuple[tuple[tuple[str, ...], str], ...]
    new_candidate_cluster_ids: tuple[str, ...]
    disappeared_old_skill_ids: tuple[str, ...]
    complex_old_skill_ids: tuple[str, ...]
    complex_candidate_cluster_ids: tuple[str, ...]
    complex_repartitions: tuple[ComplexRepartitionProposal, ...]

    def to_record(self) -> dict:
        return {
            "edges": [edge.to_record() for edge in self.edges],
            "continuations": [
                {"old_skill_id": old_id, "candidate_cluster_id": candidate_id}
                for old_id, candidate_id in self.continuations
            ],
            "splits": [
                {
                    "source_skill_id": old_id,
                    "candidate_cluster_ids": candidate_ids,
                }
                for old_id, candidate_ids in self.splits
            ],
            "merges": [
                {
                    "source_skill_ids": old_ids,
                    "candidate_cluster_id": candidate_id,
                }
                for old_ids, candidate_id in self.merges
            ],
            "new_candidate_cluster_ids": self.new_candidate_cluster_ids,
            "disappeared_old_skill_ids": self.disappeared_old_skill_ids,
            "complex_old_skill_ids": self.complex_old_skill_ids,
            "complex_candidate_cluster_ids": self.complex_candidate_cluster_ids,
            "complex_repartitions": [
                proposal.to_record() for proposal in self.complex_repartitions
            ],
        }


def build_mid_lineage(
    old_clusters: tuple[MidCluster, ...],
    candidate_clusters: tuple[MidCluster, ...],
) -> MidLineage:
    old_by_id = _cluster_map(old_clusters, "old")
    candidate_by_id = _cluster_map(candidate_clusters, "candidate")
    old_members = {
        cluster_id: set(cluster.member_segment_ids)
        for cluster_id, cluster in old_by_id.items()
    }
    historical_members = set().union(*old_members.values())
    candidate_members = {
        cluster_id: set(cluster.member_segment_ids)
        for cluster_id, cluster in candidate_by_id.items()
    }
    if not historical_members <= set().union(*candidate_members.values()):
        raise ValueError("candidate clusters lost historical segment members")

    edges = []
    old_to_candidates: dict[str, list[str]] = defaultdict(list)
    candidate_to_old: dict[str, list[str]] = defaultdict(list)
    for old_id, candidate_id in itertools.product(old_by_id, candidate_by_id):
        shared = old_members[old_id] & candidate_members[candidate_id]
        if not shared:
            continue
        candidate_historical = candidate_members[candidate_id] & historical_members
        edges.append(
            MidLineageEdge(
                old_skill_id=old_id,
                candidate_cluster_id=candidate_id,
                shared_member_count=len(shared),
                old_retention=len(shared) / len(old_members[old_id]),
                new_historical_purity=len(shared) / len(candidate_historical),
                centroid_similarity=_cosine_similarity(
                    old_by_id[old_id].centroid,
                    candidate_by_id[candidate_id].centroid,
                ),
            )
        )
        old_to_candidates[old_id].append(candidate_id)
        candidate_to_old[candidate_id].append(old_id)

    continuations = []
    splits = []
    merges = []
    complex_old = set()
    complex_candidate = set()
    for old_id, candidate_ids in old_to_candidates.items():
        if len(candidate_ids) == 1 and len(candidate_to_old[candidate_ids[0]]) == 1:
            continuations.append((old_id, candidate_ids[0]))
        elif len(candidate_ids) > 1 and all(
            len(candidate_to_old[candidate_id]) == 1
            for candidate_id in candidate_ids
        ):
            splits.append((old_id, tuple(sorted(candidate_ids))))
        elif len(candidate_ids) > 1:
            complex_old.add(old_id)
    for candidate_id, old_ids in candidate_to_old.items():
        if len(old_ids) > 1 and all(
            len(old_to_candidates[old_id]) == 1 for old_id in old_ids
        ):
            merges.append((tuple(sorted(old_ids)), candidate_id))
        elif len(old_ids) > 1:
            complex_candidate.add(candidate_id)

    complex_repartitions = []
    for old_ids, candidate_ids in _complex_components(
        old_to_candidates, candidate_to_old
    ):
        component_edges = [
            edge
            for edge in edges
            if edge.old_skill_id in old_ids
            and edge.candidate_cluster_id in candidate_ids
        ]
        complex_repartitions.append(
            ComplexRepartitionProposal(
                proposal_id=(
                    "repartition:"
                    + ",".join(old_ids)
                    + "->"
                    + ",".join(candidate_ids)
                ),
                source_old_skill_ids=old_ids,
                candidate_cluster_ids=candidate_ids,
                shared_member_count=sum(
                    edge.shared_member_count for edge in component_edges
                ),
                mean_centroid_drift=(
                    sum(1 - edge.centroid_similarity for edge in component_edges)
                    / len(component_edges)
                ),
            )
        )

    return MidLineage(
        edges=tuple(
            sorted(edges, key=lambda edge: (edge.old_skill_id, edge.candidate_cluster_id))
        ),
        continuations=tuple(sorted(continuations)),
        splits=tuple(sorted(splits)),
        merges=tuple(sorted(merges)),
        new_candidate_cluster_ids=tuple(
            sorted(set(candidate_by_id) - set(candidate_to_old))
        ),
        disappeared_old_skill_ids=tuple(
            sorted(set(old_by_id) - set(old_to_candidates))
        ),
        complex_old_skill_ids=tuple(sorted(complex_old)),
        complex_candidate_cluster_ids=tuple(sorted(complex_candidate)),
        complex_repartitions=tuple(
            sorted(complex_repartitions, key=lambda item: item.proposal_id)
        ),
    )


def legal_split_proposals(lineage: MidLineage) -> tuple[LegalSplitProposal, ...]:
    return tuple(
        LegalSplitProposal(
            proposal_id=f"split:{source_id}:{','.join(candidate_ids)}",
            source_skill_id=source_id,
        )
        for source_id, candidate_ids in lineage.splits
    )


def legal_merge_proposals(
    lineage: MidLineage,
    old_clusters: tuple[MidCluster, ...],
    candidate_clusters: tuple[MidCluster, ...],
) -> tuple[LegalMergeProposal, ...]:
    old_by_id = {cluster.cluster_id: cluster for cluster in old_clusters}
    candidate_by_id = {
        cluster.cluster_id: cluster for cluster in candidate_clusters
    }
    proposals = []
    for source_ids, candidate_id in lineage.merges:
        if len(source_ids) != 2:
            continue
        left_id, right_id = source_ids
        left_members = set(old_by_id[left_id].member_segment_ids)
        right_members = set(old_by_id[right_id].member_segment_ids)
        candidate_members = set(candidate_by_id[candidate_id].member_segment_ids)
        source_members = left_members | right_members
        shared = source_members & candidate_members
        proposals.append(
            LegalMergeProposal(
                proposal_id=f"merge:{left_id}:{right_id}:{candidate_id}",
                left_skill_id=left_id,
                right_skill_id=right_id,
                member_overlap=len(shared) / len(source_members | candidate_members),
                centroid_drift=(
                    2
                    - _cosine_similarity(
                        old_by_id[left_id].centroid,
                        candidate_by_id[candidate_id].centroid,
                    )
                    - _cosine_similarity(
                        old_by_id[right_id].centroid,
                        candidate_by_id[candidate_id].centroid,
                    )
                )
                / 2,
            )
        )
    return tuple(proposals)


def _complex_components(
    old_to_candidates: dict[str, list[str]],
    candidate_to_old: dict[str, list[str]],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    visited_old: set[str] = set()
    visited_candidate: set[str] = set()
    components = []
    for root in sorted(old_to_candidates):
        if root in visited_old:
            continue
        old_ids = set()
        candidate_ids = set()
        pending_old = [root]
        while pending_old:
            old_id = pending_old.pop()
            if old_id in visited_old:
                continue
            visited_old.add(old_id)
            old_ids.add(old_id)
            for candidate_id in old_to_candidates[old_id]:
                if candidate_id in visited_candidate:
                    continue
                visited_candidate.add(candidate_id)
                candidate_ids.add(candidate_id)
                pending_old.extend(candidate_to_old[candidate_id])
        if len(old_ids) > 1 and len(candidate_ids) > 1:
            components.append((tuple(sorted(old_ids)), tuple(sorted(candidate_ids))))
    return tuple(components)


def _cluster_map(
    clusters: tuple[MidCluster, ...], label: str
) -> dict[str, MidCluster]:
    by_id = {cluster.cluster_id: cluster for cluster in clusters}
    if not by_id or len(by_id) != len(clusters):
        raise ValueError(f"{label} clusters require unique IDs")
    members = [
        segment_id for cluster in clusters for segment_id in cluster.member_segment_ids
    ]
    if len(members) != len(set(members)):
        raise ValueError(f"{label} clusters do not partition their segment members")
    return by_id


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("lineage centroids have different dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
