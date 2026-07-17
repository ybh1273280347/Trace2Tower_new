from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from trace2tower.methods.trace2tower.core.models import MidCluster
from trace2tower.methods.trace2tower.deployment_optimization.models import (
    LineageComponent,
    LineageKind,
    LineageOverlap,
)


def build_mid_lineage(
    old_clusters: Iterable[MidCluster],
    new_clusters: Iterable[MidCluster],
) -> tuple[LineageComponent, ...]:
    old_members = _member_sets(old_clusters, "old")
    new_members = _member_sets(new_clusters, "new")
    old_to_new = defaultdict(set)
    new_to_old = defaultdict(set)
    overlaps = {}
    for old_mid_id, old_segment_ids in old_members.items():
        for new_mid_id, new_segment_ids in new_members.items():
            shared = old_segment_ids & new_segment_ids
            if not shared:
                continue
            old_to_new[old_mid_id].add(new_mid_id)
            new_to_old[new_mid_id].add(old_mid_id)
            overlaps[(old_mid_id, new_mid_id)] = LineageOverlap(
                old_mid_id=old_mid_id,
                new_mid_id=new_mid_id,
                shared_member_count=len(shared),
                old_retention=len(shared) / len(old_segment_ids),
                new_historical_purity=len(shared) / len(new_segment_ids),
            )

    components = []
    remaining_old = set(old_members)
    remaining_new = set(new_members)
    while remaining_old or remaining_new:
        if remaining_old:
            pending_old = {min(remaining_old)}
            pending_new = set()
        else:
            pending_old = set()
            pending_new = {min(remaining_new)}
        component_old = set()
        component_new = set()
        while pending_old or pending_new:
            if pending_old:
                old_mid_id = pending_old.pop()
                if old_mid_id in component_old:
                    continue
                component_old.add(old_mid_id)
                pending_new.update(old_to_new[old_mid_id] - component_new)
            else:
                new_mid_id = pending_new.pop()
                if new_mid_id in component_new:
                    continue
                component_new.add(new_mid_id)
                pending_old.update(new_to_old[new_mid_id] - component_old)
        remaining_old -= component_old
        remaining_new -= component_new
        old_mid_ids = tuple(sorted(component_old))
        new_mid_ids = tuple(sorted(component_new))
        component_overlaps = tuple(
            overlaps[(old_mid_id, new_mid_id)]
            for old_mid_id in old_mid_ids
            for new_mid_id in new_mid_ids
            if (old_mid_id, new_mid_id) in overlaps
        )
        components.append(
            LineageComponent(
                component_id=f"lineage_{len(components) + 1:04d}",
                kind=_lineage_kind(len(old_mid_ids), len(new_mid_ids)),
                old_mid_ids=old_mid_ids,
                new_mid_ids=new_mid_ids,
                overlaps=component_overlaps,
            )
        )
    return tuple(components)


def _member_sets(clusters: Iterable[MidCluster], label: str) -> dict[str, set[str]]:
    selected = tuple(clusters)
    cluster_ids = [cluster.cluster_id for cluster in selected]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError(f"{label} Mid clusters contain duplicate IDs")
    members = {cluster.cluster_id: set(cluster.member_segment_ids) for cluster in selected}
    if any(not segment_ids for segment_ids in members.values()):
        raise ValueError(f"{label} Mid clusters cannot be empty")
    return members


def _lineage_kind(old_count: int, new_count: int) -> LineageKind:
    if old_count == 0:
        return LineageKind.NEW_MID
    if new_count == 0:
        return LineageKind.DISAPPEARED_MID
    if old_count == new_count == 1:
        return LineageKind.CONTINUATION
    if old_count == 1:
        return LineageKind.SPLIT
    if new_count == 1:
        return LineageKind.MERGE
    return LineageKind.RECOMPOSED
