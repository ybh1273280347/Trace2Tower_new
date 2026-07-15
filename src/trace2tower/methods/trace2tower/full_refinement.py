from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from trace2tower.methods.trace2tower.lineage import (
    ComplexRepartitionProposal,
    MidLineage,
    legal_merge_proposals,
    legal_split_proposals,
)
from trace2tower.methods.trace2tower.models import HighPath, MidCluster, SegmentInstance
from trace2tower.methods.trace2tower.refinement import (
    LegalMergeProposal,
    LegalSplitProposal,
    ObjectiveVector,
    RankedSkillObjective,
    prioritize_merges,
    prioritize_splits,
)


@dataclass(frozen=True, slots=True)
class StructuralSelection:
    split: LegalSplitProposal | None
    merge: LegalMergeProposal | None
    repartition: ComplexRepartitionProposal | None
    rejected_split_proposal_ids: tuple[str, ...]
    rejected_repartition_proposal_ids: tuple[str, ...]
    rejected_merge_proposal_ids: tuple[str, ...]
    pareto_protected_merge_proposal_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RefinedMidStructure:
    clusters: tuple[MidCluster, ...]
    replacement_by_old_id: dict[str, tuple[str, ...]]
    primary_replacement_by_old_id: dict[str, tuple[str, ...]]
    source_old_ids_by_mid_id: dict[str, tuple[str, ...]]
    candidate_to_final_mid_id: dict[str, str]


def select_structural_updates(
    lineage: MidLineage,
    old_clusters: tuple[MidCluster, ...],
    candidate_clusters: tuple[MidCluster, ...],
    old_high_paths: tuple[HighPath, ...],
    ranked_mid_skills: dict[str, RankedSkillObjective],
    *,
    max_high_path_length: int,
    minimum_exposure_count: int,
    excluded_proposal_ids: frozenset[str] = frozenset(),
) -> StructuralSelection:
    candidate_by_id = {
        cluster.cluster_id: cluster for cluster in candidate_clusters
    }
    split_targets = dict(lineage.splits)
    selected_split = None
    selected_repartition = None
    rejected_splits = []
    rejected_repartitions = []
    split_proposals = legal_split_proposals(lineage)
    known_splits = tuple(
        proposal
        for proposal in split_proposals
        if proposal.source_skill_id in ranked_mid_skills
    )
    pure_splits = (
        *prioritize_splits(known_splits, ranked_mid_skills),
        *sorted(
            (
                proposal
                for proposal in split_proposals
                if proposal.source_skill_id not in ranked_mid_skills
            ),
            key=lambda proposal: proposal.proposal_id,
        ),
    )
    structural_candidates = [
        (
            _weak_source_key(
                (proposal.source_skill_id,),
                ranked_mid_skills,
                minimum_exposure_count,
            )
            + (0, proposal.proposal_id),
            "split",
            proposal,
        )
        for proposal in pure_splits
        if proposal.proposal_id not in excluded_proposal_ids
    ]
    structural_candidates.extend(
        (
            _weak_source_key(
                proposal.source_old_skill_ids,
                ranked_mid_skills,
                minimum_exposure_count,
            )
            + (
                1,
                -proposal.shared_member_count,
                proposal.mean_centroid_drift,
                proposal.proposal_id,
            ),
            "repartition",
            proposal,
        )
        for proposal in lineage.complex_repartitions
        if proposal.proposal_id not in excluded_proposal_ids
    )
    for _, kind, proposal in sorted(structural_candidates, key=lambda item: item[0]):
        if kind == "split":
            primary_candidate = _primary_candidate_for_old(
                lineage,
                proposal.source_skill_id,
                split_targets[proposal.source_skill_id],
            )
            replacements = {
                proposal.source_skill_id: (
                    _structural_mid_id(
                        "split",
                        (proposal.source_skill_id,),
                        candidate_by_id[primary_candidate].member_segment_ids,
                    ),
                )
            }
        else:
            final_by_candidate = {
                candidate_id: _structural_mid_id(
                    "repartition",
                    proposal.source_old_skill_ids,
                    candidate_by_id[candidate_id].member_segment_ids,
                )
                for candidate_id in proposal.candidate_cluster_ids
            }
            replacements = {
                old_id: (
                    final_by_candidate[
                        _primary_candidate_for_old(
                            lineage,
                            old_id,
                            proposal.candidate_cluster_ids,
                        )
                    ],
                )
                for old_id in proposal.source_old_skill_ids
            }
        if _preserves_high_contract(
            old_high_paths, replacements, max_high_path_length
        ):
            if kind == "split":
                selected_split = proposal
            else:
                selected_repartition = proposal
            break
        if kind == "split":
            rejected_splits.append(proposal.proposal_id)
        else:
            rejected_repartitions.append(proposal.proposal_id)

    merge_proposals = legal_merge_proposals(
        lineage, old_clusters, candidate_clusters
    )
    known_merges = tuple(
        proposal
        for proposal in merge_proposals
        if proposal.left_skill_id in ranked_mid_skills
        and proposal.right_skill_id in ranked_mid_skills
    )
    merge_ranking = prioritize_merges(known_merges, ranked_mid_skills)
    unranked_merges = tuple(
        sorted(
            (proposal for proposal in merge_proposals if proposal not in known_merges),
            key=lambda proposal: (
                -proposal.member_overlap,
                proposal.centroid_drift,
                proposal.left_skill_id,
                proposal.right_skill_id,
                proposal.proposal_id,
            ),
        )
    )
    selected_merge = None
    rejected_merges = []
    split_source = selected_split.source_skill_id if selected_split else None
    merge_target_by_sources = {
        source_ids: candidate_id for source_ids, candidate_id in lineage.merges
    }
    for proposal in (*merge_ranking.eligible, *unranked_merges):
        if proposal.proposal_id in excluded_proposal_ids:
            continue
        if selected_repartition:
            rejected_merges.append(proposal.proposal_id)
            continue
        if split_source in (proposal.left_skill_id, proposal.right_skill_id):
            rejected_merges.append(proposal.proposal_id)
            continue
        candidate_id = merge_target_by_sources[
            (proposal.left_skill_id, proposal.right_skill_id)
        ]
        merged_id = _structural_mid_id(
            "merge",
            (proposal.left_skill_id, proposal.right_skill_id),
            candidate_by_id[candidate_id].member_segment_ids,
        )
        replacements = {
            proposal.left_skill_id: (merged_id,),
            proposal.right_skill_id: (merged_id,),
        }
        if selected_split:
            primary_candidate = _primary_candidate_for_old(
                lineage,
                selected_split.source_skill_id,
                split_targets[selected_split.source_skill_id],
            )
            replacements[selected_split.source_skill_id] = (
                _structural_mid_id(
                    "split",
                    (selected_split.source_skill_id,),
                    candidate_by_id[primary_candidate].member_segment_ids,
                ),
            )
        if _preserves_high_contract(
            old_high_paths, replacements, max_high_path_length
        ):
            selected_merge = proposal
            break
        rejected_merges.append(proposal.proposal_id)

    return StructuralSelection(
        split=selected_split,
        merge=selected_merge,
        repartition=selected_repartition,
        rejected_split_proposal_ids=tuple(rejected_splits),
        rejected_repartition_proposal_ids=tuple(rejected_repartitions),
        rejected_merge_proposal_ids=tuple(rejected_merges),
        pareto_protected_merge_proposal_ids=tuple(
            proposal.proposal_id for proposal in merge_ranking.pareto_protected
        ),
    )


def apply_mid_updates(
    lineage: MidLineage,
    selection: StructuralSelection,
    old_clusters: tuple[MidCluster, ...],
    candidate_clusters: tuple[MidCluster, ...],
    segments: dict[str, SegmentInstance],
) -> RefinedMidStructure:
    old_by_id = {cluster.cluster_id: cluster for cluster in old_clusters}
    candidate_by_id = {
        cluster.cluster_id: cluster for cluster in candidate_clusters
    }
    historical_ids = {
        segment_id
        for cluster in old_clusters
        for segment_id in cluster.member_segment_ids
    }
    candidate_by_segment = {
        segment_id: cluster.cluster_id
        for cluster in candidate_clusters
        for segment_id in cluster.member_segment_ids
    }
    final_members = {
        cluster_id: set(cluster.member_segment_ids)
        for cluster_id, cluster in old_by_id.items()
    }
    source_old_ids = {cluster_id: (cluster_id,) for cluster_id in old_by_id}
    candidate_to_final = {
        candidate_id: old_id for old_id, candidate_id in lineage.continuations
    }
    replacement_by_old = {cluster_id: (cluster_id,) for cluster_id in old_by_id}
    primary_replacement_by_old = dict(replacement_by_old)

    if selection.repartition:
        source_ids = selection.repartition.source_old_skill_ids
        final_by_candidate = {}
        for source_id in source_ids:
            final_members.pop(source_id)
            source_old_ids.pop(source_id)
        for candidate_id in selection.repartition.candidate_cluster_ids:
            cluster = candidate_by_id[candidate_id]
            contributing_sources = tuple(
                sorted(
                    edge.old_skill_id
                    for edge in lineage.edges
                    if edge.candidate_cluster_id == candidate_id
                    and edge.old_skill_id in source_ids
                )
            )
            final_id = _structural_mid_id(
                "repartition", source_ids, cluster.member_segment_ids
            )
            final_members[final_id] = set(cluster.member_segment_ids)
            source_old_ids[final_id] = contributing_sources
            candidate_to_final[candidate_id] = final_id
            final_by_candidate[candidate_id] = final_id
        for source_id in source_ids:
            replacement_by_old[source_id] = tuple(
                final_by_candidate[candidate_id]
                for candidate_id in selection.repartition.candidate_cluster_ids
                if any(
                    edge.old_skill_id == source_id
                    and edge.candidate_cluster_id == candidate_id
                    for edge in lineage.edges
                )
            )
            primary_candidate = _primary_candidate_for_old(
                lineage,
                source_id,
                selection.repartition.candidate_cluster_ids,
            )
            primary_replacement_by_old[source_id] = (
                final_by_candidate[primary_candidate],
            )

    if selection.split:
        source_id = selection.split.source_skill_id
        source_historical_ids = set(old_by_id[source_id].member_segment_ids)
        final_members.pop(source_id)
        source_old_ids.pop(source_id)
        replacement_ids = []
        for candidate_id in dict(lineage.splits)[source_id]:
            cluster = candidate_by_id[candidate_id]
            local_members = tuple(
                segment_id
                for segment_id in cluster.member_segment_ids
                if segment_id not in historical_ids
                or segment_id in source_historical_ids
            )
            final_id = _structural_mid_id(
                "split", (source_id,), local_members
            )
            final_members[final_id] = set(local_members)
            source_old_ids[final_id] = (source_id,)
            candidate_to_final[candidate_id] = final_id
            replacement_ids.append(final_id)
        replacement_by_old[source_id] = tuple(replacement_ids)
        primary_candidate = _primary_candidate_for_old(
            lineage,
            source_id,
            dict(lineage.splits)[source_id],
        )
        primary_replacement_by_old[source_id] = (
            candidate_to_final[primary_candidate],
        )

    if selection.merge:
        left_id = selection.merge.left_skill_id
        right_id = selection.merge.right_skill_id
        candidate_id = dict(lineage.merges)[(left_id, right_id)]
        cluster = candidate_by_id[candidate_id]
        final_id = _structural_mid_id(
            "merge", (left_id, right_id), cluster.member_segment_ids
        )
        for source_id in (left_id, right_id):
            final_members.pop(source_id)
            source_old_ids.pop(source_id)
            replacement_by_old[source_id] = (final_id,)
            primary_replacement_by_old[source_id] = (final_id,)
        final_members[final_id] = set(cluster.member_segment_ids)
        source_old_ids[final_id] = (left_id, right_id)
        candidate_to_final[candidate_id] = final_id

    already_assigned = set().union(*final_members.values())
    edges_by_candidate: dict[str, list] = defaultdict(list)
    for edge in lineage.edges:
        edges_by_candidate[edge.candidate_cluster_id].append(edge)
    for segment_id in sorted(set(candidate_by_segment) - already_assigned):
        candidate_id = candidate_by_segment[segment_id]
        final_id = candidate_to_final.get(candidate_id)
        if final_id is None:
            anchors = sorted(
                edges_by_candidate[candidate_id],
                key=lambda edge: (
                    -edge.shared_member_count,
                    -edge.new_historical_purity,
                    -edge.centroid_similarity,
                    edge.old_skill_id,
                ),
            )
            final_id = next(
                (
                    edge.old_skill_id
                    for edge in anchors
                    if edge.old_skill_id in final_members
                ),
                None,
            )
        if final_id is None:
            members = tuple(
                sorted(
                    item
                    for item in candidate_by_id[candidate_id].member_segment_ids
                    if item not in historical_ids
                )
            )
            final_id = _structural_mid_id("new", (), members)
            final_members.setdefault(final_id, set())
            source_old_ids.setdefault(final_id, ())
            candidate_to_final[candidate_id] = final_id
        final_members[final_id].add(segment_id)

    assigned = [segment_id for members in final_members.values() for segment_id in members]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(candidate_by_segment):
        raise ValueError("refined Mid clusters do not partition candidate segments")
    refined_clusters = tuple(
        MidCluster(
            cluster_id=cluster_id,
            member_segment_ids=tuple(sorted(member_ids)),
            centroid=tuple(
                np.mean(
                    np.asarray(
                        [
                            segments[segment_id].embedding
                            for segment_id in sorted(member_ids)
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                ).tolist()
            ),
        )
        for cluster_id, member_ids in sorted(final_members.items())
    )
    return RefinedMidStructure(
        clusters=refined_clusters,
        replacement_by_old_id=replacement_by_old,
        primary_replacement_by_old_id=primary_replacement_by_old,
        source_old_ids_by_mid_id=source_old_ids,
        candidate_to_final_mid_id=candidate_to_final,
    )


def project_high_path(
    ordered_mid_ids: tuple[str, ...],
    replacement_by_old_id: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    projected = []
    for mid_id in ordered_mid_ids:
        for replacement in replacement_by_old_id.get(mid_id, (mid_id,)):
            if not projected or projected[-1] != replacement:
                projected.append(replacement)
    return tuple(projected)


def project_mid_objectives(
    structure: RefinedMidStructure,
    ranked_mid_skills: dict[str, RankedSkillObjective],
) -> dict[str, RankedSkillObjective]:
    projected = {}
    for mid_id, source_ids in structure.source_old_ids_by_mid_id.items():
        sources = [
            ranked_mid_skills[source_id]
            for source_id in source_ids
            if source_id in ranked_mid_skills
        ]
        if not sources:
            continue
        total_exposure = sum(source.exposure_count for source in sources)
        values = tuple(
            sum(
                source.exposure_count * source.objective_vector.values[index]
                for source in sources
            )
            / total_exposure
            for index in range(4)
        )
        pair_keys = tuple(
            sorted(
                {
                    key
                    for source in sources
                    for key in source.paired_episode_keys
                }
            )
        )
        projected[mid_id] = RankedSkillObjective(
            skill_id=mid_id,
            benchmark=sources[0].benchmark,
            skill_level=sources[0].skill_level,
            refinement_round=sources[0].refinement_round,
            objective_vector=ObjectiveVector(*values),
            exposure_count=total_exposure,
            paired_episode_keys=pair_keys,
            pareto_front_rank=min(source.pareto_front_rank for source in sources),
            dominated_by=(),
            dominates=(),
        )
    return projected


def _preserves_high_contract(
    paths: tuple[HighPath, ...],
    replacements: dict[str, tuple[str, ...]],
    max_length: int,
) -> bool:
    return all(
        2 <= len(projected := project_high_path(path.ordered_mid_ids, replacements))
        <= max_length
        and len(set(projected)) >= 2
        for path in paths
    )


def _structural_mid_id(
    action: str, source_ids: tuple[str, ...], member_segment_ids: tuple[str, ...]
) -> str:
    payload = "\x1f".join((action, *source_ids, *sorted(member_segment_ids)))
    return f"mid_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def _primary_candidate_for_old(
    lineage: MidLineage,
    old_skill_id: str,
    candidate_ids: tuple[str, ...],
) -> str:
    candidates = [
        edge
        for edge in lineage.edges
        if edge.old_skill_id == old_skill_id
        and edge.candidate_cluster_id in candidate_ids
    ]
    if not candidates:
        raise ValueError("lineage source has no candidate descendant")
    return min(
        candidates,
        key=lambda edge: (
            -edge.shared_member_count,
            -edge.old_retention,
            -edge.new_historical_purity,
            -edge.centroid_similarity,
            edge.candidate_cluster_id,
        ),
    ).candidate_cluster_id


def _weak_source_key(
    source_ids: tuple[str, ...],
    ranked_mid_skills: dict[str, RankedSkillObjective],
    minimum_exposure_count: int,
) -> tuple:
    available = [
        ranked_mid_skills[source_id]
        for source_id in source_ids
        if source_id in ranked_mid_skills
    ]
    if not available:
        return (2, min(source_ids))
    sufficiently_exposed = [
        skill
        for skill in available
        if skill.exposure_count >= minimum_exposure_count
    ]
    comparison_pool = sufficiently_exposed or available
    weakest = min(
        comparison_pool,
        key=lambda skill: (
            -skill.pareto_front_rank,
            skill.objective_vector.paired_reward_gain,
            skill.objective_vector.performance_level,
            -skill.exposure_count,
            skill.skill_id,
        ),
    )
    return (
        0 if sufficiently_exposed else 1,
        -weakest.pareto_front_rank,
        weakest.objective_vector.paired_reward_gain,
        weakest.objective_vector.performance_level,
        -weakest.exposure_count,
        weakest.skill_id,
    )
