from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.semantic_index import SkillEmbeddingIndex


@dataclass(frozen=True, slots=True)
class MidTransitionSignalProfile:
    mid_ids: tuple[str, ...]
    transition_strength: dict[str, dict[str, float]]
    outcome_consistency: dict[str, dict[str, float]]

    def __post_init__(self) -> None:
        known = set(self.mid_ids)
        if not known or len(known) != len(self.mid_ids):
            raise ValueError("three-signal profile requires unique Mid IDs")
        for matrix in (self.transition_strength, self.outcome_consistency):
            if set(matrix) != known or any(set(row) - known for row in matrix.values()):
                raise ValueError("three-signal profile references unknown Mid IDs")
            if any(
                not 0 <= value <= 1
                for row in matrix.values()
                for value in row.values()
            ):
                raise ValueError("three-signal profile values must be in [0, 1]")

    def to_record(self) -> dict:
        return {
            "mid_ids": self.mid_ids,
            "transition_strength": self.transition_strength,
            "outcome_consistency": self.outcome_consistency,
        }

    @classmethod
    def from_record(cls, record: Mapping) -> MidTransitionSignalProfile:
        return cls(
            mid_ids=tuple(record["mid_ids"]),
            transition_strength={
                str(source): {
                    str(target): float(value) for target, value in row.items()
                }
                for source, row in record["transition_strength"].items()
            },
            outcome_consistency={
                str(source): {
                    str(target): float(value) for target, value in row.items()
                }
                for source, row in record["outcome_consistency"].items()
            },
        )


@dataclass(frozen=True, slots=True)
class ThreeSignalMidMatch:
    skill_id: str
    score: float
    semantic_similarity: float
    transition_strength: float
    outcome_consistency: float


def build_mid_transition_signal_profile(
    records: Sequence[Mapping],
    clusters: Sequence[MidCluster],
    *,
    success_threshold: float = 0.999,
) -> MidTransitionSignalProfile:
    segment_to_mid = {
        segment_id: cluster.cluster_id
        for cluster in clusters
        for segment_id in cluster.member_segment_ids
    }
    mid_ids = tuple(sorted(cluster.cluster_id for cluster in clusters))
    transition_counts = Counter()
    successful_counts = Counter()
    source_counts = Counter()

    for record in records:
        sequence = tuple(
            segment_to_mid[segment["segment_id"]]
            for segment in sorted(record["segments"], key=lambda item: item["start_step"])
        )
        successful = float(record["primary_score"]) >= success_threshold
        for source, target in zip(sequence, sequence[1:]):
            transition_counts[(source, target)] += 1
            source_counts[source] += 1
            if successful:
                successful_counts[(source, target)] += 1

    transition_strength = {mid_id: {} for mid_id in mid_ids}
    outcome_consistency = {mid_id: {} for mid_id in mid_ids}
    for (source, target), count in transition_counts.items():
        transition_strength[source][target] = count / source_counts[source]
        # Beta(1, 1) 后验避免把单条偶发成功转移当作确定规律。
        outcome_consistency[source][target] = (
            successful_counts[(source, target)] + 1
        ) / (count + 2)

    return MidTransitionSignalProfile(
        mid_ids,
        transition_strength,
        outcome_consistency,
    )


def retrieve_mid_three_signal(
    query_vector: Sequence[float],
    mid_index: SkillEmbeddingIndex,
    candidate_mid_ids: frozenset[str],
    profile: MidTransitionSignalProfile,
    *,
    top_k: int,
    score_threshold: float,
    anchor_top_k: int = 2,
) -> tuple[ThreeSignalMidMatch, ...]:
    if set(mid_index.skill_ids) != set(profile.mid_ids):
        raise ValueError("three-signal profile and Mid index differ")
    if not candidate_mid_ids <= set(profile.mid_ids):
        raise ValueError("three-signal candidates contain unknown Mid IDs")
    if not 1 <= top_k <= len(profile.mid_ids):
        raise ValueError("three-signal Top-K must fit the Mid library")
    if not 0 <= score_threshold <= 1:
        raise ValueError("three-signal threshold must be in [0, 1]")
    if not 1 <= anchor_top_k <= len(profile.mid_ids):
        raise ValueError("three-signal anchor count must fit the Mid library")

    semantic_matches = mid_index.search(query_vector, len(profile.mid_ids))
    semantic_by_id = {
        match.skill_id: max(0.0, match.cosine_similarity)
        for match in semantic_matches
    }
    anchors = tuple(
        match
        for match in semantic_matches
        if match.cosine_similarity > 0
    )[:anchor_top_k]
    total_anchor_similarity = sum(match.cosine_similarity for match in anchors)
    if not anchors or total_anchor_similarity <= 0:
        return ()

    signals = {}
    for candidate_id in candidate_mid_ids:
        transition = 0.0
        outcome = 0.0
        for anchor in anchors:
            weight = anchor.cosine_similarity / total_anchor_similarity
            transition += weight * profile.transition_strength[anchor.skill_id].get(
                candidate_id,
                0.0,
            )
            outcome += weight * profile.outcome_consistency[anchor.skill_id].get(
                candidate_id,
                0.0,
            )
        signals[candidate_id] = (
            semantic_by_id[candidate_id],
            transition,
            outcome,
        )

    maxima = tuple(
        max((values[index] for values in signals.values()), default=0.0)
        for index in range(3)
    )
    available_signal_ids = tuple(
        index for index, maximum in enumerate(maxima) if maximum > 0
    )
    if not available_signal_ids:
        return ()

    ranked = []
    for candidate_id, values in signals.items():
        score = sum(
            values[index] / maxima[index] for index in available_signal_ids
        ) / len(available_signal_ids)
        if score < score_threshold:
            continue
        ranked.append(
            ThreeSignalMidMatch(
                candidate_id,
                score,
                values[0],
                values[1],
                values[2],
            )
        )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (-item.score, -item.semantic_similarity, item.skill_id),
        )[:top_k]
    )
