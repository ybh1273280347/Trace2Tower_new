from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.models import (
    AlfworldEventType,
    EventType,
    HighPath,
    WebShopEventType,
    event_type_from_value,
)
from trace2tower.methods.trace2tower.retrieval import (
    TowerRetrieval,
    format_tower_context,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch, diverse_search


@dataclass(frozen=True, slots=True)
class TowerGraphProfile:
    tower_snapshot_id: str
    benchmark: Benchmark
    mid_event_counts: dict[str, dict[EventType, int]]

    def __post_init__(self) -> None:
        if not self.tower_snapshot_id:
            raise ValueError("graph profile requires a Tower snapshot ID")
        if not self.mid_event_counts or any(
            not counts or any(count <= 0 for count in counts.values())
            for counts in self.mid_event_counts.values()
        ):
            raise ValueError("graph profile requires positive Mid event counts")
        expected_type = (
            AlfworldEventType if self.benchmark is Benchmark.ALFWORLD else WebShopEventType
        )
        if any(
            not isinstance(event, expected_type)
            for counts in self.mid_event_counts.values()
            for event in counts
        ):
            raise ValueError("graph profile event types do not match its benchmark")

    def to_record(self) -> dict:
        return {
            "tower_snapshot_id": self.tower_snapshot_id,
            "benchmark": self.benchmark,
            "mid_event_counts": {
                mid_id: {event: count for event, count in counts.items()}
                for mid_id, counts in self.mid_event_counts.items()
            },
        }

    @classmethod
    def from_record(cls, record: Mapping) -> TowerGraphProfile:
        return cls(
            tower_snapshot_id=str(record["tower_snapshot_id"]),
            benchmark=Benchmark(record["benchmark"]),
            mid_event_counts={
                str(mid_id): {
                    event_type_from_value(event): int(count) for event, count in counts.items()
                }
                for mid_id, counts in record["mid_event_counts"].items()
            },
        )

    def compatibility(self, mid_id: str, allowed_events: frozenset[EventType]) -> float:
        counts = self.mid_event_counts[mid_id]
        total = sum(counts.values())
        return sum(counts.get(event, 0) for event in allowed_events) / total


@dataclass(frozen=True, slots=True)
class GraphHighMatch:
    skill_id: str
    score: float
    goal_similarity: float
    state_similarity: float
    event_compatibility: float
    path_quality: float
    active_mid_id: str


@dataclass(frozen=True, slots=True)
class GraphTowerRetrieval:
    retrieval: TowerRetrieval
    high_candidates: tuple[GraphHighMatch, ...]
    graph_high_match: GraphHighMatch | None


@dataclass(frozen=True, slots=True)
class TaskHighRetrieval:
    high_card: HighSkillCard | None
    candidate: SkillMatch | None
    match: SkillMatch | None


def retrieve_task_high(
    query_vector: Sequence[float],
    high_index: SkillEmbeddingIndex,
    high_cards: Mapping[str, HighSkillCard],
    profile: TowerGraphProfile,
    required_path_events: frozenset[EventType],
    exclusive_path_events: frozenset[EventType],
    *,
    high_similarity_threshold: float = -1.0,
    min_event_compatibility: float = 0.1,
    downweighted_skill_ids: frozenset[str] = frozenset(),
    status_tie_epsilon: float = 0.0,
) -> TaskHighRetrieval:
    if not -1 <= high_similarity_threshold <= 1:
        raise ValueError("High similarity threshold must be in [-1, 1]")
    if not required_path_events <= exclusive_path_events:
        raise ValueError("required path events must be exclusive path events")
    if set(high_index.skill_ids) != set(high_cards):
        raise ValueError("High index and cards differ")

    matches = high_index.search(
        query_vector,
        len(high_cards),
        score_penalties={
            skill_id: status_tie_epsilon
            for skill_id in downweighted_skill_ids
            if skill_id in high_cards
        },
    )
    candidate = matches[0] if matches else None
    match = next(
        (
            item
            for item in matches
            if item.cosine_similarity >= high_similarity_threshold
            and frozenset(
                event
                for event in exclusive_path_events
                if any(
                    profile.compatibility(mid_id, frozenset((event,)))
                    >= min_event_compatibility
                    for mid_id in high_cards[item.skill_id].child_mid_ids
                )
            )
            == required_path_events
        ),
        None,
    )
    return TaskHighRetrieval(
        high_cards[match.skill_id] if match else None,
        candidate,
        match,
    )


def retrieve_tower_graph(
    high_query_vector: Sequence[float],
    state_query_vector: Sequence[float],
    high_index: SkillEmbeddingIndex,
    mid_index: SkillEmbeddingIndex,
    high_cards: Mapping[str, HighSkillCard],
    mid_cards: Mapping[str, MidSkillCard],
    high_paths: Mapping[str, HighPath],
    profile: TowerGraphProfile,
    allowed_events: frozenset[EventType],
    *,
    mid_context_budget: int,
    high_similarity_threshold: float = -1.0,
    direct_mid_similarity_threshold: float = 0.0,
    direct_mid_dedup_similarity_threshold: float = 0.92,
    direct_mid_mmr_lambda: float = 0.7,
    downweighted_skill_ids: frozenset[str] = frozenset(),
    status_tie_epsilon: float = 0.0,
    min_event_compatibility: float = 0.1,
    required_path_events: frozenset[EventType] = frozenset(),
    exclusive_path_events: frozenset[EventType] = frozenset(),
) -> GraphTowerRetrieval:
    if not 1 <= mid_context_budget <= 12:
        raise ValueError("Mid context budget must be in [1, 12]")
    if not allowed_events:
        raise ValueError("graph retrieval requires applicable current-state events")
    if not 0 < min_event_compatibility <= 1:
        raise ValueError("minimum event compatibility must be in (0, 1]")
    if not -1 <= direct_mid_similarity_threshold <= 1:
        raise ValueError("direct Mid similarity threshold must be in [-1, 1]")
    if not required_path_events <= exclusive_path_events:
        raise ValueError("required path events must be exclusive path events")
    if set(profile.mid_event_counts) != set(mid_cards):
        raise ValueError("graph profile and Mid cards differ")
    if set(high_cards) != set(high_paths):
        raise ValueError("High cards and paths differ")

    penalties = {skill_id: status_tie_epsilon for skill_id in downweighted_skill_ids}
    high_matches = high_index.search(
        high_query_vector,
        len(high_cards),
        score_penalties={
            skill_id: penalty for skill_id, penalty in penalties.items() if skill_id in high_cards
        },
    )
    state_matches = mid_index.search(
        state_query_vector,
        len(mid_cards),
        score_penalties={
            skill_id: penalty for skill_id, penalty in penalties.items() if skill_id in mid_cards
        },
    )
    high_similarities = {match.skill_id: match.cosine_similarity for match in high_matches}
    state_similarities = {match.skill_id: match.cosine_similarity for match in state_matches}

    graph_candidates = []
    for high_id, high_card in high_cards.items():
        if high_similarities[high_id] < high_similarity_threshold:
            continue
        represented_path_events = frozenset(
            event
            for event in exclusive_path_events
            if any(
                profile.compatibility(mid_id, frozenset((event,)))
                >= min_event_compatibility
            for mid_id in high_card.child_mid_ids
            )
        )
        if represented_path_events != required_path_events:
            continue
        nodes = []
        for mid_id in high_card.child_mid_ids:
            compatibility = profile.compatibility(mid_id, allowed_events)
            if compatibility < min_event_compatibility:
                continue
            nodes.append(
                (
                    0.65 * state_similarities[mid_id] + 0.35 * compatibility,
                    mid_id,
                    compatibility,
                )
            )
        if not nodes:
            continue
        _, active_mid_id, compatibility = max(nodes, key=lambda item: (item[0], item[1]))
        path = high_paths[high_id]
        path_quality = min(1.0, max(0.0, path.contrastive_score))
        score = (
            0.35 * high_similarities[high_id]
            + 0.35 * state_similarities[active_mid_id]
            + 0.20 * compatibility
            + 0.10 * path_quality
            - penalties.get(high_id, 0.0)
        )
        graph_candidates.append(
            GraphHighMatch(
                skill_id=high_id,
                score=score,
                goal_similarity=high_similarities[high_id],
                state_similarity=state_similarities[active_mid_id],
                event_compatibility=compatibility,
                path_quality=path_quality,
                active_mid_id=active_mid_id,
            )
        )
    ranked_highs = tuple(sorted(graph_candidates, key=lambda item: (-item.score, item.skill_id)))
    graph_high = ranked_highs[0] if ranked_highs else None
    high_card = high_cards[graph_high.skill_id] if graph_high else None

    selected_mid_ids = []
    if high_card and graph_high:
        active_index = high_card.child_mid_ids.index(graph_high.active_mid_id)
        selected_mid_ids.extend(
            mid_id
            for mid_id in high_card.child_mid_ids[active_index:]
            if profile.compatibility(mid_id, allowed_events)
            >= min_event_compatibility
        )
        selected_mid_ids = selected_mid_ids[:2]
    selected_mid_ids = list(dict.fromkeys(selected_mid_ids))[:mid_context_budget]

    allowed_mid_ids = {
        mid_id
        for mid_id in mid_cards
        if profile.compatibility(mid_id, allowed_events) >= min_event_compatibility
        and frozenset(
            event
            for event in exclusive_path_events
            if profile.compatibility(mid_id, frozenset((event,)))
            >= min_event_compatibility
        )
        <= required_path_events
    }
    allowed_index = mid_index.subset(allowed_mid_ids)
    remaining_budget = mid_context_budget - len(selected_mid_ids)
    diverse = (
        diverse_search(
            allowed_index,
            state_query_vector,
            candidate_count=len(allowed_mid_ids),
            similarity_threshold=direct_mid_similarity_threshold,
            relative_margin=2.0,
            dedup_similarity_threshold=direct_mid_dedup_similarity_threshold,
            relevance_weight=direct_mid_mmr_lambda,
            max_count=max(1, remaining_budget),
            score_penalties={
                skill_id: penalty
                for skill_id, penalty in penalties.items()
                if skill_id in allowed_mid_ids
            },
        )
        if allowed_mid_ids and remaining_budget
        else None
    )
    direct_matches = diverse.selected if diverse else ()
    for match in direct_matches:
        if len(selected_mid_ids) >= mid_context_budget:
            break
        if match.skill_id not in selected_mid_ids:
            selected_mid_ids.append(match.skill_id)

    selected_mid_cards = tuple(mid_cards[mid_id] for mid_id in selected_mid_ids)
    high_candidate = high_matches[0] if high_matches else None
    high_match = (
        next(
            match for match in high_matches if graph_high and match.skill_id == graph_high.skill_id
        )
        if graph_high
        else None
    )
    skill_ids = ((high_card.skill_id,) if high_card else ()) + tuple(
        card.skill_id for card in selected_mid_cards
    )
    retrieval = TowerRetrieval(
        high_card=high_card,
        mid_cards=selected_mid_cards,
        high_candidate=high_candidate,
        high_match=high_match,
        direct_mid_candidates=tuple(state_matches),
        direct_mid_filtered=(diverse.filtered if diverse else ()),
        direct_mid_deduplicated=(diverse.deduplicated if diverse else ()),
        direct_mid_matches=tuple(direct_matches),
        skill_ids=skill_ids,
        context_skill_ids=skill_ids,
        context=format_tower_context(high_card, selected_mid_cards),
    )
    return GraphTowerRetrieval(retrieval, ranked_highs, graph_high)
