from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
from scipy import sparse

from trace2tower.core.tower_models import WebShopEventType

PRICE_PATTERN = re.compile(r"(?:Price:\s*|\|\s*\$)(\d+(?:\.\d+)?)", re.IGNORECASE)
SELECTED_PATTERN = re.compile(r"Selected:\s*\{(?P<values>.*?)\}", re.IGNORECASE | re.DOTALL)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class CountBucket(StrEnum):
    ZERO = "zero"
    ONE = "one"
    MULTIPLE = "multiple"


class PriceSignal(StrEnum):
    UNKNOWN = "unknown"
    WITHIN_LIMIT = "within_limit"
    OVER_LIMIT = "over_limit"


class RewardBand(StrEnum):
    LOW = "low"
    PARTIAL = "partial"
    EXACT = "exact"


class DecisionSignal(StrEnum):
    QUERY_BROAD = "query_broad"
    QUERY_CONSTRAINED = "query_constrained"
    QUERY_UNAVAILABLE = "query_unavailable"
    CANDIDATE_WEAK = "candidate_weak"
    CANDIDATE_CATEGORY = "candidate_category"
    CANDIDATE_CONSTRAINED = "candidate_constrained"
    OPTION_REQUIRED = "option_required"
    OPTION_OTHER = "option_other"
    OPTION_COMPLETE = "option_complete"
    INSPECT_DESCRIPTION = "inspect_description"
    INSPECT_FEATURES = "inspect_features"
    INSPECT_ATTRIBUTES = "inspect_attributes"
    INSPECT_REVIEWS = "inspect_reviews"
    RESULT_NAVIGATION = "result_navigation"
    DETAIL_BACKTRACK = "detail_backtrack"
    SEARCH_BACKTRACK = "search_backtrack"
    PURCHASE = "purchase"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class WebShopGoal:
    goal_index: int
    query: str
    attributes: tuple[str, ...]
    options: tuple[str, ...]
    price_upper: float | None

    @classmethod
    def from_record(cls, record: Mapping) -> WebShopGoal:
        price = record.get("price_upper")
        return cls(
            goal_index=int(record["goal_index"]),
            query=str(record.get("query", "")),
            attributes=tuple(str(value) for value in record.get("attributes", ())),
            options=tuple(str(value) for value in record.get("goal_options", ())),
            price_upper=float(price) if price is not None else None,
        )


@dataclass(frozen=True, slots=True)
class WebShopDecisionKey:
    event_type: WebShopEventType
    attribute_count: CountBucket
    option_count: CountBucket
    requires_price_check: bool
    signal: DecisionSignal
    price_signal: PriceSignal
    repeated_event: bool

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WebShopBranchNode:
    node_id: str
    key: WebShopDecisionKey
    member_segment_ids: tuple[str, ...]
    support_count: int
    mean_reward: float
    reward_distribution: tuple[float, float, float]

    def to_record(self) -> dict:
        return {
            "node_id": self.node_id,
            "key": self.key.to_record(),
            "member_segment_ids": self.member_segment_ids,
            "support_count": self.support_count,
            "mean_reward": self.mean_reward,
            "reward_distribution": self.reward_distribution,
        }

    @classmethod
    def from_record(cls, record: Mapping) -> WebShopBranchNode:
        key = record["key"]
        return cls(
            node_id=str(record["node_id"]),
            key=WebShopDecisionKey(
                event_type=WebShopEventType(key["event_type"]),
                attribute_count=CountBucket(key["attribute_count"]),
                option_count=CountBucket(key["option_count"]),
                requires_price_check=bool(key["requires_price_check"]),
                signal=DecisionSignal(key["signal"]),
                price_signal=PriceSignal(key["price_signal"]),
                repeated_event=bool(key["repeated_event"]),
            ),
            member_segment_ids=tuple(record["member_segment_ids"]),
            support_count=int(record["support_count"]),
            mean_reward=float(record["mean_reward"]),
            reward_distribution=tuple(float(value) for value in record["reward_distribution"]),
        )


@dataclass(frozen=True, slots=True)
class WebShopBranchEdge:
    source_node_id: str
    target_node_id: str
    support_count: int
    semantic_similarity: float
    transition_strength: float
    outcome_consistency: float
    weight: float
    mean_reward: float | None
    reward_distribution: tuple[float, float, float]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> WebShopBranchEdge:
        return cls(
            source_node_id=str(record["source_node_id"]),
            target_node_id=str(record["target_node_id"]),
            support_count=int(record["support_count"]),
            semantic_similarity=float(record["semantic_similarity"]),
            transition_strength=float(record["transition_strength"]),
            outcome_consistency=float(record["outcome_consistency"]),
            weight=float(record["weight"]),
            mean_reward=(
                float(record["mean_reward"]) if record.get("mean_reward") is not None else None
            ),
            reward_distribution=tuple(float(value) for value in record["reward_distribution"]),
        )


@dataclass(frozen=True, slots=True)
class WebShopBranchGraph:
    nodes: tuple[WebShopBranchNode, ...]
    edges: tuple[WebShopBranchEdge, ...]
    trajectory_count: int
    exact_trajectory_count: int
    partial_trajectory_count: int
    low_trajectory_count: int

    def to_record(self) -> dict:
        return {
            "graph_type": "webshop_constraint_branch_graph_v1",
            "trajectory_count": self.trajectory_count,
            "exact_trajectory_count": self.exact_trajectory_count,
            "partial_trajectory_count": self.partial_trajectory_count,
            "low_trajectory_count": self.low_trajectory_count,
            "nodes": [node.to_record() for node in self.nodes],
            "edges": [edge.to_record() for edge in self.edges],
        }

    @classmethod
    def from_record(cls, record: Mapping) -> WebShopBranchGraph:
        if record.get("graph_type") != "webshop_constraint_branch_graph_v1":
            raise ValueError("unsupported WebShop branch graph type")
        return cls(
            nodes=tuple(WebShopBranchNode.from_record(item) for item in record["nodes"]),
            edges=tuple(WebShopBranchEdge.from_record(item) for item in record["edges"]),
            trajectory_count=int(record["trajectory_count"]),
            exact_trajectory_count=int(record["exact_trajectory_count"]),
            partial_trajectory_count=int(record["partial_trajectory_count"]),
            low_trajectory_count=int(record["low_trajectory_count"]),
        )


@dataclass(slots=True)
class _NodeAccumulator:
    segment_ids: list[str]
    rewards: list[float]
    reward_bands: Counter[RewardBand]


@dataclass(slots=True)
class _EdgeAccumulator:
    rewards: list[float]
    reward_bands: Counter[RewardBand]


def build_webshop_branch_graph(
    records: Iterable[Mapping],
    goals: Mapping[int, WebShopGoal],
) -> WebShopBranchGraph:
    node_data: dict[WebShopDecisionKey, _NodeAccumulator] = {}
    transition_data: dict[tuple[WebShopDecisionKey, WebShopDecisionKey], _EdgeAccumulator] = {}
    source_counts: Counter[WebShopDecisionKey] = Counter()
    trajectory_bands: Counter[RewardBand] = Counter()
    trajectory_count = 0

    for record in records:
        sample_id = str(record["sample_id"])
        goal_index = _goal_index(sample_id)
        if goal_index not in goals:
            raise ValueError(f"WebShop goal is missing for {sample_id}")
        goal = goals[goal_index]
        reward = float(record["primary_score"])
        band = _reward_band(reward)
        trajectory_bands[band] += 1
        trajectory_count += 1

        event_counts: Counter[WebShopEventType] = Counter()
        keys = []
        for segment in sorted(record["segments"], key=lambda item: int(item["start_step"])):
            event = WebShopEventType(segment["event_type"])
            key = _decision_key(segment, goal, repeated_event=event_counts[event] > 0)
            event_counts[event] += 1
            keys.append(key)
            accumulator = node_data.setdefault(key, _NodeAccumulator([], [], Counter()))
            accumulator.segment_ids.append(str(segment["segment_id"]))
            accumulator.rewards.append(reward)
            accumulator.reward_bands[band] += 1

        for source, target in zip(keys, keys[1:]):
            edge = transition_data.setdefault((source, target), _EdgeAccumulator([], Counter()))
            edge.rewards.append(reward)
            edge.reward_bands[band] += 1
            source_counts[source] += 1

    nodes = tuple(
        _build_node(key, node_data[key]) for key in sorted(node_data, key=_decision_key_text)
    )
    nodes_by_key = {node.key: node for node in nodes}
    candidate_edges = set(transition_data)
    for left in nodes:
        for right in nodes:
            if left.node_id == right.node_id:
                continue
            semantic = _semantic_similarity(left.key, right.key)
            if left.key.event_type is right.key.event_type and semantic >= 0.75:
                candidate_edges.add((left.key, right.key))

    edges = []
    for source, target in candidate_edges:
        source_node = nodes_by_key[source]
        target_node = nodes_by_key[target]
        edge_data = transition_data.get((source, target))
        support = len(edge_data.rewards) if edge_data else 0
        semantic = _semantic_similarity(source, target)
        transition = support / source_counts[source] if source_counts[source] else 0.0
        outcome = _outcome_consistency(
            source_node.reward_distribution,
            target_node.reward_distribution,
        )
        weight = (
            0.4 * semantic + 0.4 * transition + 0.2 * outcome
            if support
            else 0.6 * semantic + 0.4 * outcome
        )
        reward_distribution = (
            tuple(
                edge_data.reward_bands[band] / support
                for band in (RewardBand.LOW, RewardBand.PARTIAL, RewardBand.EXACT)
            )
            if edge_data
            else (0.0, 0.0, 0.0)
        )
        edges.append(
            WebShopBranchEdge(
                source_node.node_id,
                target_node.node_id,
                support,
                semantic,
                transition,
                outcome,
                weight,
                sum(edge_data.rewards) / support if edge_data else None,
                reward_distribution,
            )
        )

    return WebShopBranchGraph(
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda edge: (edge.source_node_id, edge.target_node_id))),
        trajectory_count=trajectory_count,
        exact_trajectory_count=trajectory_bands[RewardBand.EXACT],
        partial_trajectory_count=trajectory_bands[RewardBand.PARTIAL],
        low_trajectory_count=trajectory_bands[RewardBand.LOW],
    )


def load_webshop_goals(records: Iterable[Mapping]) -> dict[int, WebShopGoal]:
    goals = tuple(WebShopGoal.from_record(record) for record in records)
    by_index = {goal.goal_index: goal for goal in goals}
    if len(by_index) != len(goals):
        raise ValueError("WebShop goals contain duplicate indices")
    return by_index


def webshop_branch_graph_components(
    graph: WebShopBranchGraph,
    *,
    failure_penalty: float,
):
    from trace2tower.graph.graph import GraphComponents

    if failure_penalty < 0:
        raise ValueError("failure penalty must be non-negative")
    nodes = tuple(sorted(graph.nodes, key=lambda node: node.node_id))
    node_indices = {node.node_id: index for index, node in enumerate(nodes)}
    embeddings = np.asarray([_decision_vector(node.key) for node in nodes], dtype=np.float64)
    rows = []
    columns = []
    semantic_values = []
    transition_values = []
    outcome_values = []
    base_values = []
    positive_values = []
    negative_values = []
    adjacency_values = []
    observed_edge_count = 0
    for edge in graph.edges:
        source = node_indices[edge.source_node_id]
        target = node_indices[edge.target_node_id]
        source_node = nodes[source]
        target_node = nodes[target]
        if edge.support_count:
            low, partial, exact = edge.reward_distribution
            observed_edge_count += 1
        else:
            source_low, source_partial, source_exact = source_node.reward_distribution
            target_low, target_partial, target_exact = target_node.reward_distribution
            low = math.sqrt(source_low * target_low)
            partial = math.sqrt(source_partial * target_partial)
            exact = math.sqrt(source_exact * target_exact)
        positive = edge.weight * (exact + 0.5 * partial)
        negative = edge.weight * low
        adjacency = positive - failure_penalty * negative

        rows.append(source)
        columns.append(target)
        semantic_values.append(edge.semantic_similarity)
        transition_values.append(edge.transition_strength)
        outcome_values.append(edge.outcome_consistency)
        base_values.append(edge.weight)
        positive_values.append(positive)
        negative_values.append(negative)
        adjacency_values.append(adjacency)

    shape = (len(nodes), len(nodes))
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
    matrices = [((matrix + matrix.T) * 0.5).tocsr() for matrix in matrices]
    for matrix in matrices:
        matrix.eliminate_zeros()
    semantic, transition, outcome, base, positive, negative, adjacency = matrices
    absolute_degree = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    inverse_sqrt_degree = np.zeros_like(absolute_degree)
    nonzero = absolute_degree > 0
    inverse_sqrt_degree[nonzero] = 1 / np.sqrt(absolute_degree[nonzero])
    scaling = sparse.diags(inverse_sqrt_degree)
    laplacian = (sparse.eye(len(nodes), format="csr") - scaling @ adjacency @ scaling).tocsr()
    return GraphComponents(
        segment_ids=tuple(node.node_id for node in nodes),
        event_types=tuple(node.key.event_type for node in nodes),
        segment_embeddings=embeddings,
        rho=np.asarray([node.mean_reward for node in nodes], dtype=np.float64),
        semantic=semantic,
        transition=transition,
        outcome=outcome,
        base=base,
        positive=positive,
        negative=negative,
        adjacency=adjacency,
        laplacian=laplacian,
        neighbor_count=0,
        edge_count=len(graph.edges),
        transition_edge_count=observed_edge_count,
        cross_event_edge_count=sum(
            nodes[node_indices[edge.source_node_id]].key.event_type
            is not nodes[node_indices[edge.target_node_id]].key.event_type
            for edge in graph.edges
        ),
        node_member_segment_ids=tuple(node.member_segment_ids for node in nodes),
    )


def _decision_vector(key: WebShopDecisionKey) -> tuple[float, ...]:
    values = []
    for vocabulary, selected in (
        (tuple(WebShopEventType), key.event_type),
        (tuple(CountBucket), key.attribute_count),
        (tuple(CountBucket), key.option_count),
        (tuple(DecisionSignal), key.signal),
        (tuple(PriceSignal), key.price_signal),
    ):
        values.extend(float(value is selected) for value in vocabulary)
    values.extend((float(key.requires_price_check), float(key.repeated_event)))
    return tuple(values)


def _decision_key(
    segment: Mapping,
    goal: WebShopGoal,
    *,
    repeated_event: bool,
) -> WebShopDecisionKey:
    event = WebShopEventType(segment["event_type"])
    before = str(segment["observation_before"])
    after = str(segment["observation_after"])
    action_value = _action_value(segment)
    visible_text = f"{before}\n{after}".casefold()
    signal = _decision_signal(event, action_value, before, after, goal)
    return WebShopDecisionKey(
        event_type=event,
        attribute_count=_count_bucket(len(goal.attributes)),
        option_count=_count_bucket(len(goal.options)),
        requires_price_check=goal.price_upper is not None,
        signal=signal,
        price_signal=_price_signal(visible_text, goal.price_upper),
        repeated_event=repeated_event,
    )


def _decision_signal(
    event: WebShopEventType,
    action_value: str,
    before: str,
    after: str,
    goal: WebShopGoal,
) -> DecisionSignal:
    if event in {WebShopEventType.QUERY_FORMULATION, WebShopEventType.QUERY_REFINEMENT}:
        if "search is not available" in after.casefold():
            return DecisionSignal.QUERY_UNAVAILABLE
        constraints = (*goal.attributes, *goal.options)
        covered = sum(_phrase_visible(value, action_value) for value in constraints)
        return DecisionSignal.QUERY_CONSTRAINED if covered else DecisionSignal.QUERY_BROAD
    if event is WebShopEventType.CANDIDATE_SELECTION:
        product_terms = set(_tokens(goal.query))
        after_terms = set(_tokens(after.splitlines()[0] if after else ""))
        category_match = bool(product_terms & after_terms)
        constraint_match = any(
            _phrase_visible(value, after) for value in (*goal.attributes, *goal.options)
        )
        if category_match and constraint_match:
            return DecisionSignal.CANDIDATE_CONSTRAINED
        if category_match:
            return DecisionSignal.CANDIDATE_CATEGORY
        return DecisionSignal.CANDIDATE_WEAK
    if event is WebShopEventType.OPTION_SELECTION:
        selected_count = _selected_count(after)
        if goal.options and selected_count >= len(goal.options):
            return DecisionSignal.OPTION_COMPLETE
        return (
            DecisionSignal.OPTION_REQUIRED
            if any(_phrase_visible(action_value, option) for option in goal.options)
            else DecisionSignal.OPTION_OTHER
        )
    if event is WebShopEventType.ATTRIBUTE_INSPECTION:
        return {
            "description": DecisionSignal.INSPECT_DESCRIPTION,
            "features": DecisionSignal.INSPECT_FEATURES,
            "attributes": DecisionSignal.INSPECT_ATTRIBUTES,
            "reviews": DecisionSignal.INSPECT_REVIEWS,
        }.get(action_value.casefold(), DecisionSignal.OTHER)
    return {
        WebShopEventType.RESULT_NAVIGATION: DecisionSignal.RESULT_NAVIGATION,
        WebShopEventType.DETAIL_BACKTRACKING: DecisionSignal.DETAIL_BACKTRACK,
        WebShopEventType.SEARCH_BACKTRACKING: DecisionSignal.SEARCH_BACKTRACK,
        WebShopEventType.PURCHASE_DECISION: DecisionSignal.PURCHASE,
    }.get(event, DecisionSignal.OTHER)


def _action_value(segment: Mapping) -> str:
    values = []
    for raw_action in segment.get("raw_actions", ()):
        try:
            action = json.loads(raw_action)
        except json.JSONDecodeError:
            continue
        arguments = action.get("arguments") or {}
        value = arguments.get("keywords", arguments.get("value", ""))
        if isinstance(value, str):
            values.append(value)
    return " ".join(values)


def _price_signal(text: str, price_upper: float | None) -> PriceSignal:
    if price_upper is None:
        return PriceSignal.UNKNOWN
    prices = tuple(float(match) for match in PRICE_PATTERN.findall(text))
    if not prices:
        return PriceSignal.UNKNOWN
    return PriceSignal.WITHIN_LIMIT if min(prices) <= price_upper else PriceSignal.OVER_LIMIT


def _selected_count(observation: str) -> int:
    match = SELECTED_PATTERN.search(observation)
    return match.group("values").count(":") if match else 0


def _phrase_visible(phrase: str, text: str) -> bool:
    phrase_tokens = set(_tokens(phrase))
    return bool(phrase_tokens) and phrase_tokens <= set(_tokens(text))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(text.casefold()))


def _count_bucket(count: int) -> CountBucket:
    if count == 0:
        return CountBucket.ZERO
    if count == 1:
        return CountBucket.ONE
    return CountBucket.MULTIPLE


def _reward_band(reward: float) -> RewardBand:
    if reward >= 0.999:
        return RewardBand.EXACT
    if reward >= 0.5:
        return RewardBand.PARTIAL
    return RewardBand.LOW


def _build_node(
    key: WebShopDecisionKey,
    data: _NodeAccumulator,
) -> WebShopBranchNode:
    node_id = f"webnode_{hashlib.sha256(_decision_key_text(key).encode()).hexdigest()[:12]}"
    total = len(data.rewards)
    distribution = tuple(
        data.reward_bands[band] / total
        for band in (RewardBand.LOW, RewardBand.PARTIAL, RewardBand.EXACT)
    )
    return WebShopBranchNode(
        node_id=node_id,
        key=key,
        member_segment_ids=tuple(sorted(data.segment_ids)),
        support_count=total,
        mean_reward=sum(data.rewards) / total,
        reward_distribution=distribution,
    )


def _decision_key_text(key: WebShopDecisionKey) -> str:
    return json.dumps(key.to_record(), sort_keys=True, separators=(",", ":"))


def _semantic_similarity(left: WebShopDecisionKey, right: WebShopDecisionKey) -> float:
    signals = (
        left.event_type is right.event_type,
        left.attribute_count is right.attribute_count,
        left.option_count is right.option_count,
        left.requires_price_check == right.requires_price_check,
        left.signal is right.signal,
        left.price_signal is right.price_signal,
        left.repeated_event == right.repeated_event,
    )
    weights = (0.25, 0.1, 0.1, 0.1, 0.25, 0.1, 0.1)
    return sum(weight for matched, weight in zip(signals, weights, strict=True) if matched)


def _outcome_consistency(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _goal_index(sample_id: str) -> int:
    try:
        return int(sample_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid WebShop sample ID: {sample_id}") from exc
