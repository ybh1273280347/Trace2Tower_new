from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from math import prod

from trace2tower.adapters.webshop.branch_graph import (
    CountBucket,
    DecisionSignal,
    WebShopBranchGraph,
)


@dataclass(frozen=True, slots=True)
class WebShopBranchRule:
    source_signal: DecisionSignal
    preferred_next_signal: DecisionSignal
    avoided_next_signal: DecisionSignal
    source_node_count: int
    preferred_support: int
    avoided_support: int
    preferred_mean_reward: float
    avoided_mean_reward: float
    reward_gain: float

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WebShopPolicySubgraph:
    policy_id: str
    option_count: CountBucket
    member_node_ids: tuple[str, ...]
    observed_edge_count: int
    support_count: int
    backbone_signals: tuple[DecisionSignal, ...]
    backbone_exact_support: int
    branch_rules: tuple[WebShopBranchRule, ...]

    def to_record(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "option_count": self.option_count,
            "member_node_ids": self.member_node_ids,
            "observed_edge_count": self.observed_edge_count,
            "support_count": self.support_count,
            "backbone_signals": self.backbone_signals,
            "backbone_exact_support": self.backbone_exact_support,
            "branch_rules": [rule.to_record() for rule in self.branch_rules],
        }


def discover_webshop_policy_subgraphs(
    graph: WebShopBranchGraph,
    *,
    minimum_branch_support: int = 10,
    minimum_reward_gain: float = 0.1,
    minimum_source_nodes: int = 2,
) -> tuple[WebShopPolicySubgraph, ...]:
    if (
        minimum_branch_support <= 0
        or not 0 <= minimum_reward_gain <= 1
        or minimum_source_nodes <= 0
    ):
        raise ValueError("invalid WebShop policy discovery thresholds")
    nodes = {node.node_id: node for node in graph.nodes}
    policies = []
    for option_count in CountBucket:
        member_nodes = tuple(node for node in graph.nodes if node.key.option_count is option_count)
        member_ids = {node.node_id for node in member_nodes}
        observed_edges = tuple(
            edge
            for edge in graph.edges
            if edge.support_count
            and edge.source_node_id in member_ids
            and edge.target_node_id in member_ids
        )
        backbone_signals, backbone_exact_support = _discover_backbone(
            observed_edges,
            nodes,
        )
        outgoing = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        for edge in observed_edges:
            target_signal = nodes[edge.target_node_id].key.signal
            summary = outgoing[edge.source_node_id][target_signal]
            summary[0] += edge.support_count
            summary[1] += (edge.mean_reward or 0.0) * edge.support_count

        stable_rules = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
        for source_node_id, targets in outgoing.items():
            eligible = tuple(
                (
                    target_signal,
                    support,
                    reward_sum / support,
                )
                for target_signal, (support, reward_sum) in targets.items()
                if support >= 3
            )
            if len(eligible) < 2:
                continue
            preferred = max(eligible, key=lambda item: (item[2], item[1], item[0].value))
            avoided = min(eligible, key=lambda item: (item[2], -item[1], item[0].value))
            gain = preferred[2] - avoided[2]
            if gain < minimum_reward_gain:
                continue
            source_signal = nodes[source_node_id].key.signal
            summary = stable_rules[(source_signal, preferred[0], avoided[0])]
            summary[0] += 1
            summary[1] += preferred[1]
            summary[2] += avoided[1]
            summary[3] += preferred[2] * preferred[1]
            summary[4] += avoided[2] * avoided[1]

        rules = []
        for (
            source_signal,
            preferred_signal,
            avoided_signal,
        ), summary in stable_rules.items():
            source_node_count, preferred_support, avoided_support, preferred_sum, avoided_sum = (
                summary
            )
            if (
                source_node_count < minimum_source_nodes
                or preferred_support < minimum_branch_support
                or avoided_support < minimum_branch_support
            ):
                continue
            preferred_reward = preferred_sum / preferred_support
            avoided_reward = avoided_sum / avoided_support
            gain = preferred_reward - avoided_reward
            if gain < minimum_reward_gain:
                continue
            rules.append(
                WebShopBranchRule(
                    source_signal,
                    preferred_signal,
                    avoided_signal,
                    source_node_count,
                    preferred_support,
                    avoided_support,
                    preferred_reward,
                    avoided_reward,
                    gain,
                )
            )

        if member_nodes:
            policies.append(
                WebShopPolicySubgraph(
                    policy_id=f"webshop_policy_options_{option_count.value}",
                    option_count=option_count,
                    member_node_ids=tuple(sorted(member_ids)),
                    observed_edge_count=len(observed_edges),
                    support_count=sum(node.support_count for node in member_nodes),
                    backbone_signals=backbone_signals,
                    backbone_exact_support=backbone_exact_support,
                    branch_rules=tuple(
                        sorted(rules, key=lambda rule: (-rule.reward_gain, rule.source_signal))
                    ),
                )
            )
    return tuple(policies)


def _discover_backbone(observed_edges, nodes) -> tuple[tuple[DecisionSignal, ...], int]:
    # High 只保留成功边共同支持的短链，商品实例仍由执行时观测决定。
    exact_support = defaultdict(int)
    for edge in observed_edges:
        source = nodes[edge.source_node_id].key.signal
        target = nodes[edge.target_node_id].key.signal
        exact_support[(source, target)] += round(edge.support_count * edge.reward_distribution[2])

    starts = (DecisionSignal.QUERY_CONSTRAINED, DecisionSignal.QUERY_BROAD)
    candidates = []
    for start in starts:
        stack = [((start,), ())]
        while stack:
            path, supports = stack.pop()
            if path[-1] is DecisionSignal.PURCHASE:
                candidates.append((path, supports))
                continue
            if len(path) >= 7:
                continue
            for (source, target), support in exact_support.items():
                if source is path[-1] and support and target not in path:
                    stack.append((path + (target,), supports + (support,)))

    if not candidates:
        return (), 0
    path, supports = max(
        candidates,
        key=lambda item: (
            min(item[1]),
            prod(item[1]),
            -len(item[0]),
            tuple(signal.value for signal in item[0]),
        ),
    )
    return path, min(supports)
