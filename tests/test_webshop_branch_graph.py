from __future__ import annotations

import json

from trace2tower.methods.trace2tower.webshop_branch_graph import (
    DecisionSignal,
    WebShopGoal,
    build_webshop_branch_graph,
    webshop_branch_graph_components,
)
from trace2tower.methods.trace2tower.webshop_policy_discovery import (
    discover_webshop_policy_subgraphs,
)


def segment(index: int, event: str, action: dict, before: str, after: str) -> dict:
    return {
        "segment_id": f"trajectory:segment:{index}",
        "start_step": index,
        "event_type": event,
        "raw_actions": [json.dumps(action)],
        "observation_before": before,
        "observation_after": after,
    }


def test_branch_graph_keeps_decisions_and_removes_product_identity() -> None:
    goals = {
        1: WebShopGoal(1, "running shoes", ("waterproof",), ("red",), 80.0),
        2: WebShopGoal(2, "coffee maker", ("programmable",), ("black",), 80.0),
    }
    records = (
        {
            "sample_id": "webshop:1",
            "primary_score": 1.0,
            "segments": (
                segment(
                    0,
                    "QUERY_FORMULATION",
                    {"name": "search_action", "arguments": {"keywords": "running shoes waterproof"}},
                    "WebShop search page.",
                    "Search results page 1:\nSHOE123 | Waterproof Running Shoes | $70.00",
                ),
                segment(
                    1,
                    "CANDIDATE_SELECTION",
                    {"name": "click_action", "arguments": {"value": "SHOE123"}},
                    "Search results page 1",
                    "Product: Waterproof Running Shoes\nASIN: SHOE123\nPrice: $70.00",
                ),
                segment(
                    2,
                    "OPTION_SELECTION",
                    {"name": "click_action", "arguments": {"value": "red"}},
                    "Product: Waterproof Running Shoes",
                    "Product: Waterproof Running Shoes\nSelected: {'color': 'red'}",
                ),
                segment(
                    3,
                    "PURCHASE_DECISION",
                    {"name": "click_action", "arguments": {"value": "Buy Now"}},
                    "Product: Waterproof Running Shoes\nPrice: $70.00",
                    "Purchase completed. Reward: 1.0",
                ),
            ),
        },
        {
            "sample_id": "webshop:2",
            "primary_score": 0.2,
            "segments": (
                segment(
                    0,
                    "QUERY_FORMULATION",
                    {"name": "search_action", "arguments": {"keywords": "appliance"}},
                    "WebShop search page.",
                    "Search results page 1:\nOTHER999 | Silver Toaster | $90.00",
                ),
                segment(
                    1,
                    "CANDIDATE_SELECTION",
                    {"name": "click_action", "arguments": {"value": "OTHER999"}},
                    "Search results page 1",
                    "Product: Silver Toaster\nASIN: OTHER999\nPrice: $90.00",
                ),
                segment(
                    2,
                    "SEARCH_BACKTRACKING",
                    {"name": "click_action", "arguments": {"value": "Back to Search"}},
                    "Product: Silver Toaster",
                    "WebShop search page.",
                ),
            ),
        },
    )

    graph = build_webshop_branch_graph(records, goals)
    payload = json.dumps(graph.to_record())
    signals = {node.key.signal for node in graph.nodes}

    assert graph.trajectory_count == 2
    assert graph.exact_trajectory_count == 1
    assert graph.low_trajectory_count == 1
    assert DecisionSignal.CANDIDATE_CONSTRAINED in signals
    assert DecisionSignal.CANDIDATE_WEAK in signals
    assert any(edge.support_count for edge in graph.edges)
    assert "Waterproof Running Shoes" not in payload
    assert "Silver Toaster" not in payload
    assert "SHOE123" not in payload
    assert "OTHER999" not in payload

    components = webshop_branch_graph_components(graph, failure_penalty=1.0)
    assert components.node_member_segment_ids
    assert components.adjacency.shape == (len(graph.nodes), len(graph.nodes))
    assert components.positive.nnz > 0


def test_policy_discovery_emits_success_backbone() -> None:
    goals = {1: WebShopGoal(1, "running shoes", ("waterproof",), ("red",), 80.0)}
    records = tuple(
        {
            "sample_id": "webshop:1",
            "primary_score": 1.0,
            "segments": (
                segment(
                    0,
                    "QUERY_FORMULATION",
                    {"name": "search_action", "arguments": {"keywords": "running shoes waterproof"}},
                    "WebShop search page.",
                    "Search results page 1",
                ),
                segment(
                    1,
                    "CANDIDATE_SELECTION",
                    {"name": "click_action", "arguments": {"value": "SHOE123"}},
                    "Search results page 1",
                    "Product: Waterproof Running Shoes\nPrice: $70.00",
                ),
                segment(
                    2,
                    "OPTION_SELECTION",
                    {"name": "click_action", "arguments": {"value": "red"}},
                    "Product: Waterproof Running Shoes",
                    "Product: Waterproof Running Shoes\nSelected: {'color': 'red'}",
                ),
                segment(
                    3,
                    "PURCHASE_DECISION",
                    {"name": "click_action", "arguments": {"value": "Buy Now"}},
                    "Product: Waterproof Running Shoes\nPrice: $70.00",
                    "Purchase completed. Reward: 1.0",
                ),
            ),
        }
        for _ in range(3)
    )

    policies = discover_webshop_policy_subgraphs(build_webshop_branch_graph(records, goals))
    policy = next(item for item in policies if item.option_count.value == "one")

    assert policy.backbone_signals[0] is DecisionSignal.QUERY_CONSTRAINED
    assert policy.backbone_signals[-1] is DecisionSignal.PURCHASE
    assert policy.backbone_exact_support == 3
