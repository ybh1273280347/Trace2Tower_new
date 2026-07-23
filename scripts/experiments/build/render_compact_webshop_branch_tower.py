from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.adapters.webshop.branch_graph import WebShopBranchGraph
from trace2tower.methods.trace2tower.core.models import HighCommunity, MidCluster, PrimitiveAction
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard, MidSkillCard


def main(options: argparse.Namespace) -> int:
    graph = WebShopBranchGraph.from_record(json.loads(options.graph.read_text(encoding="utf-8")))
    clusters = tuple(
        MidCluster.from_record(item)
        for item in json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    )
    structure = json.loads(options.communities.read_text(encoding="utf-8"))
    communities = tuple(HighCommunity.from_record(item) for item in structure["communities"])
    node_by_segment = {
        segment_id: node for node in graph.nodes for segment_id in node.member_segment_ids
    }
    signal_counts = {
        cluster.cluster_id: Counter(
            node_by_segment[segment_id].key.signal.value
            for segment_id in cluster.member_segment_ids
        )
        for cluster in clusters
    }
    main_id = max(signal_counts, key=lambda mid_id: signal_counts[mid_id]["purchase"])
    recovery_id = max(
        (mid_id for mid_id in signal_counts if mid_id != main_id),
        key=lambda mid_id: signal_counts[mid_id]["search_backtrack"],
    )
    evidence_id = next(mid_id for mid_id in signal_counts if mid_id not in {main_id, recovery_id})
    clusters_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    cards = {
        main_id: MidSkillCard(
            main_id,
            clusters_by_id[main_id].member_segment_ids,
            "Screen a candidate, bind variants, and purchase",
            "Use after search exposes a plausible product candidate.",
            (
                "Open the best plausible result and judge only the current product page.",
                "Select every requested variant exactly as shown and confirm the selected state.",
                "Click Buy Now once visible price and required selections satisfy the request.",
            ),
            (
                "Do not infer a product property from another candidate or from title "
                "similarity alone.",
                "If an exact requested option is unavailable, leave the candidate "
                "instead of substituting.",
            ),
            (PrimitiveAction.CLICK,),
        ),
        evidence_id: MidSkillCard(
            evidence_id,
            clusters_by_id[evidence_id].member_segment_ids,
            "Search with constraints and resolve one missing fact",
            "Use at task start or when one required property remains unverified.",
            (
                "Search with the product identity plus the most discriminative "
                "requested constraint.",
                "Use one relevant detail view only for a required property that is "
                "still unresolved.",
                "Return to the same product page after reading that detail.",
            ),
            (
                "Do not put the price ceiling into search keywords.",
                "Do not browse extra tabs after all required facts are visible.",
            ),
            (PrimitiveAction.SEARCH, PrimitiveAction.CLICK),
        ),
        recovery_id: MidSkillCard(
            recovery_id,
            clusters_by_id[recovery_id].member_segment_ids,
            "Recover from a mismatched candidate or weak result set",
            "Use only after the current candidate contradicts the request or no "
            "plausible result remains.",
            (
                "Return to a page where search is available.",
                "Revise the query with a missing identifying constraint rather than broadening it.",
                "Open an untried plausible candidate and resume normal verification.",
            ),
            (
                "Do not repeat an identical query or reopen a rejected candidate.",
                "Do not abandon a candidate merely because a detail view must return "
                "to its product page.",
            ),
            (PrimitiveAction.CLICK, PrimitiveAction.SEARCH),
        ),
    }
    procedure = (
        "Search with the requested product identity plus the most discriminative "
        "requested attribute or variant.",
        "Open the best plausible result and use only the current product page as evidence.",
        "Select every requested variant exactly as shown and confirm the selected state.",
        "Click Buy Now as soon as visible price and required selections satisfy the request.",
    )
    common_guard = (
        "Never claim that an unobserved product property is satisfied; inspect only "
        "a required unresolved property.",
    )
    recovery_guards = (
        "After rejecting a candidate, search again with a missing identifying "
        "constraint rather than a broader query.",
        "After reading a detail tab, return to the same product page unless the "
        "detail contradicts the request.",
    )
    high_cards = []
    for community in communities:
        recovery = recovery_id in community.member_mid_ids
        high_cards.append(
            HighSkillCard(
                skill_id=community.community_id,
                ordered_mid_ids=(),
                name=(
                    "Search, verify, recover when needed, and buy"
                    if recovery
                    else "Search, verify, select, and buy directly"
                ),
                description=(
                    "Use for shopping tasks whose search or candidate may require one "
                    "targeted recovery."
                    if recovery
                    else "Use for shopping tasks that can proceed from a constrained "
                    "search to a plausible candidate."
                ),
                procedure=procedure,
                constraints=common_guard + (recovery_guards if recovery else ()),
                member_mid_ids=community.member_mid_ids,
            )
        )
    payload = {
        "mid_cards": [cards[cluster.cluster_id].to_record() for cluster in clusters],
        "high_cards": [card.to_record() for card in high_cards],
        "usage": [],
        "role_assignment": {
            "main_mid_id": main_id,
            "evidence_mid_id": evidence_id,
            "recovery_mid_id": recovery_id,
        },
    }
    write_json(options.output, payload)
    print(json.dumps(payload["role_assignment"], indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--communities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
