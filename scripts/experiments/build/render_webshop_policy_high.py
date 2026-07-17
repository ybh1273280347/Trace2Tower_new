from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.webshop_branch_graph import DecisionSignal


STEP_TEXT = {
    DecisionSignal.QUERY_CONSTRAINED: (
        "Search with the requested product identity plus the most discriminative requested "
        "attribute or variant; omit the price ceiling from keywords."
    ),
    DecisionSignal.QUERY_BROAD: "Search for the requested product identity.",
    DecisionSignal.CANDIDATE_CONSTRAINED: (
        "Open the strongest plausible result and use only that product page's visible title, "
        "price, details, and options as evidence."
    ),
    DecisionSignal.CANDIDATE_CATEGORY: (
        "Open a category-matching result and check the current product page for the remaining "
        "requested constraints."
    ),
    DecisionSignal.CANDIDATE_WEAK: (
        "Open the best available plausible result; resolve uncertain constraints from the "
        "current product page instead of assuming that title wording is decisive."
    ),
    DecisionSignal.OPTION_COMPLETE: (
        "Select every requested variant exactly as shown, and confirm the selected state before "
        "continuing."
    ),
    DecisionSignal.OPTION_REQUIRED: (
        "Select the requested option value that is currently available on the product page."
    ),
    DecisionSignal.PURCHASE: (
        "Click Buy Now as soon as the current product's visible price and required selections "
        "satisfy the request."
    ),
}


RULE_TEXT = {
    (
        DecisionSignal.SEARCH_BACKTRACK,
        DecisionSignal.QUERY_CONSTRAINED,
        DecisionSignal.QUERY_BROAD,
    ): (
        "After abandoning a candidate, search again with a missing identifying constraint; do "
        "not restart with a broader query."
    ),
    (
        DecisionSignal.INSPECT_ATTRIBUTES,
        DecisionSignal.DETAIL_BACKTRACK,
        DecisionSignal.SEARCH_BACKTRACK,
    ): (
        "After reading a detail tab, return to the same product page to select options or buy; "
        "leave the candidate only when the observed detail contradicts the request."
    ),
}


def render_policy_high(policies: list[dict]) -> tuple[str, dict]:
    backbones = [
        tuple(DecisionSignal(value) for value in policy["backbone_signals"])
        for policy in policies
        if policy["backbone_signals"]
    ]
    observed_signals = {signal for backbone in backbones for signal in backbone}
    query_signal = (
        DecisionSignal.QUERY_CONSTRAINED
        if DecisionSignal.QUERY_CONSTRAINED in observed_signals
        else DecisionSignal.QUERY_BROAD
    )
    candidate_signals = {
        DecisionSignal.CANDIDATE_CONSTRAINED,
        DecisionSignal.CANDIDATE_CATEGORY,
        DecisionSignal.CANDIDATE_WEAK,
    }
    procedure = [STEP_TEXT[query_signal]]
    if observed_signals & candidate_signals:
        procedure.append(
            "Open the best plausible result. Treat title wording as a screening clue, then use "
            "only the current product page's visible price, details, and options as evidence."
        )
    if DecisionSignal.OPTION_REQUIRED in observed_signals:
        procedure.append(STEP_TEXT[DecisionSignal.OPTION_REQUIRED])
    if DecisionSignal.OPTION_COMPLETE in observed_signals:
        procedure.append(STEP_TEXT[DecisionSignal.OPTION_COMPLETE])
    procedure.append(STEP_TEXT[DecisionSignal.PURCHASE])
    procedure = tuple(procedure)

    rule_records = [rule for policy in policies for rule in policy["branch_rules"]]
    constraints = tuple(
        RULE_TEXT[key]
        for key in RULE_TEXT
        if any(
            key
            == (
                DecisionSignal(rule["source_signal"]),
                DecisionSignal(rule["preferred_next_signal"]),
                DecisionSignal(rule["avoided_next_signal"]),
            )
            for rule in rule_records
        )
    )
    constraints += (
        "Never claim that an unobserved product property is satisfied; use a relevant detail "
        "view only when that property is required and still unresolved.",
    )
    title = "Graph-discovered shopping policy"
    markdown = "\n".join(
        (
            f"## {title}",
            "",
            "Apply this compact policy to product-purchase tasks. Product facts come only from "
            "the current task and visible page; the policy supplies decision order, not facts.",
            "",
            "### Procedure",
            "",
            *(f"{index}. {step}" for index, step in enumerate(procedure, 1)),
            "",
            "### Recovery guards",
            "",
            *(f"- {constraint}" for constraint in constraints),
        )
    )
    audit = {
        "name": title,
        "procedure": procedure,
        "constraints": constraints,
        "policy_ids": tuple(policy["policy_id"] for policy in policies),
        "backbones": [
            {
                "option_count": policy["option_count"],
                "signals": policy["backbone_signals"],
                "exact_support": policy["backbone_exact_support"],
            }
            for policy in policies
        ],
        "branch_rule_count": len(rule_records),
    }
    return markdown, audit


def main(options: argparse.Namespace) -> int:
    policies = json.loads(options.policies.read_text(encoding="utf-8"))["policies"]
    markdown, audit = render_policy_high(policies)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(markdown + "\n", encoding="utf-8")
    write_json(options.audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
