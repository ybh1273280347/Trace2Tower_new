from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.adapters.webshop.branch_graph import WebShopBranchGraph
from trace2tower.methods.trace2tower.adapters.webshop.policy_discovery import (
    discover_webshop_policy_subgraphs,
)


def load_graph(path: Path) -> WebShopBranchGraph:
    return WebShopBranchGraph.from_record(json.loads(path.read_text(encoding="utf-8")))


def main(options: argparse.Namespace) -> int:
    policies = discover_webshop_policy_subgraphs(
        load_graph(options.graph),
        minimum_branch_support=options.minimum_branch_support,
        minimum_reward_gain=options.minimum_reward_gain,
        minimum_source_nodes=options.minimum_source_nodes,
    )
    write_json(options.output, {"policies": [policy.to_record() for policy in policies]})
    report = {
        "policy_count": len(policies),
        "policies": [
            {
                "policy_id": policy.policy_id,
                "support_count": policy.support_count,
                "observed_edge_count": policy.observed_edge_count,
                "branch_rule_count": len(policy.branch_rules),
            }
            for policy in policies
        ],
    }
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-branch-support", type=int, default=10)
    parser.add_argument("--minimum-reward-gain", type=float, default=0.1)
    parser.add_argument("--minimum-source-nodes", type=int, default=2)
    raise SystemExit(main(parser.parse_args()))
