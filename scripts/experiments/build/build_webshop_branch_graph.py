from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.webshop_branch_graph import (
    build_webshop_branch_graph,
    load_webshop_goals,
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def main(options: argparse.Namespace) -> int:
    goals = load_webshop_goals(json.loads(options.goals.read_text(encoding="utf-8")))
    graph = build_webshop_branch_graph(read_jsonl(options.preprocessed), goals)
    write_json(options.output, graph.to_record())
    report = {
        "graph_type": "webshop_constraint_branch_graph_v1",
        "trajectory_count": graph.trajectory_count,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "observed_branch_count": sum(edge.support_count > 0 for edge in graph.edges),
        "semantic_edge_count": sum(edge.support_count == 0 for edge in graph.edges),
        "exact_trajectory_count": graph.exact_trajectory_count,
        "partial_trajectory_count": graph.partial_trajectory_count,
        "low_trajectory_count": graph.low_trajectory_count,
    }
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--goals", type=Path, default=Path("Datasets/webshop/goals.json"))
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
