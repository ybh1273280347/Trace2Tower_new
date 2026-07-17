from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.spectral import spectral_clustering
from trace2tower.methods.trace2tower.webshop_branch_graph import (
    WebShopBranchGraph,
    webshop_branch_graph_components,
)


def main(options: argparse.Namespace) -> int:
    config = Trace2TowerConfig.from_record(load_yaml(options.config))
    graph = WebShopBranchGraph.from_record(
        json.loads(options.graph.read_text(encoding="utf-8"))
    )
    components = webshop_branch_graph_components(
        graph,
        failure_penalty=config.failure_penalty,
    )
    clustering = spectral_clustering(components, config)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "semantic",
        "transition",
        "outcome",
        "base",
        "positive",
        "negative",
        "adjacency",
        "laplacian",
    ):
        sparse.save_npz(options.output_dir / f"{name}.npz", getattr(components, name))
    np.savez_compressed(
        options.output_dir / "spectral.npz",
        node_ids=np.asarray(components.segment_ids),
        rho=components.rho,
        eigenvalues=np.asarray(clustering.eigenvalues),
        representation=clustering.representation,
        labels=np.asarray(clustering.labels),
    )
    write_json(
        options.output_dir / "clusters.json",
        {"clusters": [cluster.to_record() for cluster in clustering.clusters]},
    )
    report = {
        "graph": options.graph.as_posix(),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "observed_edge_count": sum(edge.support_count > 0 for edge in graph.edges),
        "mid_community_count": clustering.cluster_count,
        "cluster_sizes": [
            len(cluster.member_segment_ids) for cluster in clustering.clusters
        ],
        "eigenvalues": clustering.eigenvalues,
    }
    write_json(options.output_dir / "report.json", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
