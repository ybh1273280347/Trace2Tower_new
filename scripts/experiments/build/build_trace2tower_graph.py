from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.graph import build_graph, ordered_segment_groups
from trace2tower.methods.trace2tower.models import SegmentInstance
from trace2tower.methods.trace2tower.spectral import (
    semantic_only_clustering,
    spectral_clustering,
)


def main(options: argparse.Namespace) -> int:
    config_record = load_yaml(options.config)
    config = Trace2TowerConfig.from_record(config_record)
    records = [
        json.loads(line)
        for line in options.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    groups = tuple(
        tuple(SegmentInstance.from_record(segment) for segment in record["segments"])
        for record in records
    )
    ordered_groups = ordered_segment_groups(groups)
    segments = tuple(segment for group in ordered_groups for segment in group)
    invocation = {
        "input": options.input.as_posix(),
        "config": options.config.as_posix(),
        "output_dir": options.output_dir.as_posix(),
        "full_report": options.full_report.as_posix() if options.full_report else None,
        "trajectory_count": len(ordered_groups),
        "segment_count": len(segments),
        "dry_run": options.dry_run,
    }
    print(yaml.safe_dump({"method_config": config_record, "invocation": invocation}))
    if options.dry_run:
        return 0

    options.output_dir.mkdir(parents=True, exist_ok=True)
    if config.semantic_only:
        if options.full_report is None:
            raise ValueError("semantic-only clustering requires --full-report")
        full_report = json.loads(options.full_report.read_text(encoding="utf-8"))
        cluster_count = int(full_report["cluster_count"])
        segment_ids = tuple(segment.segment_id for segment in segments)
        embeddings = np.asarray(
            [segment.embedding for segment in segments], dtype=np.float64
        )
        clustering = semantic_only_clustering(
            segment_ids,
            embeddings,
            cluster_count=cluster_count,
            random_state=config.random_state,
        )
        graph = None
    else:
        graph = build_graph(ordered_groups, config)
        clustering = spectral_clustering(graph, config)
        for name in (
            "semantic",
            "transition",
            "outcome",
            "base",
            "adjacency",
            "laplacian",
        ):
            sparse.save_npz(options.output_dir / f"{name}.npz", getattr(graph, name))
        np.savez_compressed(
            options.output_dir / "spectral.npz",
            segment_ids=np.asarray(graph.segment_ids),
            rho=graph.rho,
            eigenvalues=np.asarray(clustering.eigenvalues),
            representation=clustering.representation,
            labels=np.asarray(clustering.labels),
        )

    write_json(
        options.output_dir / "clusters.json",
        {"clusters": [cluster.to_record() for cluster in clustering.clusters]},
    )
    cluster_sizes = Counter(clustering.labels)
    report = {
        **invocation,
        "method": config.method.value,
        "cluster_count": clustering.cluster_count,
        "cluster_sizes": dict(sorted(cluster_sizes.items())),
        "eigenvalues": list(clustering.eigenvalues),
        "neighbor_count": graph.neighbor_count if graph else None,
        "edge_count": graph.edge_count if graph else None,
        "rho_min": float(graph.rho.min()) if graph else None,
        "rho_max": float(graph.rho.max()) if graph else None,
        "positive_adjacency_entries": int((graph.adjacency.data > 0).sum()) if graph else None,
        "negative_adjacency_entries": int((graph.adjacency.data < 0).sum()) if graph else None,
        "config": config_record,
    }
    write_json(options.output_dir / "report.json", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
