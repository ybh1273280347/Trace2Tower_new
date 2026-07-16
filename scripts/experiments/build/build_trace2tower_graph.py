from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
)
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.graph import build_graph, ordered_segment_groups
from trace2tower.methods.trace2tower.models import SegmentInstance, event_type_from_value
from trace2tower.methods.trace2tower.spectral import (
    semantic_only_clustering,
    separate_exclusive_event_clusters,
    spectral_clustering,
)


def load_segment_groups(
    path: Path,
) -> tuple[Benchmark, tuple[tuple[SegmentInstance, ...], ...]]:
    benchmark = None
    groups = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            current_benchmark = Benchmark(record["benchmark"])
            if benchmark is not None and current_benchmark is not benchmark:
                raise ValueError("preprocessed graph input mixes benchmarks")
            benchmark = current_benchmark
            groups.append(
                tuple(_compact_segment(segment) for segment in record["segments"])
            )
    if benchmark is None:
        raise ValueError("graph input contains no trajectories")
    return benchmark, tuple(groups)


def _compact_segment(record: dict) -> SegmentInstance:
    event_type = record["event_type"]
    return SegmentInstance(
        segment_id=str(record["segment_id"]),
        trajectory_id=str(record["trajectory_id"]),
        start_step=int(record["start_step"]),
        end_step=int(record["end_step"]),
        transition_ids=tuple(record["transition_ids"]),
        embedding=np.asarray(record["embedding"], dtype=np.float32),
        trajectory_score=float(record["trajectory_score"]),
        event_type=event_type_from_value(event_type) if event_type is not None else None,
        raw_actions=tuple(record["raw_actions"]),
        observation_before=str(record["observation_before"]),
        observation_after=str(record["observation_after"]),
    )


def main(options: argparse.Namespace) -> int:
    config_record = load_yaml(options.config)
    config = Trace2TowerConfig.from_record(config_record)
    benchmark, groups = load_segment_groups(options.input)
    ordered_groups = ordered_segment_groups(groups)
    segments = tuple(segment for group in ordered_groups for segment in group)
    invocation = {
        "input": options.input.as_posix(),
        "benchmark": benchmark.value,
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
        embeddings = np.asarray([segment.embedding for segment in segments], dtype=np.float64)
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
        spectral_cluster_count = clustering.cluster_count
        if benchmark is Benchmark.ALFWORLD and not config.collapse_duplicate_embeddings:
            clustering = separate_exclusive_event_clusters(
                clustering,
                graph,
                ALFWORLD_EXCLUSIVE_PATH_EVENTS,
            )
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
    cluster_sizes = {
        index: len(cluster.member_segment_ids)
        for index, cluster in enumerate(clustering.clusters)
    }
    event_distribution = Counter(
        segment.event_type.value for segment in segments if segment.event_type is not None
    )
    report = {
        **invocation,
        "method": config.method.value,
        "exclusive_event_post_merge": (
            benchmark is Benchmark.ALFWORLD
            and not config.collapse_duplicate_embeddings
        ),
        "graph_node_count": len(graph.segment_ids) if graph else len(segments),
        "collapsed_segment_count": (
            len(segments) - len(graph.segment_ids) if graph else 0
        ),
        "cluster_count": clustering.cluster_count,
        "spectral_cluster_count": (
            spectral_cluster_count if not config.semantic_only else clustering.cluster_count
        ),
        "cluster_sizes": dict(sorted(cluster_sizes.items())),
        "eigenvalues": list(clustering.eigenvalues),
        "neighbor_count": graph.neighbor_count if graph else None,
        "edge_count": graph.edge_count if graph else None,
        "transition_edge_count": graph.transition_edge_count if graph else None,
        "cross_event_edge_count": graph.cross_event_edge_count if graph else None,
        "event_distribution": dict(sorted(event_distribution.items())),
        "transition_weight_count": (len(np.unique(graph.transition.data)) if graph else None),
        "connected_component_count": (
            connected_components(abs(graph.adjacency), directed=False)[0] if graph else None
        ),
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
