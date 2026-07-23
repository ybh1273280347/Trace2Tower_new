from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot
from trace2tower.methods.trace2tower.core.models import MidCluster
from trace2tower.methods.trace2tower.deployment_optimization.lineage import build_mid_lineage
from trace2tower.methods.trace2tower.deployment_optimization.models import RefinementAction
from trace2tower.methods.trace2tower.deployment_optimization.structural import (
    SegmentStructuralEvidence,
    build_structural_proposals,
)


def main(options: argparse.Namespace) -> int:
    snapshot = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    new_clusters = tuple(
        MidCluster.from_record(record)
        for record in json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    )
    lineage = build_mid_lineage(
        snapshot.mid_clusters,
        new_clusters,
        min_old_retention=options.min_overlap_fraction,
        min_new_historical_purity=options.min_overlap_fraction,
    )
    historical_ids = frozenset(
        segment_id
        for cluster in snapshot.mid_clusters
        for segment_id in cluster.member_segment_ids
    )
    evidence = _read_evidence(options.preprocessed, historical_ids)
    proposals = build_structural_proposals(
        lineage,
        snapshot.mid_clusters,
        new_clusters,
        evidence,
        min_historical_coverage=options.min_historical_coverage,
        tolerance=options.tolerance,
    )
    selected = []
    for action in (RefinementAction.SPLIT, RefinementAction.MERGE):
        candidates = sorted(
            (proposal for proposal in proposals if proposal.action is action),
            key=lambda proposal: (
                proposal.pareto_front_rank,
                tuple(-value for value in proposal.objectives.values),
                proposal.component.component_id,
            ),
        )
        if candidates:
            selected.append(candidates[0])

    payload = {
        "protocol_id": "alfworld-deployment-optimization-v1-structural-pareto",
        "tower": _file_record(options.tower),
        "preprocessed": _file_record(options.preprocessed),
        "clusters": _file_record(options.clusters),
        "lineage_contract": {
            "min_old_retention": options.min_overlap_fraction,
            "min_new_historical_purity": options.min_overlap_fraction,
            "edge_rule": "old_retention >= threshold OR new_historical_purity >= threshold",
        },
        "metric_contract": {
            "scope": "historical_segments_only",
            "outcome_consistency": "segment_weighted_binary_outcome_purity",
            "transition_role_coherence": "segment_weighted_event_type_purity",
            "spectral_compactness": "mean_cosine_to_partition_recomputed_centroid",
            "min_historical_coverage": options.min_historical_coverage,
            "tolerance": options.tolerance,
        },
        "lineage_kind_counts": dict(sorted(Counter(item.kind.value for item in lineage).items())),
        "lineage": [_lineage_record(component) for component in lineage],
        "proposals": [_proposal_record(proposal) for proposal in proposals],
        "selected": [_proposal_record(proposal) for proposal in selected],
    }
    write_json(options.output, payload)
    print(json.dumps({key: payload[key] for key in ("lineage_kind_counts", "selected")}, indent=2))
    return 0


def _read_evidence(
    path: Path,
    historical_ids: frozenset[str],
) -> dict[str, SegmentStructuralEvidence]:
    evidence = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            successful = float(record["primary_score"]) >= 0.999
            for segment in record["segments"]:
                segment_id = str(segment["segment_id"])
                if segment_id not in historical_ids:
                    continue
                evidence[segment_id] = SegmentStructuralEvidence(
                    successful=successful,
                    event_type=str(segment["event_type"]),
                    embedding=np.asarray(segment["embedding"], dtype=np.float32),
                )
    if set(evidence) != set(historical_ids):
        raise ValueError("preprocessed input does not cover all historical Tower segments")
    return evidence


def _lineage_record(component) -> dict:
    return {
        "component_id": component.component_id,
        "kind": component.kind.value,
        "old_mid_ids": list(component.old_mid_ids),
        "new_mid_ids": list(component.new_mid_ids),
        "overlaps": [
            {
                "old_mid_id": overlap.old_mid_id,
                "new_mid_id": overlap.new_mid_id,
                "shared_member_count": overlap.shared_member_count,
                "old_retention": overlap.old_retention,
                "new_historical_purity": overlap.new_historical_purity,
            }
            for overlap in component.overlaps
        ],
    }


def _proposal_record(proposal) -> dict:
    return {
        "action": proposal.action.value,
        "component_id": proposal.component.component_id,
        "old_mid_ids": list(proposal.component.old_mid_ids),
        "new_mid_ids": list(proposal.component.new_mid_ids),
        "objectives": {
            "outcome_consistency_gain": proposal.objectives.outcome_consistency_gain,
            "transition_role_coherence_gain": proposal.objectives.transition_role_coherence_gain,
            "spectral_compactness_gain": proposal.objectives.spectral_compactness_gain,
        },
        "pareto_front_rank": proposal.pareto_front_rank,
    }


def _file_record(path: Path) -> dict:
    return {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.2)
    parser.add_argument("--min-historical-coverage", type=float, default=0.8)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    raise SystemExit(main(parser.parse_args()))
