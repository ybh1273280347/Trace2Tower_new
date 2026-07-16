from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.graph_retrieval import TowerGraphProfile
from trace2tower.methods.trace2tower.models import event_type_from_value
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def main(options: argparse.Namespace) -> int:
    tower = TowerSnapshot.from_record(json.loads(options.tower.read_text(encoding="utf-8")))
    mid_by_segment = {
        segment_id: cluster.cluster_id
        for cluster in tower.mid_clusters
        for segment_id in cluster.member_segment_ids
    }
    counts = {mid_id: Counter() for mid_id in mid_by_segment.values()}
    seen_segment_ids = set()
    with options.preprocessed.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            for segment in json.loads(line)["segments"]:
                segment_id = segment["segment_id"]
                mid_id = mid_by_segment.get(segment_id)
                if mid_id is None:
                    continue
                event = segment["event_type"]
                if event is None:
                    raise ValueError("graph profile requires domain event labels")
                counts[mid_id][event_type_from_value(event)] += 1
                seen_segment_ids.add(segment_id)
    if seen_segment_ids != set(mid_by_segment):
        raise ValueError("preprocessed data does not cover every Tower Mid segment")
    profile = TowerGraphProfile(
        tower_snapshot_id=tower.snapshot_id,
        benchmark=tower.benchmark,
        mid_event_counts={mid_id: dict(counts[mid_id]) for mid_id in sorted(counts)},
    )
    write_json(options.output, profile.to_record())
    print(options.output)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
