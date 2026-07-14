from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    ManifestEntry,
    read_manifest,
    write_manifest,
)


def main(options: argparse.Namespace) -> int:
    goal_count = len(json.loads(options.goals.read_text(encoding="utf-8")))
    if options.candidate_start < 0 or options.candidate_end > goal_count:
        raise ValueError("candidate range exceeds the WebShop goal set")
    if options.candidate_start >= options.candidate_end:
        raise ValueError("candidate range must be non-empty")

    excluded_ids = set()
    excluded_manifests = []
    for path in options.exclude_manifest:
        entries = read_manifest(path)
        if any(entry.benchmark is not Benchmark.WEBSHOP for entry in entries):
            raise ValueError(f"{path} contains a non-WebShop entry")
        excluded_ids.update(entry.dataset_index for entry in entries)
        excluded_manifests.append(
            {
                "path": path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_count": len({entry.sample_id for entry in entries}),
            }
        )

    candidates = [
        index
        for index in range(options.candidate_start, options.candidate_end)
        if index not in excluded_ids
    ]
    if options.sample_count > len(candidates):
        raise ValueError("sample count exceeds the remaining candidate pool")
    selected = sorted(random.Random(options.seed).sample(candidates, options.sample_count))
    entries = [
        ManifestEntry(
            benchmark=Benchmark.WEBSHOP,
            split=ExperimentSplit.TEST,
            sample_id=f"webshop:{index}",
            dataset_index=index,
            source_split="goals",
            repeat_id=0,
        )
        for index in selected
    ]
    write_manifest(options.output_manifest, entries)
    write_json(
        options.output_audit,
        {
            "protocol_id": "webshop-original-concept-v1-test-b",
            "selection_frozen_before_rollout": True,
            "algorithm": "random.Random(seed).sample without replacement",
            "seed": options.seed,
            "candidate_range": [options.candidate_start, options.candidate_end],
            "excluded_manifests": excluded_manifests,
            "remaining_candidate_count": len(candidates),
            "sample_count": len(entries),
            "sample_ids": [entry.sample_id for entry in entries],
            "manifest": options.output_manifest.as_posix(),
            "manifest_sha256": hashlib.sha256(
                options.output_manifest.read_bytes()
            ).hexdigest(),
        },
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--goals",
        type=Path,
        default=Path("Datasets/webshop/goals.json"),
    )
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--candidate-end", type=int, default=1000)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
