from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    build_alfworld_manifests,
    build_webshop_manifests,
    shard_counts,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", *Benchmark), default="all")
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    common = yaml.safe_load((options.config_root / "common.yaml").read_text(encoding="utf-8"))
    repeat_ids = tuple(int(value) for value in common["repeat_ids"])
    num_shards = int(common["num_shards"])
    output_root = Path(common["manifests_dir"])
    summary = {}

    if options.benchmark in ("all", Benchmark.ALFWORLD):
        config = yaml.safe_load((options.config_root / "alfworld.yaml").read_text(encoding="utf-8"))
        manifests = build_alfworld_manifests(
            Path(config["dataset_root"]),
            {
                ExperimentSplit(split): source_split
                for split, source_split in config["splits"].items()
            },
            repeat_ids,
        )
        for split, entries in manifests.items():
            if not options.dry_run:
                write_manifest(output_root / f"alfworld_{split}.jsonl", entries)
            summary[f"alfworld_{split}"] = {
                "count": len(entries),
                "shards": shard_counts(entries, num_shards),
            }

    if options.benchmark in ("all", Benchmark.WEBSHOP):
        config = yaml.safe_load((options.config_root / "webshop.yaml").read_text(encoding="utf-8"))
        manifests = build_webshop_manifests(
            Path(config["dataset_root"]) / "goals.json",
            {
                ExperimentSplit(split): (int(bounds["start"]), int(bounds["end"]))
                for split, bounds in config["splits"].items()
            },
            repeat_ids,
        )
        for split, entries in manifests.items():
            if not options.dry_run:
                write_manifest(output_root / f"webshop_{split}.jsonl", entries)
            summary[f"webshop_{split}"] = {
                "count": len(entries),
                "shards": shard_counts(entries, num_shards),
            }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
