from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json


def read_ids(path: Path, split: str | None = None) -> set[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if split is not None and any(row["split"] != split for row in rows):
        raise ValueError(f"manifest {path} contains a non-{split} row")
    ids = {str(row["sample_id"]) for row in rows}
    if len(ids) != len(rows):
        raise ValueError(f"manifest {path} contains duplicate sample IDs")
    return ids


def read_history_ids(runs_root: Path) -> set[str]:
    ids = set()
    for path in runs_root.rglob("results.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                ids.add(str(json.loads(line)["sample_id"]))
    for path in runs_root.rglob("errors.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                key = record.get("episode_key", {})
                if key.get("sample_id") is not None:
                    ids.add(str(key["sample_id"]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--p100-audit", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    options = parser.parse_args()

    train_rows = [
        json.loads(line)
        for line in options.train_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(row["split"] != "train" for row in train_rows):
        raise ValueError("refinement source must be a train manifest")
    train_ids = {str(row["sample_id"]) for row in train_rows}
    if len(train_ids) != len(train_rows):
        raise ValueError("train manifest contains duplicate sample IDs")
    p100_audit = json.loads(options.p100_audit.read_text(encoding="utf-8"))
    p100_ids = set(p100_audit["pools"]["p100"]["sample_ids"])
    excluded = {path.as_posix(): read_ids(path) for path in options.exclude}
    history_ids = read_history_ids(options.runs_root)
    excluded["p100_pool"] = p100_ids
    excluded["historical_runs"] = history_ids
    excluded_ids = set().union(*excluded.values())
    candidates = sorted(train_ids - excluded_ids, key=lambda item: int(item.rsplit(":", 1)[1]))
    if len(candidates) < options.count:
        raise ValueError(f"only {len(candidates)} eligible train tasks remain")
    selected = sorted(random.Random(options.seed).sample(candidates, options.count), key=lambda item: int(item.rsplit(":", 1)[1]))
    output_rows = [
        {
            "benchmark": "webshop",
            "split": "train",
            "sample_id": sample_id,
            "dataset_index": int(sample_id.rsplit(":", 1)[1]),
            "source_split": "goals",
            "repeat_id": 0,
        }
        for sample_id in selected
    ]
    options.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    options.output_manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    audit = {
        "protocol_id": "webshop-train-refinement-v1",
        "selection_frozen_before_rollout": True,
        "benchmark": "webshop",
        "split": "train",
        "seed": options.seed,
        "task_count": len(selected),
        "repeat_ids": [0, 1, 2],
        "source_train_manifest": options.train_manifest.as_posix(),
        "source_train_manifest_sha256": hashlib.sha256(options.train_manifest.read_bytes()).hexdigest(),
        "p100_audit": options.p100_audit.as_posix(),
        "excluded_manifests": {
            path: {"count": len(ids), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for path, ids in excluded.items()
            if path not in {"p100_pool", "historical_runs"}
        },
        "excluded_counts": {name: len(ids) for name, ids in excluded.items()},
        "sample_ids": selected,
        "sample_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "disjoint_from_all_exclusions": not set(selected) & excluded_ids,
        "output_manifest": options.output_manifest.as_posix(),
    }
    write_json(options.output_audit, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", action="help")
    raise SystemExit(main())
