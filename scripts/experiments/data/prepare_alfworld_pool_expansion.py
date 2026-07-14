from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    audit = json.loads(options.audit.read_text(encoding="utf-8"))
    deficient = tuple(sorted(audit["deficient_families"]))
    if not deficient:
        raise ValueError("ALFWorld pool audit has no deficient families")
    excluded = set(protocol["calibration"]["sample_ids"]) | set(
        protocol["trajectory_pool"]["sample_ids"]
    )
    rows = []
    for item in parquet.read_table(
        options.dataset_root / "train.parquet", columns=["extra_info"]
    ).column(0).combine_chunks().to_pylist():
        sample_id = f"alfworld:train:{item['task_id']}"
        if sample_id not in excluded and item["task_family"] in deficient:
            rows.append((item["task_family"], sample_id))

    selected = []
    for family in deficient:
        candidates = sorted(sample_id for current, sample_id in rows if current == family)
        if len(candidates) < options.tasks_per_family:
            raise ValueError(f"insufficient expansion candidates for {family}")
        selected.extend(candidates[: options.tasks_per_family])
    payload = {
        "protocol_id": protocol["protocol_id"],
        "source_pool_audit": options.audit.as_posix(),
        "source_pool_audit_sha256": hashlib.sha256(
            options.audit.read_bytes()
        ).hexdigest(),
        "selection_rule": "lexicographic first unused train tasks per deficient family",
        "deficient_families": list(deficient),
        "tasks_per_family": options.tasks_per_family,
        "sample_ids": selected,
        "repeat_ids": [0, 1, 2, 3],
        "expected_episode_count": len(selected) * 4,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiments/alfworld_protocol_v1.json"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    parser.add_argument("--tasks-per-family", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
