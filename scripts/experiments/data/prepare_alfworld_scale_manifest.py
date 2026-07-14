from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet


FAMILIES = (
    "look_at_obj_in_light",
    "pick_and_place",
    "pick_clean_then_place",
    "pick_cool_then_place",
    "pick_heat_then_place",
    "pick_two_obj_and_place",
)


def load_train_rows(dataset_root: Path) -> list[dict]:
    values = parquet.read_table(
        dataset_root / "train.parquet", columns=["extra_info"]
    ).column(0).combine_chunks().to_pylist()
    return [
        {
            "sample_id": f"alfworld:train:{item['task_id']}",
            "dataset_index": index,
            "source_split": "train",
            "task_family": item["task_family"],
        }
        for index, item in enumerate(values)
    ]


def read_jsonl_ids(path: Path) -> set[str]:
    return {
        json.loads(line)["sample_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def read_calibration_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload["calibration"]["sample_ids"])


def main(options: argparse.Namespace) -> int:
    rows = load_train_rows(options.dataset_root)
    by_id = {row["sample_id"]: row for row in rows}
    existing = read_jsonl_ids(options.existing_pool)
    calibration = read_calibration_ids(options.protocol)
    missing_existing = existing - set(by_id)
    if missing_existing:
        raise ValueError(f"existing pool contains unknown tasks: {sorted(missing_existing)[:3]}")
    excluded = existing | calibration
    existing_counts = Counter(by_id[sample_id]["task_family"] for sample_id in existing)
    base = {"look_at_obj_in_light": 167, "pick_and_place": 167, "pick_clean_then_place": 167,
            "pick_cool_then_place": 167, "pick_heat_then_place": 166, "pick_two_obj_and_place": 166}
    if sum(base.values()) != options.target_total:
        raise ValueError("target total does not match the fixed balanced target")
    if any(existing_counts[family] > base[family] for family in FAMILIES):
        raise ValueError("existing pool exceeds the fixed family target")

    selected = set(existing)
    additions = {}
    for family in FAMILIES:
        needed = base[family] - existing_counts[family]
        candidates = sorted(
            row["sample_id"]
            for row in rows
            if row["task_family"] == family and row["sample_id"] not in excluded
        )
        if len(candidates) < needed:
            raise ValueError(f"not enough candidates for {family}: {len(candidates)} < {needed}")
        additions[family] = candidates[:needed]
        selected.update(additions[family])

    if len(selected) != options.target_total:
        raise ValueError(f"selected {len(selected)} tasks, expected {options.target_total}")
    ordered = sorted(selected)
    manifest_rows = [by_id[sample_id] for sample_id in ordered]
    options.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with options.output_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if options.output_additions_jsonl:
        addition_rows = [by_id[sample_id] for sample_id in sorted(selected - existing)]
        options.output_additions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with options.output_additions_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
            for row in addition_rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    payload = {
        "protocol_id": "alfworld-agentbench-v1",
        "benchmark": "alfworld",
        "source_split": "train",
        "trajectory_pool_agent_model": "deepseek-v4-flash",
        "trajectory_pool_rollout": "additions-only",
        "target_total": options.target_total,
        "repeat_ids": [0, 1, 2, 3],
        "selection_rule": "preserve the audited expanded pool, then choose lexicographic first unused tasks per family to fixed balanced targets",
        "existing_pool": options.existing_pool.as_posix(),
        "existing_pool_sha256": hashlib.sha256(options.existing_pool.read_bytes()).hexdigest(),
        "calibration_protocol": options.protocol.as_posix(),
        "calibration_ids_excluded": len(calibration),
        "existing_task_count": len(existing),
        "added_task_count": len(selected - existing),
        "family_targets": dict(base),
        "family_counts": dict(sorted(Counter(row["task_family"] for row in manifest_rows).items())),
        "added_by_family": additions,
        "manifest_jsonl": options.output_jsonl.as_posix(),
        "manifest_jsonl_sha256": hashlib.sha256(options.output_jsonl.read_bytes()).hexdigest(),
        "sample_ids": ordered,
    }
    options.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    options.output_metadata.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "sample_ids"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    parser.add_argument("--existing-pool", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=1000)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-additions-jsonl", type=Path)
    parser.add_argument("--output-metadata", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
