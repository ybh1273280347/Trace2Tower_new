from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as parquet


EXCLUDED_SAMPLE_IDS = {
    "valid_seen": {"alfworld:valid_seen:trial_T20190906_173120_350651"},
    "valid_unseen": {"alfworld:valid_unseen:trial_T20190908_222917_366542"},
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(dataset_root: Path, split: str) -> list[dict]:
    table = parquet.read_table(dataset_root / f"{split}.parquet", columns=["extra_info"])
    rows = []
    for dataset_index, item in enumerate(table.column(0).combine_chunks().to_pylist()):
        rows.append(
            {
                "sample_id": f"alfworld:{split}:{item['task_id']}",
                "dataset_index": dataset_index,
                "source_split": split,
            }
        )
    return rows


def sample_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count < 0 or count > len(rows):
        raise ValueError(f"requested {count} rows from a pool of {len(rows)}")
    candidates = sorted(rows, key=lambda row: row["sample_id"])
    return sorted(random.Random(seed).sample(candidates, count), key=lambda row: row["sample_id"])


def summarize(rows: list[dict]) -> dict:
    return {
        "sample_count": len(rows),
        "sample_ids": [row["sample_id"] for row in rows],
    }


def main(options: argparse.Namespace) -> int:
    splits = {
        split: load_rows(options.dataset_root, split)
        for split in ("train", "valid_seen", "valid_unseen")
    }
    calibration = sample_rows(splits["train"], options.calibration_count, 1701)
    calibration_ids = {row["sample_id"] for row in calibration}
    pool_candidates = [row for row in splits["train"] if row["sample_id"] not in calibration_ids]
    trajectory_pool = sample_rows(pool_candidates, options.pool_count, 2903)
    dev = [
        row
        for row in splits["valid_seen"]
        if row["sample_id"] not in EXCLUDED_SAMPLE_IDS["valid_seen"]
    ]
    test = [
        row
        for row in splits["valid_unseen"]
        if row["sample_id"] not in EXCLUDED_SAMPLE_IDS["valid_unseen"]
    ]
    payload = {
        "protocol_id": "alfworld-agentbench-v1",
        "benchmark_metric": "binary_success_rate",
        "max_agent_steps": 20,
        "selection_order": ["success_rate", "mean_steps", "input_tokens", "invalid_action_rate"],
        "sampling_rule": "global deterministic sample over the complete split; no task-family partition",
        "fixed_before_alfworld_validation": {
            "high_top_k": 1,
            "retrieval_strategy": "diverse",
            "evidence_ablations": ["success-only", "mixed"],
        },
        "not_transferred_from_webshop": [
            "direct_mid_top_k",
            "flat_top_k",
            "similarity_thresholds",
        ],
        "dataset_sha256": {
            split: file_sha256(options.dataset_root / f"{split}.parquet")
            for split in splits
        },
        "calibration": {
            **summarize(calibration),
            "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
            "repeat_ids": [0],
            "purpose": "choose the trajectory-mining model only",
        },
        "trajectory_pool": {
            **summarize(trajectory_pool),
            "repeat_ids": [0, 1, 2, 3],
            "expected_episode_count": len(trajectory_pool) * 4,
            "expansion_rule": "add globally sampled train tasks and repeat four times",
        },
        "validation": {
            **summarize(dev),
            "source_split": "valid_seen",
            "excluded_prior_exposure": sorted(EXCLUDED_SAMPLE_IDS["valid_seen"]),
            "repeat_ids": [0, 1, 2],
            "configuration_selection_only": True,
        },
        "test": {
            **summarize(test),
            "source_split": "valid_unseen",
            "excluded_prior_exposure": sorted(EXCLUDED_SAMPLE_IDS["valid_unseen"]),
            "repeat_ids": [0, 1, 2],
            "opened_after_configuration_freeze": True,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "dataset_sha256"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    parser.add_argument("--output", type=Path, default=Path("configs/experiments/alfworld_protocol.json"))
    parser.add_argument("--calibration-count", type=int, default=30)
    parser.add_argument("--pool-count", type=int, default=300)
    raise SystemExit(main(parser.parse_args()))
