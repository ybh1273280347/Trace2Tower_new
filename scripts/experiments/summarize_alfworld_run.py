from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

import pyarrow.parquet as parquet


def load_results(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(run_dir.glob("shard-*/results.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def task_families(dataset_root: Path, split: str) -> dict[str, str]:
    table = parquet.read_table(dataset_root / f"{split}.parquet", columns=["extra_info"])
    return {
        f"alfworld:{split}:{item['task_id']}": item["task_family"]
        for item in table.column(0).combine_chunks().to_pylist()
    }


def summarize(rows: list[dict]) -> dict:
    total_steps = sum(int(row["steps"]) for row in rows)
    successful = [row for row in rows if row["success"]]
    usage = [row for row in rows if row.get("input_tokens") is not None]
    return {
        "episode_count": len(rows),
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows),
        "mean_steps": fmean(int(row["steps"]) for row in rows),
        "mean_success_steps": (
            fmean(int(row["steps"]) for row in successful) if successful else None
        ),
        "invalid_action_rate": (
            sum(int(row["invalid_actions"]) for row in rows) / max(total_steps, 1)
        ),
        "mean_input_tokens": (
            fmean(int(row["input_tokens"]) for row in usage) if usage else None
        ),
    }


def main(options: argparse.Namespace) -> int:
    families = task_families(options.dataset_root, options.source_split)
    output = {"source_split": options.source_split, "runs": {}}
    for assignment in options.run:
        label, separator, raw_path = assignment.partition("=")
        if not separator:
            raise ValueError("--run expects LABEL=PATH")
        rows = load_results(Path(raw_path))
        expected_ids = set(options.expected_sample_id)
        if expected_ids and {row["sample_id"] for row in rows} != expected_ids:
            missing = sorted(expected_ids - {row["sample_id"] for row in rows})
            raise ValueError(f"{label} result coverage incomplete: {missing[:5]}")
        grouped = defaultdict(list)
        for row in rows:
            grouped[families[row["sample_id"]]].append(row)
        output["runs"][label] = {
            "aggregate": summarize(rows),
            "by_family": {
                family: summarize(grouped[family]) for family in sorted(grouped)
            },
            "finish_reason_counts": dict(
                sorted(Counter(row["finish_reason"] for row in rows).items())
            ),
        }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--expected-sample-id", action="append", default=[])
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
