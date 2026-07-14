from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.experiments.data.prepare_alfworld_protocol import FAMILIES, load_rows
from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.manifests import AlfworldTaskFamily

FAMILY_DESCRIPTIONS = {
    family.value: family.retrieval_description for family in AlfworldTaskFamily
}


def main(options: argparse.Namespace) -> int:
    family_by_sample = {
        row["sample_id"]: row["task_family"]
        for row in load_rows(options.dataset_root, "train")
    }
    records = [
        json.loads(line)
        for line in options.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    grouped = {family: [] for family in FAMILIES}
    for record in records:
        try:
            family = family_by_sample[record["sample_id"]]
        except KeyError as exc:
            raise ValueError(f"unknown ALFWorld training sample: {exc.args[0]}") from exc
        grouped[family].append(record)

    options.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for family in FAMILIES:
        output = options.output_dir / f"{family}.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for record in grouped[family]:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        outputs[family] = {
            "output": output.as_posix(),
            "trajectory_count": len(grouped[family]),
            "success_count": sum(
                float(record["primary_score"]) >= options.success_threshold
                for record in grouped[family]
            ),
            "segment_count": sum(
                len(record.get("segments", ())) for record in grouped[family]
            ),
            "has_preprocessed_segments": all(
                "segments" in record for record in grouped[family]
            ),
        }
    report = {
        "input": options.input.as_posix(),
        "trajectory_count": len(records),
        "family_counts": dict(
            sorted(Counter(family_by_sample[record["sample_id"]] for record in records).items())
        ),
        "families": outputs,
    }
    write_json(options.output_dir / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("Datasets/alfworld"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    raise SystemExit(main(parser.parse_args()))
