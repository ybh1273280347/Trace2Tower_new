from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preexisting_webshop_ids(runs_root: Path) -> set[int]:
    used = set()
    for filename in ("results.jsonl", "errors.jsonl"):
        for path in runs_root.glob(f"**/{filename}"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id", ""))
                if sample_id.startswith("webshop:"):
                    used.add(int(sample_id.removeprefix("webshop:")))
    return used


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as output:
        temporary = Path(output.name)
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main(options: argparse.Namespace) -> int:
    goals = json.loads(options.goals.read_text(encoding="utf-8"))
    if len(goals) < options.candidate_end:
        raise ValueError("candidate range exceeds the WebShop goal set")
    used = preexisting_webshop_ids(options.runs_root)
    candidates = [
        index
        for index in range(options.candidate_start, options.candidate_end)
        if index not in used
    ]
    required = len(options.seed) * options.tasks_per_seed
    if len(candidates) < required:
        raise ValueError("insufficient unused candidates for disjoint selections")

    remaining = set(candidates)
    selections = []
    for seed in options.seed:
        selected = sorted(random.Random(seed).sample(sorted(remaining), options.tasks_per_seed))
        remaining.difference_update(selected)
        selections.append(
            {
                "seed": seed,
                "sample_ids": [f"webshop:{index}" for index in selected],
                "sample_count": len(selected),
            }
        )

    artifacts = {
        "flat_cap3": options.flat_artifact,
        "success_only_tower_cap3": options.success_artifact,
        "mixed_tower_cap3": options.mixed_artifact,
    }
    artifact_records = {
        name: {"path": path.as_posix(), "sha256": sha256_file(path)}
        for name, path in artifacts.items()
    }
    protocol = {
        "version": "webshop-final-random300-v1",
        "benchmark": "webshop",
        "candidate_index_range": [options.candidate_start, options.candidate_end],
        "candidate_semantics": "WebShop held-out goal indices below the train split start",
        "goals_path": options.goals.as_posix(),
        "goals_sha256": sha256_file(options.goals),
        "preexisting_used_id_count": len(used),
        "preexisting_used_ids_sha256": canonical_hash(sorted(used)),
        "unused_candidate_count_before_sampling": len(candidates),
        "sampling": {
            "algorithm": "Python random.Random(seed).sample over sorted remaining IDs",
            "without_replacement_across_seeds": True,
            "seeds": options.seed,
            "tasks_per_seed": options.tasks_per_seed,
            "repeat_ids": [0, 1, 2],
            "expected_independent_task_count": required,
            "expected_episode_count_per_method": required * 3,
        },
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
        "methods": [
            "no_skill",
            "flat_skill_summary_cap3",
            "trace2tower_success_only_cap3",
            "trace2tower_mixed_cap3",
        ],
        "method_artifacts": artifact_records,
        "selection_policy": (
            "Configurations and artifacts are frozen before rollout. Results may not be "
            "used to choose a different cap, evidence policy, or artifact on this test set."
        ),
        "aggregation_policy": (
            "Report each seed's 100-task/300-episode aggregate and a combined estimate "
            "clustered over 300 independent tasks; repeats are not independent samples."
        ),
        "selections": selections,
    }
    protocol["selection_id"] = f"selection_{canonical_hash(protocol)[:16]}"
    write_json(options.output, protocol)
    print(json.dumps({
        "selection_id": protocol["selection_id"],
        "unused_candidate_count": len(candidates),
        "selected_task_count": required,
        "overlap_count": required - len({item for group in selections for item in group["sample_ids"]}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", type=Path, default=Path("Datasets/webshop/goals.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--candidate-end", type=int, default=1000)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--tasks-per-seed", type=int, default=100)
    parser.add_argument(
        "--flat-artifact",
        type=Path,
        default=Path("artifacts/flat_skill_summary/webshop-flash50-repeat4-mmr-v1/library.json"),
    )
    parser.add_argument(
        "--success-artifact",
        type=Path,
        default=Path("artifacts/trace2tower/towers/webshop-flash50-repeat4-success-only-task-support10-cap3-v4.json"),
    )
    parser.add_argument(
        "--mixed-artifact",
        type=Path,
        default=Path("artifacts/trace2tower/towers/webshop-flash50-repeat4-mixed-task-support10-cap3-v4.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/experiments/webshop_final_random300_v1.json"),
    )
    options = parser.parse_args()
    if not options.seed:
        options.seed = [42, 43, 44]
    raise SystemExit(main(options))
