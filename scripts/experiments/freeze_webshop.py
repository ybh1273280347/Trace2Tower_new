from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VALIDATION_RUNS = (
    "webshop-test100b-repeat3-no-skill-v3",
    "webshop-test100b-repeat3-flat94-diverse3-v1",
    "webshop-test100b-repeat3-flat94-diverse-v1",
    "webshop-test100b-repeat3-tower-success-only-cap3-v4",
    "webshop-test100b-repeat3-tower-success-only-cap5-v4",
    "webshop-test100b-repeat3-tower-success-only-cap8-v4",
    "webshop-test100b-repeat3-tower-success-only-cap12-v4",
    "webshop-test100b-repeat3-tower-mixed-cap3-v4",
    "webshop-test100b-repeat3-tower-mixed-cap5-v4",
    "webshop-test100b-repeat3-tower-mixed-cap8-v4",
    "webshop-test100b-repeat3-tower-mixed-cap12-v4",
)
FINAL_RUNS = tuple(
    f"webshop-final-random300-{model}-{method}-v1"
    for model in ("flash", "pro")
    for method in (
        "noskill",
        "flat-cap3",
        "skillx-official",
        "success-tower-cap3",
        "mixed-tower-cap3",
    )
)
MID_ONLY_RUNS = tuple(
    f"webshop-final-random300-{model}-{evidence}-tower-cap3-mid-only-v1"
    for model in ("flash", "pro")
    for evidence in ("success", "mixed")
)
CROSS_HIGH_RUNS = tuple(
    f"webshop-final-random300-{model}-{cross}-cap3-v1"
    for model in ("flash", "pro")
    for cross in (
        "success-mid-mixed-high",
        "mixed-mid-success-high",
    )
)
GROUPS = {
    "validation": VALIDATION_RUNS,
    "final_baselines": FINAL_RUNS,
    "mid_only": MID_ONLY_RUNS,
    "cross_high": CROSS_HIGH_RUNS,
}
FROZEN_FILES = (
    "configs/experiments/webshop_final_random300_v1.json",
    "configs/experiments/skillx.yaml",
    "configs/experiments/trace2tower_static_diverse3.yaml",
    "configs/experiments/trace2tower_static_diverse3_mid_only.yaml",
    "src/trace2tower/methods/trace2tower/config.py",
    "docs/methods/retrieval.md",
    "artifacts/manifests/webshop-final-random300-v1/webshop_test.jsonl",
    "artifacts/flat_skill_summary/webshop-flash50-repeat4-mmr-v1/library.json",
    "artifacts/trace2tower/towers/webshop-flash50-repeat4-success-only-task-support10-cap3-v4.json",
    "artifacts/trace2tower/towers/webshop-flash50-repeat4-mixed-task-support10-cap3-v4.json",
    "artifacts/skillx/webshop-success94-official-parallel2-recoverable-v6/library.json",
    "artifacts/skillx/webshop-success94-official-parallel2-recoverable-v6/report.json",
    "artifacts/skillx/webshop-success94-official-execution-v1/library.json",
    "artifacts/evaluations/webshop-final-random300-v1/report.json",
    "artifacts/evaluations/webshop-cross-high-random300-v1/report.json",
    "scripts/experiments/summarize_webshop_final_random300.py",
    "scripts/experiments/summarize_webshop_cross_high_random300.py",
    "scripts/experiments/freeze_webshop.py",
    "docs/protocols/webshop-validation-and-final-test.md",
    "docs/reports/webshop/final-random300-report.md",
    "docs/reports/webshop/mid-only-ablation.md",
    "docs/baselines/skillx.md",
    "docs/reports/webshop/complete-experiment-report.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_digest(records: list[dict]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_record(root: Path, run_id: str, expected_episodes: int) -> dict:
    run_dir = root / "artifacts" / "runs" / run_id
    result_paths = sorted(run_dir.glob("**/results.jsonl"))
    error_paths = sorted(run_dir.glob("**/errors.jsonl"))
    if len(result_paths) != 10:
        raise ValueError(f"{run_id} does not have ten result shards")
    rows = [
        json.loads(line)
        for path in result_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    episode_count = len(rows)
    error_count = sum(
        len(path.read_text(encoding="utf-8").splitlines()) for path in error_paths
    )
    if episode_count != expected_episodes or error_count:
        raise ValueError(
            f"{run_id} expected {expected_episodes}/0 episodes/errors, "
            f"found {episode_count}/{error_count}"
        )
    keys = [(str(row["sample_id"]), int(row["repeat_id"])) for row in rows]
    expected_task_count = expected_episodes // 3
    task_ids = {sample_id for sample_id, _ in keys}
    if (
        len(set(keys)) != expected_episodes
        or len(task_ids) != expected_task_count
        or {repeat_id for _, repeat_id in keys} != {0, 1, 2}
        or any(row.get("error") is not None for row in rows)
    ):
        raise ValueError(f"{run_id} has duplicate, incomplete, or erroneous episode keys")
    result_shards = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
        }
        for path in result_paths
    ]
    config_path = run_dir / "resolved-config.yaml"
    return {
        "run_id": run_id,
        "episode_count": episode_count,
        "task_count": len(task_ids),
        "unique_episode_key_count": len(set(keys)),
        "repeat_ids": [0, 1, 2],
        "episode_key_sha256": combined_digest(
            [
                {"sample_id": sample_id, "repeat_id": repeat_id}
                for sample_id, repeat_id in sorted(keys)
            ]
        ),
        "error_count": error_count,
        "result_shard_count": len(result_paths),
        "resolved_config_path": config_path.relative_to(root).as_posix(),
        "resolved_config_sha256": sha256(config_path),
        "result_set_sha256": combined_digest(result_shards),
        "result_shards": result_shards,
    }


def build_manifest(root: Path) -> dict:
    run_groups = {
        group: [
            run_record(root, run_id, 300 if group == "validation" else 900)
            for run_id in run_ids
        ]
        for group, run_ids in GROUPS.items()
    }
    frozen_files = [
        {
            "path": relative,
            "sha256": sha256(root / relative),
        }
        for relative in FROZEN_FILES
    ]
    return {
        "version": "webshop-freeze-v1",
        "frozen_on": "2026-07-14",
        "status": "frozen",
        "scope": {
            "benchmark": "webshop",
            "condition_count": sum(len(run_ids) for run_ids in GROUPS.values()),
            "episode_count": sum(
                300 if group == "validation" else 900
                for group, run_ids in GROUPS.items()
                for _ in run_ids
            ),
            "unresolved_error_count": 0,
        },
        "policy": {
            "validation_only_selects_configuration": True,
            "random300_never_selects_configuration": True,
            "post_selection_ablations_never_select_configuration": True,
            "allowed_after_freeze": [
                "recompute statistics from frozen results",
                "verify hashes and completeness",
                "correct documentation errors without changing experiments",
                "maintain reproducibility without changing behavior",
            ],
        },
        "protocol_provenance": {
            "high_top_k": 1,
            "selection_mode": "fixed algorithm contract, not validation-selected",
            "introduced_in_commit": "9b0c7069cd9597c23c88da8a9a9540afb18c48c7",
            "random300_preregistered_in_commit": "17e2b054fbd2aef0a51ce706815ec94a99e6399b",
            "contract_source": "src/trace2tower/methods/trace2tower/config.py",
        },
        "run_groups": run_groups,
        "frozen_files": frozen_files,
    }


def main(options: argparse.Namespace) -> int:
    root = options.root.resolve()
    actual = build_manifest(root)
    if options.verify:
        expected = json.loads(options.output.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("WebShop frozen state differs from the manifest")
        print(
            f"verified {actual['scope']['condition_count']} conditions, "
            f"{actual['scope']['episode_count']} episodes"
        )
        return 0
    options.output.write_text(
        json.dumps(actual, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"froze {actual['scope']['condition_count']} conditions, "
        f"{actual['scope']['episode_count']} episodes"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/webshop/freeze-manifest.json"),
    )
    parser.add_argument("--verify", action="store_true")
    raise SystemExit(main(parser.parse_args()))
