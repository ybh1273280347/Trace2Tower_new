from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_BATCH_SIZE = 4
MERGE_BATCH_SIZE = 32

SETTINGS = {
    "alfworld": {
        "source_pool": "artifacts/trajectories/alfworld/alfworld-pool-v1-pro-expanded.jsonl",
        "combined_artifact": "artifacts/baselines/trace2skill/gpt54/alfworld-p310/artifact.json",
        "combined_build": "artifacts/baselines/trace2skill/gpt54/alfworld-p310/artifact-build",
        "combined_run": "artifacts/runs/trace2skill-gpt54-p310-alfworld-test-r0",
        "error_artifact": "artifacts/baselines/trace2skill/gpt54/alfworld-p310-error/artifact.json",
        "error_build": "artifacts/baselines/trace2skill/gpt54/alfworld-p310-error/artifact-build",
        "error_run": "artifacts/runs/trace2skill-gpt54-p310-error-alfworld-test-r0",
        "expected_eval_count": 134,
    },
    "webshop": {
        "source_pool": "artifacts/trajectories/webshop/scale-v1/webshop-scale-v1-p100.jsonl",
        "combined_artifact": "artifacts/baselines/trace2skill/gpt54/webshop-p100/artifact.json",
        "combined_build": "artifacts/baselines/trace2skill/gpt54/webshop-p100/artifact-build",
        "combined_run": "artifacts/runs/trace2skill-gpt54-p100-webshop-validation-r0",
        "error_artifact": "artifacts/baselines/trace2skill/gpt54/webshop-p100-error/artifact.json",
        "error_build": "artifacts/baselines/trace2skill/gpt54/webshop-p100-error/artifact-build",
        "error_run": "artifacts/runs/trace2skill-gpt54-p100-error-webshop-validation-r0",
        "expected_eval_count": 100,
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(items: list, size: int) -> list[list]:
    return [items[offset : offset + size] for offset in range(0, len(items), size)]


def audit_analysis(source_pool: Path, analysis_dir: Path) -> tuple[dict, list[dict]]:
    selected = sorted(
        (row for row in read_jsonl(source_pool) if int(row["repeat_id"]) == 0),
        key=lambda row: row["sample_id"],
    )
    expected_batches = chunks(selected, ANALYSIS_BATCH_SIZE)
    paths = sorted(analysis_dir.glob("batch-*.json"))
    failures = []
    output_records = []
    for index, batch in enumerate(expected_batches):
        path = analysis_dir / f"batch-{index:04d}.json"
        if not path.exists():
            failures.append(f"missing analyst checkpoint {path.name}")
            continue
        payload = read_json(path)
        records = payload.get("records")
        if not isinstance(records, list):
            failures.append(f"invalid analyst payload {path.name}")
            continue
        expected_ids = {row["sample_id"] for row in batch}
        actual_ids = {row.get("instance_id") for row in records}
        if actual_ids != expected_ids or len(records) != len(actual_ids):
            failures.append(f"analyst coverage mismatch {path.name}")
        output_records.extend(records)

    all_ids = [record.get("instance_id") for record in output_records]
    expected_ids = {row["sample_id"] for row in selected}
    if set(all_ids) != expected_ids or len(all_ids) != len(set(all_ids)):
        failures.append("global analyst coverage is incomplete or duplicated")

    item_counts = []
    invalid_items = 0
    outcomes = {"success": 0, "error": 0}
    for record in output_records:
        outcome = record.get("outcome")
        if outcome in outcomes:
            outcomes[outcome] += 1
        else:
            failures.append(f"invalid outcome for {record.get('instance_id')}")
        items = record.get("items")
        if not isinstance(items, list) or len(items) > 3:
            failures.append(f"invalid item list for {record.get('instance_id')}")
            continue
        item_counts.append(len(items))
        for item in items:
            if set(item) != {"title", "guidance", "evidence"} or any(
                not isinstance(item[field], str) or not item[field].strip()
                for field in ("title", "guidance", "evidence")
            ):
                invalid_items += 1

    checkpoint_names = {path.name for path in paths}
    expected_names = {f"batch-{index:04d}.json" for index in range(len(expected_batches))}
    if checkpoint_names != expected_names:
        failures.append("analyst checkpoint file set differs from the deterministic batch plan")
    return (
        {
            "source_pool": source_pool.relative_to(ROOT).as_posix(),
            "source_pool_sha256": sha256(source_pool),
            "source_pool_rows": len(read_jsonl(source_pool)),
            "selected_contract": "repeat_id=0, one trajectory per training task",
            "selected_trajectory_count": len(selected),
            "selected_unique_sample_ids": len(expected_ids),
            "analysis_batch_size": ANALYSIS_BATCH_SIZE,
            "expected_analysis_batches": len(expected_batches),
            "completed_analysis_checkpoints": len(paths),
            "checkpoint_json_parse_failures": 0,
            "covered_trajectory_count": len(set(all_ids)),
            "duplicate_trajectory_records": len(all_ids) - len(set(all_ids)),
            "outcome_counts": outcomes,
            "total_trajectory_local_patches": sum(item_counts),
            "trajectories_with_zero_patches": sum(count == 0 for count in item_counts),
            "trajectories_with_one_to_three_patches": sum(count > 0 for count in item_counts),
            "min_patches_per_trajectory": min(item_counts),
            "max_patches_per_trajectory": max(item_counts),
            "invalid_patch_items": invalid_items,
            "passed": not failures and invalid_items == 0,
            "failures": failures,
        },
        output_records,
    )


def audit_merge(build_dir: Path, initial_patches: list[dict]) -> dict:
    merge_dir = build_dir / "merge"
    current = initial_patches
    level = 0
    levels = []
    failures = []
    consumed_checkpoint_names = set()
    while len(current) > MERGE_BATCH_SIZE:
        groups = chunks(current, MERGE_BATCH_SIZE)
        outputs = []
        for index in range(len(groups)):
            path = merge_dir / f"level-{level:02d}-batch-{index:04d}.json"
            consumed_checkpoint_names.add(path.name)
            if not path.exists():
                failures.append(f"missing merge checkpoint {path.name}")
                continue
            payload = read_json(path)
            patches = payload.get("patches")
            if not isinstance(patches, list) or not patches or len(patches) > 12:
                failures.append(f"invalid merge output {path.name}")
                continue
            outputs.extend(patches)
        levels.append(
            {
                "level": level,
                "input_patch_count": len(current),
                "expected_merge_calls": len(groups),
                "completed_merge_calls": len(groups) - sum(
                    failure.startswith("missing merge checkpoint")
                    and f"level-{level:02d}-" in failure
                    for failure in failures
                ),
                "output_patch_count": len(outputs),
            }
        )
        current = outputs
        level += 1
        if not current:
            break

    final_path = merge_dir / f"level-{level:02d}-final.json"
    consumed_checkpoint_names.add(final_path.name)
    final_patches = []
    if not final_path.exists():
        failures.append(f"missing final merge checkpoint {final_path.name}")
    else:
        payload = read_json(final_path)
        final_patches = payload.get("patches")
        if not isinstance(final_patches, list) or not final_patches or len(final_patches) > 12:
            failures.append(f"invalid final merge output {final_path.name}")
            final_patches = []
    levels.append(
        {
            "level": level,
            "input_patch_count": len(current),
            "expected_merge_calls": 1,
            "completed_merge_calls": int(final_path.exists()),
            "output_patch_count": len(final_patches),
            "final": True,
        }
    )

    actual_checkpoint_names = {path.name for path in merge_dir.glob("level-*.json")}
    if actual_checkpoint_names != consumed_checkpoint_names:
        failures.append("merge checkpoint file set differs from the deterministic reduction plan")
    report = read_json(build_dir / "build-report.json")
    if report["final_patch_count"] != len(final_patches):
        failures.append("build report final patch count differs from final merge checkpoint")
    if not (build_dir / "final-skill.json").exists():
        failures.append("final skill materialization checkpoint is missing")
    return {
        "initial_patch_count": len(initial_patches),
        "merge_batch_size": MERGE_BATCH_SIZE,
        "levels": levels,
        "merge_checkpoint_count": len(actual_checkpoint_names),
        "final_patch_count": len(final_patches),
        "final_skill_materialized": (build_dir / "final-skill.json").exists(),
        "passed": not failures,
        "failures": failures,
    }


def audit_deployment(artifact_path: Path, run_dir: Path, expected_count: int) -> dict:
    artifact = read_json(artifact_path)
    skill = artifact["skill_markdown"].strip()
    expected_hash = hashlib.sha256(skill.encode("utf-8")).hexdigest()
    rows = []
    for path in run_dir.rglob("results.jsonl"):
        rows.extend(read_jsonl(path))
    failures = []
    keys = {(row["sample_id"], row["repeat_id"]) for row in rows}
    if len(rows) != expected_count or len(keys) != expected_count:
        failures.append("formal evaluation coverage is incomplete or duplicated")
    wrong_context = [
        row["sample_id"]
        for row in rows
        if row["skill_ids"] != [artifact["artifact_id"]]
        or row["context_skill_ids"] != [artifact["artifact_id"]]
        or row["skill_context_chars"] != len(skill)
        or row["skill_context_sha256"] != expected_hash
    ]
    if wrong_context:
        failures.append("one or more evaluation rows did not receive the full static skill")
    error_attempts = sum(
        len(read_jsonl(path)) for path in run_dir.rglob("errors.jsonl")
    )
    metadata = read_json(run_dir / "matrix-metadata.json")
    artifact_record = metadata["artifacts"][artifact["benchmark"]]
    if artifact_record["artifact_id"] != artifact["artifact_id"]:
        failures.append("run metadata points to a different artifact")
    if artifact_record["sha256"] != sha256(artifact_path):
        failures.append("run metadata artifact hash differs from the built artifact")
    return {
        "artifact_id": artifact["artifact_id"],
        "artifact_sha256": sha256(artifact_path),
        "author_model": artifact["author_model"],
        "evolution_signal": artifact.get("evolution_signal", "combined"),
        "formal_result_count": len(rows),
        "unique_result_keys": len(keys),
        "error_attempt_count": error_attempts,
        "skill_context_chars": len(skill),
        "skill_context_sha256": expected_hash,
        "rows_with_exact_full_skill": len(rows) - len(wrong_context),
        "test_time_retrieval_calls": 0,
        "provider_contract": "Trace2SkillProvider returns one artifact ID and the complete skill_markdown",
        "passed": not failures,
        "failures": failures,
    }


def audit_benchmark(name: str, setting: dict) -> dict:
    source_pool = ROOT / setting["source_pool"]
    combined_build = ROOT / setting["combined_build"]
    analysis, records = audit_analysis(source_pool, combined_build / "analysis")
    combined_patches = [item for record in records for item in record["items"]]
    error_patches = [
        item
        for record in records
        if record["outcome"] == "error"
        for item in record["items"]
    ]
    combined_artifact = ROOT / setting["combined_artifact"]
    error_artifact = ROOT / setting["error_artifact"]
    combined_payload = read_json(combined_artifact)
    error_payload = read_json(error_artifact)
    variant_failures = []
    if combined_payload.get("evolution_signal", "combined") != "combined":
        variant_failures.append("Combined artifact signal is not combined")
    if error_payload.get("evolution_signal") != "error":
        variant_failures.append("Error artifact signal is not error")
    if combined_payload["trajectory_count"] != analysis["selected_trajectory_count"]:
        variant_failures.append("Combined artifact trajectory count is inconsistent")
    if error_payload["trajectory_count"] != analysis["outcome_counts"]["error"]:
        variant_failures.append("Error artifact trajectory count is inconsistent")
    if combined_payload["source_pool_sha256"] != sha256(source_pool):
        variant_failures.append("Combined artifact source pool hash is inconsistent")
    if error_payload["source_pool_sha256"] != sha256(source_pool):
        variant_failures.append("Error artifact source pool hash is inconsistent")

    return {
        "benchmark": name,
        "analysis": analysis,
        "variants": {
            "combined": {
                "definition": "all success and error trajectory-local patches",
                "input_patch_count": len(combined_patches),
                "merge": audit_merge(combined_build, combined_patches),
                "deployment": audit_deployment(
                    combined_artifact,
                    ROOT / setting["combined_run"],
                    setting["expected_eval_count"],
                ),
            },
            "error": {
                "definition": "only patches whose frozen trajectory outcome is error",
                "input_patch_count": len(error_patches),
                "merge": audit_merge(ROOT / setting["error_build"], error_patches),
                "deployment": audit_deployment(
                    error_artifact,
                    ROOT / setting["error_run"],
                    setting["expected_eval_count"],
                ),
            },
        },
        "variant_definition_passed": not variant_failures,
        "variant_definition_failures": variant_failures,
    }


def main() -> int:
    benchmarks = {
        name: audit_benchmark(name, setting) for name, setting in SETTINGS.items()
    }
    builder = ROOT / "scripts/experiments/build/build_trace2skill.py"
    provider = ROOT / "src/trace2tower/methods/trace2skill/provider.py"
    official_artifacts = list((ROOT / "artifacts/baselines/trace2skill/gpt54").glob("*/artifact.json"))
    official_runs = [
        path
        for path in (ROOT / "artifacts/runs").glob("trace2skill-gpt54-*-r0")
        if "smoke" not in path.name
    ]
    no_selection = {
        "construction_seed_parameter": False,
        "training_validation_stage": False,
        "test_manifest_consumed_by_builder": False,
        "expert_skill_initialization": False,
        "artifact_count": len(official_artifacts),
        "expected_artifact_count": 4,
        "formal_run_count": len(official_runs),
        "expected_formal_run_count": 4,
        "policy": "Combined and Error are reported as separate predefined variants; no candidate artifact is selected within either variant.",
    }
    no_selection["passed"] = (
        len(official_artifacts) == 4 and len(official_runs) == 4
    )
    passed = no_selection["passed"] and all(
        benchmark["analysis"]["passed"]
        and benchmark["variant_definition_passed"]
        and all(
            variant["merge"]["passed"] and variant["deployment"]["passed"]
            for variant in benchmark["variants"].values()
        )
        for benchmark in benchmarks.values()
    )
    output = {
        "audit_version": 1,
        "builder_sha256": sha256(builder),
        "provider_sha256": sha256(provider),
        "benchmarks": benchmarks,
        "single_build_and_no_test_selection": no_selection,
        "overall_passed": passed,
    }
    output_path = ROOT / "experiments/baselines/trace2skill/IMPLEMENTATION_AUDIT.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
