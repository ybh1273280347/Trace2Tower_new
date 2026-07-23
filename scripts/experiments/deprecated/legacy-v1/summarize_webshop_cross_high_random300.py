from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.experiments.analyze.summarize_webshop_final_random300 import (
    BOOTSTRAP_SEED,
    Condition,
    aggregate,
    load_results,
    paired_comparison,
    sha256_file,
    write_json,
)


CONDITIONS = (
    Condition("success_full", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-success-tower-cap3-v1", "trace2tower_static"),
    Condition("success_mid", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-success-tower-cap3-mid-only-v1", "trace2tower_static"),
    Condition("success_cross", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-success-mid-mixed-high-cap3-v1", "trace2tower_static"),
    Condition("mixed_full", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-mixed-tower-cap3-v1", "trace2tower_static"),
    Condition("mixed_mid", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-mixed-tower-cap3-mid-only-v1", "trace2tower_static"),
    Condition("mixed_cross", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-mixed-mid-success-high-cap3-v1", "trace2tower_static"),
    Condition("success_full", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-success-tower-cap3-v1", "trace2tower_static"),
    Condition("success_mid", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-success-tower-cap3-mid-only-v1", "trace2tower_static"),
    Condition("success_cross", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-success-mid-mixed-high-cap3-v1", "trace2tower_static"),
    Condition("mixed_full", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-mixed-tower-cap3-v1", "trace2tower_static"),
    Condition("mixed_mid", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-mixed-tower-cap3-mid-only-v1", "trace2tower_static"),
    Condition("mixed_cross", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-mixed-mid-success-high-cap3-v1", "trace2tower_static"),
)


def validate(condition: Condition, rows: list[dict], expected_keys: set[tuple[str, int]], runs_root: Path) -> dict:
    keys = [(str(row["sample_id"]), int(row["repeat_id"])) for row in rows]
    if len(rows) != 900 or len(set(keys)) != 900 or set(keys) != expected_keys:
        raise ValueError(f"{condition.run_id} has incomplete, duplicate, or unexpected keys")
    if any(
        row["run_id"] != condition.run_id
        or row["benchmark"] != "webshop"
        or row["split"] != "test"
        or row["method"] != "trace2tower_static"
        or row.get("error") is not None
        for row in rows
    ):
        raise ValueError(f"{condition.run_id} contains scope, method, or error mismatches")
    config_path = runs_root / condition.run_id / "resolved-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["agent_model"] != condition.model:
        raise ValueError(f"{condition.run_id} agent model mismatch")
    if sorted(config["selection"]["repeat_ids"]) != [0, 1, 2]:
        raise ValueError(f"{condition.run_id} repeat selection mismatch")
    if int(config["method"]["direct_mid_top_k"]) != 3:
        raise ValueError(f"{condition.run_id} is not cap3")
    artifact = config["artifacts"]["webshop"]
    artifact_path = Path(artifact["path"])
    if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["sha256"]:
        raise ValueError(f"{condition.run_id} artifact is missing or has changed")
    return {
        "run_id": condition.run_id,
        "episode_count": 900,
        "unique_key_count": 900,
        "error_count": 0,
        "resolved_config_sha256": sha256_file(config_path),
        "artifact_path": artifact["path"],
        "artifact_sha256": artifact["sha256"],
    }


def main(options: argparse.Namespace) -> int:
    selection = json.loads(options.selection.read_text(encoding="utf-8"))
    sample_ids = [sample_id for group in selection["selections"] for sample_id in group["sample_ids"]]
    expected_keys = {(sample_id, repeat_id) for sample_id in sample_ids for repeat_id in (0, 1, 2)}
    rows_by_key = {}
    audits = {}
    hashes = {}
    for condition in CONDITIONS:
        rows, result_hashes = load_results(options.runs_root, condition)
        key = (condition.model, condition.label)
        rows_by_key[key] = rows
        audits[f"{condition.model}:{condition.label}"] = validate(condition, rows, expected_keys, options.runs_root)
        hashes[f"{condition.model}:{condition.label}"] = result_hashes

    models = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        combined = {label: aggregate(rows_by_key[(model, label)]) for label in (
            "success_mid", "success_full", "success_cross", "mixed_mid", "mixed_full", "mixed_cross"
        )}
        comparisons = {
            "mixed_high_on_success_mid_vs_success_mid": paired_comparison(rows_by_key[(model, "success_mid")], rows_by_key[(model, "success_cross")], bootstrap_seed=BOOTSTRAP_SEED),
            "success_high_on_mixed_mid_vs_mixed_mid": paired_comparison(rows_by_key[(model, "mixed_mid")], rows_by_key[(model, "mixed_cross")], bootstrap_seed=BOOTSTRAP_SEED),
            "mixed_high_vs_success_high_on_success_mid": paired_comparison(rows_by_key[(model, "success_full")], rows_by_key[(model, "success_cross")], bootstrap_seed=BOOTSTRAP_SEED),
            "success_high_vs_mixed_high_on_mixed_mid": paired_comparison(rows_by_key[(model, "mixed_full")], rows_by_key[(model, "mixed_cross")], bootstrap_seed=BOOTSTRAP_SEED),
        }
        models[model] = {"combined": combined, "comparisons": comparisons}
    report = {
        "selection_id": selection["selection_id"],
        "expected_task_count": 300,
        "expected_episode_count_per_condition": 900,
        "full_success_threshold": 0.999,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "audits": audits,
        "result_source_hashes": hashes,
        "models": models,
    }
    write_json(options.output, report)
    print(json.dumps(models, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("configs/experiments/webshop_final_random300_v1.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/webshop-cross-high-random300-v1/report.json"))
    raise SystemExit(main(parser.parse_args()))
