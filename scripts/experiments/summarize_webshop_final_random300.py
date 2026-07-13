from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np
import yaml


FULL_SUCCESS_THRESHOLD = 0.999
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class Condition:
    label: str
    model: str
    method: str
    run_id: str
    method_dir: str
    artifact_name: str | None = None


CONDITIONS = (
    Condition("no_skill", "deepseek-v4-flash", "no_skill", "webshop-final-random300-flash-noskill-v1", "no_skill"),
    Condition("flat_cap3", "deepseek-v4-flash", "flat_skill_summary", "webshop-final-random300-flash-flat-cap3-v1", "flat_skill_summary", "flat_cap3"),
    Condition("success_tower_cap3", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-success-tower-cap3-v1", "trace2tower_static", "success_only_tower_cap3"),
    Condition("mixed_tower_cap3", "deepseek-v4-flash", "trace2tower_static", "webshop-final-random300-flash-mixed-tower-cap3-v1", "trace2tower_static", "mixed_tower_cap3"),
    Condition("no_skill", "deepseek-v4-pro", "no_skill", "webshop-final-random300-pro-noskill-v1", "no_skill"),
    Condition("flat_cap3", "deepseek-v4-pro", "flat_skill_summary", "webshop-final-random300-pro-flat-cap3-v1", "flat_skill_summary", "flat_cap3"),
    Condition("success_tower_cap3", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-success-tower-cap3-v1", "trace2tower_static", "success_only_tower_cap3"),
    Condition("mixed_tower_cap3", "deepseek-v4-pro", "trace2tower_static", "webshop-final-random300-pro-mixed-tower-cap3-v1", "trace2tower_static", "mixed_tower_cap3"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as output:
        temporary = Path(output.name)
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_results(runs_root: Path, condition: Condition) -> tuple[list[dict], dict[str, str]]:
    root = runs_root / condition.run_id / "webshop" / "test" / condition.method_dir
    paths = sorted(root.glob("shard-*/results.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"{condition.run_id} does not have ten result shards")
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    error_count = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in root.glob("shard-*/errors.jsonl")
    )
    if error_count:
        raise ValueError(f"{condition.run_id} has {error_count} error records")
    return rows, {path.as_posix(): sha256_file(path) for path in paths}


def validate_condition(
    condition: Condition,
    rows: list[dict],
    expected_keys: set[tuple[str, int]],
    selection: dict,
    runs_root: Path,
) -> dict:
    keys = [(str(row["sample_id"]), int(row["repeat_id"])) for row in rows]
    if len(rows) != len(expected_keys) or len(set(keys)) != len(keys):
        raise ValueError(f"{condition.run_id} has incomplete or duplicate results")
    if set(keys) != expected_keys:
        raise ValueError(f"{condition.run_id} result keys differ from preregistration")
    if any(
        row["run_id"] != condition.run_id
        or row["benchmark"] != "webshop"
        or row["split"] != "test"
        or row["method"] != condition.method
        or row.get("error") is not None
        for row in rows
    ):
        raise ValueError(f"{condition.run_id} contains scope or method mismatches")

    config_path = runs_root / condition.run_id / "resolved-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["agent_model"] != condition.model:
        raise ValueError(f"{condition.run_id} agent model differs from preregistration")
    if sorted(config.get("selection", {}).get("repeat_ids", [])) != [0, 1, 2]:
        raise ValueError(f"{condition.run_id} repeat selection differs from preregistration")
    if condition.label == "flat_cap3" and int(config["method"]["flat_top_k"]) != 3:
        raise ValueError("Flat final run is not cap 3")
    if "tower" in condition.label and int(config["method"]["direct_mid_top_k"]) != 3:
        raise ValueError("Tower final run is not cap 3")
    if condition.artifact_name:
        artifact = config["artifacts"]["webshop"]
        expected = selection["method_artifacts"][condition.artifact_name]
        if artifact["sha256"] != expected["sha256"]:
            raise ValueError(f"{condition.run_id} artifact hash differs from preregistration")
    elif config["artifacts"]:
        raise ValueError("NoSkill final run unexpectedly binds an artifact")
    return {
        "run_id": condition.run_id,
        "resolved_config_path": config_path.as_posix(),
        "resolved_config_sha256": sha256_file(config_path),
        "episode_count": len(rows),
        "unique_key_count": len(set(keys)),
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "error_count": 0,
    }


def aggregate(rows: list[dict]) -> dict:
    steps = sum(int(row["steps"]) for row in rows)
    invalid = sum(int(row["invalid_actions"]) for row in rows)
    observed_input = [row["input_tokens"] for row in rows if row.get("input_tokens") is not None]
    observed_output = [row["output_tokens"] for row in rows if row.get("output_tokens") is not None]
    return {
        "episode_count": len(rows),
        "task_count": len({row["sample_id"] for row in rows}),
        "mean_reward": fmean(float(row["primary_score"]) for row in rows),
        "full_success_rate": fmean(
            float(row["primary_score"] >= FULL_SUCCESS_THRESHOLD) for row in rows
        ),
        "completion_rate": fmean(row["finish_reason"] == "completed" for row in rows),
        "mean_steps": fmean(int(row["steps"]) for row in rows),
        "invalid_action_rate": invalid / max(steps, 1),
        "mean_input_tokens": fmean(observed_input) if observed_input else None,
        "mean_output_tokens": fmean(observed_output) if observed_output else None,
        "mean_skill_context_chars": fmean(int(row["skill_context_chars"]) for row in rows),
    }


def task_values(rows: list[dict], value) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(float(value(row)))
    if any(len(values) != 3 for values in grouped.values()):
        raise ValueError("every task must have exactly three repeats")
    return {sample_id: fmean(values) for sample_id, values in grouped.items()}


def bootstrap_interval(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_SAMPLES, len(array)))
    estimates = array[indices].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def paired_comparison(
    baseline: list[dict], candidate: list[dict], *, bootstrap_seed: int
) -> dict:
    baseline_by_key = {
        (str(row["sample_id"]), int(row["repeat_id"])): row for row in baseline
    }
    candidate_by_key = {
        (str(row["sample_id"]), int(row["repeat_id"])): row for row in candidate
    }
    if set(baseline_by_key) != set(candidate_by_key):
        raise ValueError("paired comparison requires identical episode keys")
    task_ids = sorted({sample_id for sample_id, _ in baseline_by_key})
    reward_differences = []
    success_differences = []
    for sample_id in task_ids:
        keys = sorted(key for key in baseline_by_key if key[0] == sample_id)
        reward_differences.append(
            fmean(
                float(candidate_by_key[key]["primary_score"])
                - float(baseline_by_key[key]["primary_score"])
                for key in keys
            )
        )
        success_differences.append(
            fmean(
                float(candidate_by_key[key]["primary_score"] >= FULL_SUCCESS_THRESHOLD)
                - float(baseline_by_key[key]["primary_score"] >= FULL_SUCCESS_THRESHOLD)
                for key in keys
            )
        )
    rng = np.random.default_rng(bootstrap_seed)
    episode_differences = [
        float(candidate_by_key[key]["primary_score"])
        - float(baseline_by_key[key]["primary_score"])
        for key in sorted(baseline_by_key)
    ]
    return {
        "pair_count": len(episode_differences),
        "task_count": len(task_ids),
        "reward_difference": fmean(episode_differences),
        "reward_confidence_interval": bootstrap_interval(reward_differences, rng),
        "full_success_rate_difference": fmean(success_differences),
        "full_success_rate_confidence_interval": bootstrap_interval(
            success_differences, rng
        ),
        "candidate_wins": sum(value > 0 for value in episode_differences),
        "ties": sum(value == 0 for value in episode_differences),
        "candidate_losses": sum(value < 0 for value in episode_differences),
        "mean_step_difference": fmean(
            int(candidate_by_key[key]["steps"]) - int(baseline_by_key[key]["steps"])
            for key in baseline_by_key
        ),
        "mean_input_token_difference": fmean(
            int(candidate_by_key[key]["input_tokens"])
            - int(baseline_by_key[key]["input_tokens"])
            for key in baseline_by_key
        ),
    }


def interaction(
    flash_baseline: list[dict],
    flash_candidate: list[dict],
    pro_baseline: list[dict],
    pro_candidate: list[dict],
    *,
    bootstrap_seed: int,
) -> dict:
    maps = [
        {(row["sample_id"], int(row["repeat_id"])): row for row in rows}
        for rows in (flash_baseline, flash_candidate, pro_baseline, pro_candidate)
    ]
    keys = set(maps[0])
    if any(set(mapping) != keys for mapping in maps[1:]):
        raise ValueError("interaction requires identical episode keys")
    task_differences = []
    episode_differences = []
    for sample_id in sorted({key[0] for key in keys}):
        current = []
        for key in sorted(item for item in keys if item[0] == sample_id):
            flash_uplift = float(maps[1][key]["primary_score"]) - float(
                maps[0][key]["primary_score"]
            )
            pro_uplift = float(maps[3][key]["primary_score"]) - float(
                maps[2][key]["primary_score"]
            )
            current.append(pro_uplift - flash_uplift)
        episode_differences.extend(current)
        task_differences.append(fmean(current))
    return {
        "difference_in_differences": fmean(episode_differences),
        "confidence_interval": bootstrap_interval(
            task_differences, np.random.default_rng(bootstrap_seed)
        ),
        "task_count": len(task_differences),
        "episode_count": len(episode_differences),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# WebShop Final Random-300 Results",
        "",
        f"Preregistered selection: `{report['selection_id']}`  ",
        "Tasks: 300 independent tasks, three repeats each  ",
        "Full success: reward >= 0.999",
        "",
    ]
    for model in report["models"]:
        lines.extend(
            [
                f"## {model}",
                "",
                "| Method | Mean reward | Full success | Completion | Steps | Input tokens | Context chars |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, values in report["models"][model]["combined"].items():
            lines.append(
                f"| {label} | {values['mean_reward']:.4f} | "
                f"{values['full_success_rate']:.1%} | {values['completion_rate']:.1%} | "
                f"{values['mean_steps']:.2f} | {values['mean_input_tokens']:.0f} | "
                f"{values['mean_skill_context_chars']:.0f} |"
            )
        lines.extend(
            [
                "",
                "| Comparison | Reward difference | Reward 95% CI | Full-success difference | Success 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, values in report["models"][model]["comparisons"].items():
            reward_ci = values["reward_confidence_interval"]
            success_ci = values["full_success_rate_confidence_interval"]
            lines.append(
                f"| {label} | {values['reward_difference']:+.4f} | "
                f"[{reward_ci[0]:+.4f}, {reward_ci[1]:+.4f}] | "
                f"{values['full_success_rate_difference']:+.1%} | "
                f"[{success_ci[0]:+.1%}, {success_ci[1]:+.1%}] |"
            )
        lines.append("")
    lines.extend(["## Per-seed aggregates", ""])
    for seed, values in report["per_seed"].items():
        lines.extend(
            [
                f"### Seed {seed}",
                "",
                "| Model | Method | Mean reward | Full success | Completion |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for model, methods in values.items():
            for label, aggregate_values in methods.items():
                lines.append(
                    f"| {model} | {label} | {aggregate_values['mean_reward']:.4f} | "
                    f"{aggregate_values['full_success_rate']:.1%} | "
                    f"{aggregate_values['completion_rate']:.1%} |"
                )
        lines.append("")
    lines.extend(["## Model-by-skill interactions", ""])
    for label, values in report["model_by_skill_interactions"].items():
        ci = values["confidence_interval"]
        lines.append(
            f"- `{label}`: {values['difference_in_differences']:+.4f}, "
            f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
    lines.append("")
    return "\n".join(lines)


def main(options: argparse.Namespace) -> int:
    selection = json.loads(options.selection.read_text(encoding="utf-8"))
    manifest_path = Path(selection["execution_manifest_path"])
    if sha256_file(manifest_path) != selection["execution_manifest_sha256"]:
        raise ValueError("execution manifest hash differs from preregistration")
    sample_ids = [
        sample_id
        for group in selection["selections"]
        for sample_id in group["sample_ids"]
    ]
    expected_keys = {(sample_id, repeat_id) for sample_id in sample_ids for repeat_id in (0, 1, 2)}
    seed_by_sample = {
        sample_id: str(group["seed"])
        for group in selection["selections"]
        for sample_id in group["sample_ids"]
    }

    rows_by_condition: dict[tuple[str, str], list[dict]] = {}
    audits = {}
    result_hashes = {}
    for condition in CONDITIONS:
        rows, hashes = load_results(options.runs_root, condition)
        key = (condition.model, condition.label)
        rows_by_condition[key] = rows
        audits[f"{condition.model}:{condition.label}"] = validate_condition(
            condition, rows, expected_keys, selection, options.runs_root
        )
        result_hashes[f"{condition.model}:{condition.label}"] = hashes

    models = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        combined = {
            label: aggregate(rows_by_condition[(model, label)])
            for label in ("no_skill", "flat_cap3", "success_tower_cap3", "mixed_tower_cap3")
        }
        baseline = rows_by_condition[(model, "no_skill")]
        comparisons = {
            f"{label}_vs_no_skill": paired_comparison(
                baseline,
                rows_by_condition[(model, label)],
                bootstrap_seed=BOOTSTRAP_SEED,
            )
            for label in ("flat_cap3", "success_tower_cap3", "mixed_tower_cap3")
        }
        comparisons["mixed_vs_success_tower"] = paired_comparison(
            rows_by_condition[(model, "success_tower_cap3")],
            rows_by_condition[(model, "mixed_tower_cap3")],
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        comparisons["success_tower_vs_flat"] = paired_comparison(
            rows_by_condition[(model, "flat_cap3")],
            rows_by_condition[(model, "success_tower_cap3")],
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        models[model] = {"combined": combined, "comparisons": comparisons}

    per_seed = {}
    per_seed_comparisons = {}
    for group in selection["selections"]:
        seed = str(group["seed"])
        selected = set(group["sample_ids"])
        per_seed[seed] = {
            model: {
                label: aggregate(
                    [
                        row
                        for row in rows_by_condition[(model, label)]
                        if row["sample_id"] in selected
                    ]
                )
                for label in ("no_skill", "flat_cap3", "success_tower_cap3", "mixed_tower_cap3")
            }
            for model in ("deepseek-v4-flash", "deepseek-v4-pro")
        }
        per_seed_comparisons[seed] = {}
        for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
            baseline = [
                row
                for row in rows_by_condition[(model, "no_skill")]
                if row["sample_id"] in selected
            ]
            per_seed_comparisons[seed][model] = {
                f"{label}_vs_no_skill": paired_comparison(
                    baseline,
                    [
                        row
                        for row in rows_by_condition[(model, label)]
                        if row["sample_id"] in selected
                    ],
                    bootstrap_seed=BOOTSTRAP_SEED,
                )
                for label in (
                    "flat_cap3",
                    "success_tower_cap3",
                    "mixed_tower_cap3",
                )
            }

    flash_no = rows_by_condition[("deepseek-v4-flash", "no_skill")]
    pro_no = rows_by_condition[("deepseek-v4-pro", "no_skill")]
    interactions = {
        label: interaction(
            flash_no,
            rows_by_condition[("deepseek-v4-flash", label)],
            pro_no,
            rows_by_condition[("deepseek-v4-pro", label)],
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        for label in ("flat_cap3", "success_tower_cap3", "mixed_tower_cap3")
    }
    cross_model = {
        "pro_vs_flash_no_skill": paired_comparison(
            flash_no,
            pro_no,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
    }
    report = {
        "selection_id": selection["selection_id"],
        "selection_path": options.selection.as_posix(),
        "selection_sha256": sha256_file(options.selection),
        "execution_manifest_path": manifest_path.as_posix(),
        "execution_manifest_sha256": selection["execution_manifest_sha256"],
        "full_success_threshold": FULL_SUCCESS_THRESHOLD,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "expected_task_count": 300,
        "expected_episode_count_per_condition": 900,
        "audits": audits,
        "result_source_hashes": result_hashes,
        "models": models,
        "per_seed": per_seed,
        "per_seed_comparisons": per_seed_comparisons,
        "cross_model_comparisons": cross_model,
        "model_by_skill_interactions": interactions,
    }
    write_json(options.output_dir / "report.json", report)
    write_text(options.output_dir / "report.md", render_markdown(report))
    print(json.dumps({"models": models, "interactions": interactions}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/experiments/webshop_final_random300_v1.json"),
    )
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluations/webshop-final-random300-v1"),
    )
    raise SystemExit(main(parser.parse_args()))
