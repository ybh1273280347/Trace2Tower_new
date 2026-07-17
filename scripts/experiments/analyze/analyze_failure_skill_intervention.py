from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.analyze_refinement_test import read_first_completion

REPEAT_IDS = (0, 1, 2)


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("run must be METHOD=PATH")
    return name, Path(raw_path)


def load_method_rows(paths: list[Path], sample_ids: tuple[str, ...]) -> dict[tuple[str, int], dict]:
    rows = {}
    expected = {(sample_id, repeat_id) for sample_id in sample_ids for repeat_id in REPEAT_IDS}
    for path in paths:
        run_rows, _ = read_first_completion(path)
        selected = {key: row for key, row in run_rows.items() if key in expected}
        overlap = set(rows) & set(selected)
        if overlap:
            raise ValueError(f"run paths overlap on {len(overlap)} selected keys")
        rows.update(selected)
    if set(rows) != expected:
        missing = sorted(expected - set(rows))
        raise ValueError(f"method is missing {len(missing)} selected task-repeats: {missing}")
    return rows


def load_trajectories(
    paths: list[Path], sample_ids: tuple[str, ...]
) -> dict[tuple[str, int], dict]:
    expected = {(sample_id, repeat_id) for sample_id in sample_ids for repeat_id in REPEAT_IDS}
    trajectories = {}
    for run_dir in paths:
        for path in run_dir.glob("**/trajectories/*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (str(record["sample_id"]), int(record["repeat_id"]))
            if key not in expected:
                continue
            if key in trajectories:
                raise ValueError(f"duplicate selected trajectory: {key}")
            trajectories[key] = record
    if set(trajectories) != expected:
        missing = sorted(expected - set(trajectories))
        raise ValueError(f"baseline is missing {len(missing)} trajectories: {missing}")
    return trajectories


def summarize_method(rows: dict[tuple[str, int], dict], sample_ids: tuple[str, ...]) -> dict:
    task_metrics = {}
    for sample_id in sample_ids:
        sample_rows = [rows[(sample_id, repeat_id)] for repeat_id in REPEAT_IDS]
        task_metrics[sample_id] = {
            "reward": float(np.mean([row["primary_score"] for row in sample_rows])),
            "full_success": float(
                np.mean([float(row["primary_score"]) >= 0.999 for row in sample_rows])
            ),
            "zero_rate": float(np.mean([float(row["primary_score"]) == 0 for row in sample_rows])),
            "steps": float(np.mean([row["steps"] for row in sample_rows])),
            "invalid_actions": float(np.mean([row["invalid_actions"] for row in sample_rows])),
        }
    return {
        "aggregate": {
            metric: float(np.mean([values[metric] for values in task_metrics.values()]))
            for metric in ("reward", "full_success", "zero_rate", "steps", "invalid_actions")
        },
        "task_metrics": task_metrics,
    }


def compare_tasks(intervention: dict, baseline: dict, sample_ids: tuple[str, ...]) -> dict:
    differences = np.array(
        [
            intervention["task_metrics"][sample_id]["reward"]
            - baseline["task_metrics"][sample_id]["reward"]
            for sample_id in sample_ids
        ]
    )
    return {
        "mean_reward_difference": float(np.mean(differences)),
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "losses": int(np.sum(differences < 0)),
        "per_task_difference": {
            sample_id: float(difference)
            for sample_id, difference in zip(sample_ids, differences, strict=True)
        },
    }


def trajectory_diagnostics(trajectories: dict[tuple[str, int], dict]) -> dict:
    skill_counts = Counter()
    skill_samples: dict[str, set[str]] = defaultdict(set)
    by_sample = defaultdict(
        lambda: {
            "search_actions": 0,
            "repeated_exact_searches": 0,
            "zero_reward_purchases": 0,
            "twenty_step_runs": 0,
        }
    )
    for (sample_id, _), trajectory in trajectories.items():
        seen_queries = set()
        for step in trajectory["steps"]:
            skill_counts.update(step.get("retrieved_context_skill_ids", ()))
            for skill_id in step.get("retrieved_context_skill_ids", ()):
                skill_samples[skill_id].add(sample_id)
            if step["action_name"] == "search_action":
                by_sample[sample_id]["search_actions"] += 1
                query = str(step["action_arguments"].get("keywords", "")).strip().lower()
                if query in seen_queries:
                    by_sample[sample_id]["repeated_exact_searches"] += 1
                seen_queries.add(query)
            if (
                step["action_name"] == "click_action"
                and step["action_arguments"].get("value") == "Buy Now"
                and float(step["reward"]) == 0
            ):
                by_sample[sample_id]["zero_reward_purchases"] += 1
        if len(trajectory["steps"]) >= 20:
            by_sample[sample_id]["twenty_step_runs"] += 1
    return {
        "by_sample": dict(by_sample),
        "retrieved_skills": [
            {
                "skill_id": skill_id,
                "retrieval_count": count,
                "sample_coverage": len(skill_samples[skill_id]),
            }
            for skill_id, count in skill_counts.most_common()
        ],
    }


def load_skill_names(path: Path) -> dict[str, str]:
    tower = json.loads(path.read_text(encoding="utf-8"))
    return {
        card["skill_id"]: card["name"] for key in ("mid_cards", "high_cards") for card in tower[key]
    }


def markdown_report(payload: dict) -> str:
    summaries = payload["summary"]
    sample_ids = payload["sample_ids"]
    lines = [
        "# Failure-Set Skill Intervention",
        "",
        "> Post-hoc diagnostic on six repeat-0 common failures. "
        "This is not a held-out generalization estimate.",
        "",
        "## Aggregate result",
        "",
        "| Method | Mean reward | Full success | Zero reward | Mean steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, summary in summaries.items():
        aggregate = summary["aggregate"]
        lines.append(
            f"| {method} | {aggregate['reward']:.4f} | {aggregate['full_success']:.1%} | "
            f"{aggregate['zero_rate']:.1%} | {aggregate['steps']:.2f} |"
        )
    lines.extend(
        (
            "",
            "## Per-task mean reward across repeats 0-2",
            "",
            "| Sample | Final T1 | SkillX | Generic manual | Recovery skill |",
            "|---|---:|---:|---:|---:|",
        )
    )
    display_methods = ("final_t1", "skillx", "generic_manual", "recovery_skill")
    for sample_id in sample_ids:
        values = [
            summaries[method]["task_metrics"][sample_id]["reward"] for method in display_methods
        ]
        lines.append(
            f"| {sample_id} | {values[0]:.4f} | {values[1]:.4f} | "
            f"{values[2]:.4f} | {values[3]:.4f} |"
        )
    lines.extend(
        (
            "",
            "## Paired task-level comparison",
            "",
            "| Recovery skill minus | Mean delta | W/T/L |",
            "|---|---:|---:|",
        )
    )
    for baseline, comparison in payload["paired_comparisons"].items():
        lines.append(
            f"| {baseline} | {comparison['mean_reward_difference']:+.4f} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} |"
        )
    lines.extend(
        (
            "",
            "## What Final T1 retrieved on these failures",
            "",
            "| Skill | Name | Retrievals | Sample coverage |",
            "|---|---|---:|---:|",
        )
    )
    for skill in payload["baseline_diagnostics"]["retrieved_skills"][:12]:
        name = skill["name"].replace("|", "\\|")
        lines.append(
            f"| `{skill['skill_id']}` | {name} | {skill['retrieval_count']} | "
            f"{skill['sample_coverage']}/6 |"
        )
    lines.extend(
        (
            "",
            "## Why the retrieved skills did not break through",
            "",
            "| Retrieved guidance family | What it supplies | Missing control signal |",
            "|---|---|---|",
            "| Search/open/refine | A plausible next query or candidate | "
            "No memory of rejected queries and candidates; no remaining-step budget |",
            "| Configure/purchase | Select visible options, then buy | "
            "No hard rule that an unknown requirement blocks purchase |",
            "| Detail inspection | Check hidden evidence | "
            "Retrieved only 8 times with 1/6 sample coverage |",
            "| Return to search | Backtrack after a mismatch | "
            "No systematic pivot order or stop condition |",
        )
    )
    lines.extend(
        (
            "",
            "## Failure behavior",
            "",
            "| Sample | Pattern | Repeated exact searches | Zero-reward purchases | 20-step runs |",
            "|---|---|---:|---:|---:|",
        )
    )
    for sample_id in sample_ids:
        diagnostics = payload["baseline_diagnostics"]["by_sample"][sample_id]
        pattern = (
            "premature purchase" if diagnostics["zero_reward_purchases"] else "search exhaustion"
        )
        lines.append(
            f"| {sample_id} | {pattern} | {diagnostics['repeated_exact_searches']} | "
            f"{diagnostics['zero_reward_purchases']} | {diagnostics['twenty_step_runs']} |"
        )
    lines.extend(
        (
            "",
            "## What changed under the recovery skill",
            "",
            "| Sample | Mean reward | Search actions | Zero-reward purchases | 20-step runs |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for sample_id in sample_ids:
        reward = summaries["recovery_skill"]["task_metrics"][sample_id]["reward"]
        diagnostics = payload["intervention_diagnostics"]["by_sample"][sample_id]
        lines.append(
            f"| {sample_id} | {reward:.4f} | {diagnostics['search_actions']} | "
            f"{diagnostics['zero_reward_purchases']} | {diagnostics['twenty_step_runs']} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "The retrieved skills repeatedly cover search, opening a likely match, "
            "selecting options, and buying. They do not strongly encode a cross-step "
            "rejection ledger, a hard gate for unknown constraints, or a bounded "
            "recovery plan. The intervention tests those missing control signals "
            "without adding product answers.",
            "",
            "The recovery skill raises mean reward from 0 to 0.1111, entirely through "
            "`webshop:969`, which succeeds in two of three repeats after rejecting "
            "plausible but mismatched candidates. The other premature-purchase task "
            "still buys the wrong candidate, and all four search-exhaustion tasks still "
            "reach 20 steps. Static guidance can help, but it does not reliably enforce "
            "the ledger, constraint gate, or query budget across steps.",
            "",
            "Because the six tasks were selected after observing failure, any gain is "
            "evidence of a recoverable guidance gap on this failure class, not an "
            "unbiased estimate of general performance.",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--baseline-method", default="final_t1")
    parser.add_argument("--intervention-method", default="recovery_skill")
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    options = parser.parse_args()

    sample_ids = tuple(sorted(set(options.sample_id)))
    paths_by_method: dict[str, list[Path]] = defaultdict(list)
    for method, path in options.run:
        paths_by_method[method].append(path)
    required_methods = {"final_t1", "skillx", "generic_manual", "recovery_skill"}
    if set(paths_by_method) != required_methods:
        raise ValueError(f"runs must contain exactly these methods: {sorted(required_methods)}")

    rows_by_method = {
        method: load_method_rows(paths, sample_ids)
        for method, paths in sorted(paths_by_method.items())
    }
    summaries = {
        method: summarize_method(rows, sample_ids) for method, rows in rows_by_method.items()
    }
    intervention = summaries[options.intervention_method]
    comparisons = {
        baseline: compare_tasks(intervention, summaries[baseline], sample_ids)
        for baseline in ("final_t1", "skillx", "generic_manual")
    }

    diagnostics = trajectory_diagnostics(
        load_trajectories(paths_by_method[options.baseline_method], sample_ids)
    )
    intervention_diagnostics = trajectory_diagnostics(
        load_trajectories(paths_by_method[options.intervention_method], sample_ids)
    )
    skill_names = load_skill_names(options.tower)
    for skill in diagnostics["retrieved_skills"]:
        skill["name"] = skill_names.get(skill["skill_id"], "Unknown skill")

    payload = {
        "protocol_id": "webshop-common-zero-recovery-skill-repeat3-v1",
        "selection": "post_hoc_repeat0_common_zero_final_t1_and_skillx",
        "statistical_unit": "task_mean_across_real_repeats_0_1_2",
        "sample_ids": sample_ids,
        "runs": {
            method: [path.as_posix() for path in paths]
            for method, paths in sorted(paths_by_method.items())
        },
        "summary": summaries,
        "paired_comparisons": comparisons,
        "baseline_diagnostics": diagnostics,
        "intervention_diagnostics": intervention_diagnostics,
    }
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    options.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    options.output_md.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "paired_comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
