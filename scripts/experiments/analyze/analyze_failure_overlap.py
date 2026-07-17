from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.analyze.analyze_refinement_test import read_first_completion


def read_trajectories(run_dir: Path) -> dict[str, dict]:
    trajectories = {}
    for path in run_dir.glob("**/trajectories/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record["repeat_id"]) == 0:
            trajectories[str(record["sample_id"])] = record
    if len(trajectories) != 100:
        raise ValueError("failure audit requires 100 repeat0 trajectories")
    return trajectories


def set_overlap(left: set[str], right: set[str]) -> dict:
    intersection = left & right
    union = left | right
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "left_coverage": len(intersection) / len(left) if left else 1.0,
        "right_coverage": len(intersection) / len(right) if right else 1.0,
    }


def sample_record(
    sample_id: str,
    final_rows: dict[str, dict],
    skillx_rows: dict[str, dict],
    final_trajectories: dict[str, dict],
    skillx_trajectories: dict[str, dict],
) -> dict:
    final_trajectory = final_trajectories[sample_id]
    skillx_trajectory = skillx_trajectories[sample_id]
    final_last = final_trajectory["steps"][-1] if final_trajectory["steps"] else None
    skillx_last = skillx_trajectory["steps"][-1] if skillx_trajectory["steps"] else None
    return {
        "sample_id": sample_id,
        "task_goal": final_trajectory["task_goal"],
        "final_score": final_rows[sample_id]["primary_score"],
        "skillx_score": skillx_rows[sample_id]["primary_score"],
        "final_steps": final_rows[sample_id]["steps"],
        "skillx_steps": skillx_rows[sample_id]["steps"],
        "final_last_action": (
            {
                "name": final_last["action_name"],
                "arguments": final_last["action_arguments"],
            }
            if final_last
            else None
        ),
        "skillx_last_action": (
            {
                "name": skillx_last["action_name"],
                "arguments": skillx_last["action_arguments"],
            }
            if skillx_last
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-run", type=Path, required=True)
    parser.add_argument("--skillx-run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--noskill-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    options = parser.parse_args()

    run_paths = {
        "final": options.final_run,
        "skillx": options.skillx_run,
        "legacy": options.legacy_run,
        "noskill": options.noskill_run,
    }
    rows = {}
    for name, path in run_paths.items():
        run_rows, _ = read_first_completion(path)
        selected = {
            sample_id: row for (sample_id, repeat_id), row in run_rows.items() if repeat_id == 0
        }
        if len(selected) != 100:
            raise ValueError(f"{name} must cover 100 repeat0 tasks")
        rows[name] = selected
    if any(set(run_rows) != set(rows["final"]) for run_rows in rows.values()):
        raise ValueError("failure audit runs cover different Test-A tasks")

    failures = {
        name: {
            "zero": {
                sample_id for sample_id, row in run_rows.items() if float(row["primary_score"]) == 0
            },
            "nonfull": {
                sample_id
                for sample_id, row in run_rows.items()
                if float(row["primary_score"]) < 0.999
            },
        }
        for name, run_rows in rows.items()
    }
    overlaps = {
        kind: {
            f"final_vs_{name}": set_overlap(failures["final"][kind], failures[name][kind])
            for name in ("skillx", "legacy", "noskill")
        }
        for kind in ("zero", "nonfull")
    }
    sample_ids = tuple(sorted(rows["final"]))
    final_scores = np.array([rows["final"][sample_id]["primary_score"] for sample_id in sample_ids])
    score_correlations = {
        name: float(
            np.corrcoef(
                final_scores,
                [rows[name][sample_id]["primary_score"] for sample_id in sample_ids],
            )[0, 1]
        )
        for name in ("skillx", "legacy", "noskill")
    }

    final_trajectories = read_trajectories(options.final_run)
    skillx_trajectories = read_trajectories(options.skillx_run)
    categories = {
        "common_zero": failures["final"]["zero"] & failures["skillx"]["zero"],
        "final_only_zero": failures["final"]["zero"] - failures["skillx"]["zero"],
        "skillx_only_zero": failures["skillx"]["zero"] - failures["final"]["zero"],
        "final_only_nonfull": failures["final"]["nonfull"] - failures["skillx"]["nonfull"],
        "skillx_only_nonfull": failures["skillx"]["nonfull"] - failures["final"]["nonfull"],
    }
    category_records = {
        name: [
            sample_record(
                sample_id,
                rows["final"],
                rows["skillx"],
                final_trajectories,
                skillx_trajectories,
            )
            for sample_id in sorted(sample_ids)
        ]
        for name, sample_ids in categories.items()
    }
    payload = {
        "protocol_id": "webshop-test-a-failure-overlap-v1",
        "agent_model": "deepseek-v4-flash",
        "repeat_id": 0,
        "runs": {name: path.as_posix() for name, path in run_paths.items()},
        "overlaps": overlaps,
        "score_correlations": score_correlations,
        "final_skillx_categories": category_records,
    }
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    options.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    zero = overlaps["zero"]["final_vs_skillx"]
    nonfull = overlaps["nonfull"]["final_vs_skillx"]
    lines = [
        "# Test-A Failure Overlap",
        "",
        "## Summary",
        "",
        "| Failure definition | Final | SkillX | Intersection | Jaccard | "
        "Final covered | SkillX covered |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Zero reward | {zero['left_count']} | {zero['right_count']} | "
        f"{zero['intersection_count']} | {zero['jaccard']:.3f} | "
        f"{zero['left_coverage']:.1%} | {zero['right_coverage']:.1%} |",
        f"| Non-full reward | {nonfull['left_count']} | {nonfull['right_count']} | "
        f"{nonfull['intersection_count']} | {nonfull['jaccard']:.3f} | "
        f"{nonfull['left_coverage']:.1%} | {nonfull['right_coverage']:.1%} |",
        "",
        f"Final/SkillX reward correlation is `{score_correlations['skillx']:.3f}`.",
        "",
        "## Common zero-reward tasks",
        "",
        "| Sample | Goal | Final steps | SkillX steps |",
        "|---|---|---:|---:|",
    ]
    for record in category_records["common_zero"]:
        goal = record["task_goal"].replace("|", "\\|")
        lines.append(
            f"| {record['sample_id']} | {goal} | {record['final_steps']} | "
            f"{record['skillx_steps']} |"
        )
    lines.extend(("", "## Method-specific hard failures", ""))
    for name in ("final_only_zero", "skillx_only_zero"):
        lines.append(f"### {name}")
        lines.append("")
        for record in category_records[name]:
            lines.append(f"- `{record['sample_id']}`: {record['task_goal']}")
        if not category_records[name]:
            lines.append("- None")
        lines.append("")
    options.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["overlaps"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
