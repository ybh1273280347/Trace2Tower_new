from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sum_tower_usage(payload: dict) -> dict:
    usage = payload["usage"]
    input_tokens = sum(item["input_tokens"] for item in usage)
    output_tokens = sum(item["output_tokens"] for item in usage)
    cached_input_tokens = sum(item.get("cached_input_tokens") or 0 for item in usage)
    return {
        "calls": len(usage),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": input_tokens - cached_input_tokens,
    }


def normalize_skillx_usage(payload: dict) -> dict:
    usage = payload["llm_usage"]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    cached_input_tokens = usage["cached_input_tokens"]
    return {
        "calls": usage["calls"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": input_tokens - cached_input_tokens,
        "validation_failures": usage["validation_failures"],
        "transport_failures": usage["transport_failures"],
    }


def relative_usage(tower: dict, skillx: dict) -> dict:
    metrics = (
        "calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
    )
    return {
        metric: {
            "ratio": tower[metric] / skillx[metric],
            "relative_change": tower[metric] / skillx[metric] - 1,
        }
        for metric in metrics
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower-cards", type=Path, required=True)
    parser.add_argument("--tower-report", type=Path, required=True)
    parser.add_argument("--skillx-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    tower_cards = load_json(options.tower_cards)
    tower_report = load_json(options.tower_report)
    skillx_report = load_json(options.skillx_report)
    tower_usage = sum_tower_usage(tower_cards)
    skillx_usage = normalize_skillx_usage(skillx_report)

    payload = {
        "protocol_id": "webshop-p100-construction-llm-usage-v1",
        "scope": "skill-construction LLM chat usage only",
        "excluded": [
            "trajectory collection agent usage",
            "embedding usage",
            "runtime evaluation usage",
            "currency cost estimates",
        ],
        "comparability_notes": [
            "Both methods use gpt-5.4 for skill construction.",
            "Trace2Tower renders Mid and High cards after deterministic graph construction.",
            "SkillX usage covers upstream extraction, validation, planning, and merging.",
            "The evidence sets differ: Trace2Tower uses mixed trajectories while SkillX uses full-success trajectories.",
            "Cached-token prices are provider-specific, so token usage does not prove a billing-cost ratio.",
        ],
        "sources": {
            "trace2tower_cards": options.tower_cards.as_posix(),
            "trace2tower_report": options.tower_report.as_posix(),
            "skillx_report": options.skillx_report.as_posix(),
        },
        "trace2tower": {
            "source_trajectory_count": tower_report["trajectory_count"],
            "mid_skill_count": tower_report["rendered_mid_count"],
            "high_skill_count": tower_report["rendered_high_count"],
            "llm_usage": tower_usage,
        },
        "skillx": {
            "source_trajectory_count": skillx_report["source_trajectory_count"],
            "planning_skill_count": skillx_report["statistics"]["planning_skills"],
            "atomic_skill_count": skillx_report["statistics"]["atomic_skills"],
            "llm_usage": skillx_usage,
        },
        "trace2tower_relative_to_skillx": relative_usage(
            tower_usage, skillx_usage
        ),
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
