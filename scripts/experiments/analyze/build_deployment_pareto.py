from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BundleMetrics:
    bundle_id: str
    exposure_count: int
    performance_level: float
    paired_reward_gain: float
    guarded_step_saving: float
    mean_steps: float
    mean_chat_tokens: float | None
    pareto_front_rank: int = 0


def read_results(root: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for path in root.rglob("results.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["sample_id"]), int(record["repeat_id"]))
            if key in rows:
                raise ValueError(f"duplicate episode key in {root}: {key}")
            rows[key] = record
    return rows


def build_metrics(
    baseline_root: Path, tower_root: Path
) -> tuple[BundleMetrics, ...]:
    baseline = read_results(baseline_root)
    tower = read_results(tower_root)
    if set(baseline) != set(tower):
        raise ValueError(
            f"NoSkill/Tower key mismatch: baseline={len(baseline)} tower={len(tower)}"
        )
    grouped: dict[str, list[tuple[dict, dict]]] = {}
    for key in sorted(baseline):
        record = tower[key]
        skill_ids = tuple(record.get("skill_ids", ()))
        if not skill_ids or not str(skill_ids[0]).startswith("high_"):
            raise ValueError(f"Tower episode has no identifiable High bundle: {key}")
        grouped.setdefault(str(skill_ids[0]), []).append((baseline[key], record))

    metrics = []
    for bundle_id, pairs in sorted(grouped.items()):
        rewards = []
        gains = []
        step_savings = []
        steps = []
        chat_tokens = []
        for baseline_record, tower_record in pairs:
            baseline_score = float(baseline_record["primary_score"])
            tower_score = float(tower_record["primary_score"])
            gain = tower_score - baseline_score
            raw_step = (
                int(baseline_record["steps"]) - int(tower_record["steps"])
            ) / max(int(baseline_record["steps"]), 1)
            rewards.append(tower_score)
            gains.append(gain)
            step_savings.append(min(raw_step, 0.0) if gain < 0 else raw_step)
            steps.append(int(tower_record["steps"]))
            if (
                baseline_record.get("chat_input_tokens") is not None
                and baseline_record.get("chat_output_tokens") is not None
                and tower_record.get("chat_input_tokens") is not None
                and tower_record.get("chat_output_tokens") is not None
            ):
                chat_tokens.append(
                    int(tower_record["chat_input_tokens"])
                    + int(tower_record["chat_output_tokens"])
                )
        metrics.append(
            BundleMetrics(
                bundle_id=bundle_id,
                exposure_count=len(pairs),
                performance_level=sum(rewards) / len(rewards),
                paired_reward_gain=sum(gains) / len(gains),
                guarded_step_saving=sum(step_savings) / len(step_savings),
                mean_steps=sum(steps) / len(steps),
                mean_chat_tokens=(sum(chat_tokens) / len(chat_tokens) if chat_tokens else None),
            )
        )
    return tuple(metrics)


def dominates(left: BundleMetrics, right: BundleMetrics) -> bool:
    left_values = (
        left.performance_level,
        left.paired_reward_gain,
        left.guarded_step_saving,
    )
    right_values = (
        right.performance_level,
        right.paired_reward_gain,
        right.guarded_step_saving,
    )
    differences = tuple(a - b for a, b in zip(left_values, right_values, strict=True))
    return all(value >= 0 for value in differences) and any(
        value > 0 for value in differences
    )


def rank_fronts(metrics: tuple[BundleMetrics, ...]) -> tuple[BundleMetrics, ...]:
    remaining = list(metrics)
    ranked = []
    rank = 1
    while remaining:
        front = [
            item
            for item in remaining
            if not any(dominates(other, item) for other in remaining if other != item)
        ]
        ranked.extend(
            BundleMetrics(**{**asdict(item), "pareto_front_rank": rank})
            for item in sorted(front, key=lambda value: value.bundle_id)
        )
        remaining = [item for item in remaining if item not in front]
        rank += 1
    return tuple(ranked)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    ranked = rank_fronts(build_metrics(options.baseline, options.tower))
    eligible = [item for item in ranked if item.pareto_front_rank > 1]
    selected = min(
        eligible,
        key=lambda item: (
            -item.pareto_front_rank,
            item.paired_reward_gain,
            item.performance_level,
            -item.exposure_count,
            item.bundle_id,
        ),
        default=None,
    )
    updates = []
    if selected is not None:
        updates.append(
            {
                "skill_id": selected.bundle_id,
                "action": "downweight",
                "previous_status": "active",
                "new_status": "downweighted",
                "refinement_round": 1,
                "pareto_front_rank": selected.pareto_front_rank,
            }
        )
    payload = {
        "benchmark": "webshop",
        "split": options.split,
        "agent_model": "deepseek-v4-flash",
        "tower_snapshot_id": options.snapshot_id,
        "direct_mid_top_k": 8,
        "retrieval_policy_scope": "high_context_bundle",
        "primary_objectives": [
            "performance_level",
            "paired_reward_gain",
            "guarded_step_saving",
        ],
        "secondary_metrics": ["mean_chat_tokens"],
        "ranking_status": "complete",
        "bundles": [asdict(item) for item in ranked],
        "downweight": updates,
        "card_level_lifecycle": {
            "status": "disabled",
            "reason": "Mid cards are co-injected inside High bundles and have no independent exposure",
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
