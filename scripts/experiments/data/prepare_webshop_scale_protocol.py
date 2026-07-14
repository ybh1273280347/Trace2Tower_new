from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json


POOL_SIZES = (50, 100, 200)
TRAIN_REPEAT_IDS = (0, 1, 2, 3)
EVALUATION_REPEAT_IDS = (0, 1, 2)


def read_manifest(path: Path, expected_split: str) -> list[str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(
        row["benchmark"] != "webshop" or row["split"] != expected_split
        for row in rows
    ):
        raise ValueError(f"{path} is not a WebShop {expected_split} manifest")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return sample_ids


def main(options: argparse.Namespace) -> int:
    p50_audit = json.loads(options.p50_audit.read_text(encoding="utf-8"))
    p50_ids = list(p50_audit["sample_ids"])
    if (
        p50_audit["benchmark"] != "webshop"
        or p50_audit["agent_model"] != "deepseek-v4-flash"
        or p50_audit["sample_count"] != 50
        or p50_audit["trajectory_count"] != 200
        or tuple(p50_audit["repeat_ids"]) != TRAIN_REPEAT_IDS
        or len(p50_ids) != len(set(p50_ids))
    ):
        raise ValueError("the existing P50 audit does not satisfy the scale protocol")

    train_ids = read_manifest(options.train_manifest, "train")
    test_ids = read_manifest(options.test_manifest, "test")
    if not set(p50_ids) <= set(train_ids):
        raise ValueError("P50 contains samples outside the WebShop train manifest")

    remaining_train_ids = sorted(set(train_ids) - set(p50_ids))
    random.Random(options.train_seed).shuffle(remaining_train_ids)
    additions = remaining_train_ids[:150]
    pools = {
        "p50": p50_ids,
        "p100": p50_ids + additions[:50],
        "p200": p50_ids + additions,
    }
    if not set(pools["p50"]) < set(pools["p100"]) < set(pools["p200"]):
        raise ValueError("generated pools are not strictly nested")

    evaluation_ids = sorted(test_ids)
    random.Random(options.evaluation_seed).shuffle(evaluation_ids)
    evaluation_ids = evaluation_ids[:100]
    payload = {
        "protocol_id": "webshop-scale-v1",
        "benchmark": "webshop",
        "selection_frozen_before_rollout": True,
        "training": {
            "agent_model": "deepseek-v4-flash",
            "repeat_ids": list(TRAIN_REPEAT_IDS),
            "train_selection_seed": options.train_seed,
            "p50_source_audit": options.p50_audit.as_posix(),
            "p50_source_audit_sha256": hashlib.sha256(
                options.p50_audit.read_bytes()
            ).hexdigest(),
            "pools": {
                name: {
                    "task_count": size,
                    "expected_episode_count": size * len(TRAIN_REPEAT_IDS),
                    "sample_ids": pools[name],
                    "new_sample_ids": (
                        []
                        if name == "p50"
                        else pools[name][POOL_SIZES[POOL_SIZES.index(size) - 1] :]
                    ),
                }
                for name, size in zip(pools, POOL_SIZES, strict=True)
            },
        },
        "construction": {
            "methods": [
                "flat_success_only",
                "skillx_success_only",
                "tower_success_only",
                "tower_mixed",
            ],
            "renderer_model_env": "RENDERER_MODEL",
            "required_statistics": [
                "builder_chat_input_tokens",
                "builder_chat_output_tokens",
                "embedding_input_tokens",
                "final_skill_count",
            ],
        },
        "evaluation": {
            "agent_model": "deepseek-v4-pro",
            "selection_seed": options.evaluation_seed,
            "sample_count": 100,
            "repeat_ids": list(EVALUATION_REPEAT_IDS),
            "expected_episode_count_per_condition": 300,
            "sample_ids": evaluation_ids,
            "conditions": [
                "noskill",
                *[
                    f"{pool}-{method}"
                    for pool in pools
                    for method in ("flat", "skillx", "success", "mixed")
                ],
            ],
        },
        "stage_gate": {
            "evaluate_p100_before_p200": True,
            "continue_if_any": [
                "tower_vs_flat_gap_improves",
                "mixed_negative_effect_shrinks",
                "graph_support_or_cluster_stability_improves",
                "construction_cost_curve_beats_flat_or_skillx",
            ],
            "p200_requires_recorded_decision": True,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(options.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p50-audit",
        type=Path,
        default=Path(
            "artifacts/trajectories/webshop/multirepeat/"
            "webshop-flash50-repeat4-pool-v1.audit.json"
        ),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("artifacts/manifests/webshop_train.jsonl"),
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("artifacts/manifests/webshop_test.jsonl"),
    )
    parser.add_argument("--train-seed", type=int, default=20260714)
    parser.add_argument("--evaluation-seed", type=int, default=20260715)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/experiments/webshop_scale_v1.json"),
    )
    raise SystemExit(main(parser.parse_args()))
