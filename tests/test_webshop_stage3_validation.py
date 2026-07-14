from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.experiments.run.run_webshop_stage3_validation import (
    command_for,
    validation_conditions,
)


def condition(method: str, cap: int, model: str = "deepseek-v4-flash") -> dict:
    return {
        "condition_id": f"p50_{method}_cap{cap}_{model}",
        "method": method,
        "pool": "p50",
        "direct_mid_top_k": cap,
        "agent_model": model,
    }


def test_stage3_contract_requires_both_methods_and_frozen_caps() -> None:
    conditions = [
        condition(method, cap, model)
        for model in ("deepseek-v4-flash", "deepseek-v4-pro")
        for method in ("semantic_clustering", "trace2tower")
        for cap in (3, 5, 8)
    ]
    protocol = {"stages": [{"stage": 3, "conditions": conditions}]}

    assert validation_conditions(protocol) == conditions


def test_stage3_command_binds_manifest_artifact_repeats_and_cap() -> None:
    selected = condition("trace2tower", 5)
    command = command_for(
        selected,
        Path("validation.jsonl"),
        Namespace(
            episode_concurrency=20,
            api_concurrency=30,
            dry_run=True,
        ),
    )

    assert "webshop=validation.jsonl" in command
    assert "--direct-mid-top-k" in command
    assert command[command.index("--direct-mid-top-k") + 1] == "5"
    assert command.count("--repeat-id") == 3
    assert command[-1] == "--dry-run"
