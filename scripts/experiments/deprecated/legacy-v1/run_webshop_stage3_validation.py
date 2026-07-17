from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json

ARTIFACTS = {
    "semantic_clustering": Path(
        "artifacts/trace2tower/event-tower-v2/p50/semantic-only/tower.json"
    ),
    "trace2tower": Path(
        "artifacts/trace2tower/event-tower-v2/p50/full/tower.json"
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_conditions(protocol: dict) -> list[dict]:
    stage = next(item for item in protocol["stages"] if item["stage"] == 3)
    conditions = stage["conditions"]
    if {
        condition["method"] for condition in conditions
    } != set(ARTIFACTS) or any(
        condition["direct_mid_top_k"] not in (3, 5, 8)
        for condition in conditions
    ):
        raise ValueError("stage 3 protocol contains an unexpected condition")
    return conditions


def command_for(
    condition: dict,
    manifest: Path,
    options: argparse.Namespace,
) -> list[str]:
    method = condition["method"]
    command = [
        sys.executable,
        "-m",
        "scripts.experiments.run.run_matrix",
        "--benchmark",
        "webshop",
        "--split",
        "dev",
        "--method",
        method,
        "--manifest",
        f"webshop={manifest.as_posix()}",
        "--artifact",
        f"webshop={ARTIFACTS[method].as_posix()}",
        "--shard-id",
        "all",
        "--num-shards",
        "10",
        "--run-id",
        f"webshop-event-tower-v2-validation-{condition['condition_id']}",
        "--agent-model",
        condition["agent_model"],
        "--repeat-id",
        "0",
        "--repeat-id",
        "1",
        "--repeat-id",
        "2",
        "--direct-mid-top-k",
        str(condition["direct_mid_top_k"]),
        "--episode-concurrency",
        str(options.episode_concurrency),
        "--api-concurrency",
        str(options.api_concurrency),
    ]
    if options.dry_run:
        command.append("--dry-run")
    return command


def main(options: argparse.Namespace) -> int:
    protocol = json.loads(options.protocol.read_text(encoding="utf-8"))
    conditions = validation_conditions(protocol)
    if options.condition_id:
        selected_ids = set(options.condition_id)
        conditions = [
            condition
            for condition in conditions
            if condition["condition_id"] in selected_ids
        ]
        if {condition["condition_id"] for condition in conditions} != selected_ids:
            raise ValueError("one or more requested condition IDs are absent")

    if options.dry_run:
        for condition in conditions:
            subprocess.run(
                command_for(condition, options.manifest, options),
                check=True,
            )
        return 0

    options.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = options.output_dir / "ledger.json"
    ledger = {
        "protocol": options.protocol.as_posix(),
        "protocol_sha256": sha256_file(options.protocol),
        "manifest": options.manifest.as_posix(),
        "manifest_sha256": sha256_file(options.manifest),
        "conditions": [],
    }
    if ledger_path.exists():
        existing = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (
            existing["protocol_sha256"] != ledger["protocol_sha256"]
            or existing["manifest_sha256"] != ledger["manifest_sha256"]
        ):
            raise ValueError("stage 3 ledger belongs to a different protocol")
        ledger = existing

    completed_ids = {
        item["condition_id"]
        for item in ledger["conditions"]
        if item["return_code"] == 0
    }
    for condition in conditions:
        if condition["condition_id"] in completed_ids:
            continue
        command = command_for(condition, options.manifest, options)
        log_path = options.output_dir / f"{condition['condition_id']}.log"
        started_at = datetime.now(UTC)
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        record = {
            **condition,
            "artifact": ARTIFACTS[condition["method"]].as_posix(),
            "artifact_sha256": sha256_file(ARTIFACTS[condition["method"]]),
            "command": command,
            "run_id": (
                "webshop-event-tower-v2-validation-"
                f"{condition['condition_id']}"
            ),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "return_code": result.returncode,
            "log": log_path.as_posix(),
            "log_sha256": sha256_file(log_path),
        }
        ledger["conditions"] = [
            item
            for item in ledger["conditions"]
            if item["condition_id"] != condition["condition_id"]
        ] + [record]
        ledger["conditions"].sort(key=lambda item: item["condition_id"])
        write_json(ledger_path, ledger)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/webshop_event_tower_v2.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "experiments/webshop/event-tower-v2/manifests/validation.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/experiments/webshop-event-tower-v2/stage-3-validation"
        ),
    )
    parser.add_argument("--condition-id", action="append", default=[])
    parser.add_argument("--episode-concurrency", type=int, default=50)
    parser.add_argument("--api-concurrency", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
