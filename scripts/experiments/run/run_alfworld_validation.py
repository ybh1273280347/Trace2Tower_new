from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "success_cap3": {
        "method": "trace2tower_static",
        "run_id": "alfworld-valid-v1-pro-success-cap3-v2",
        "artifact": "artifacts/trace2tower/towers/alfworld-success-only-family-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse3.yaml",
    },
    "success_cap8": {
        "method": "trace2tower_static",
        "run_id": "alfworld-valid-v1-pro-success-cap8-v2",
        "artifact": "artifacts/trace2tower/towers/alfworld-success-only-family-cap8-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse8.yaml",
    },
    "mixed_cap3": {
        "method": "trace2tower_static",
        "run_id": "alfworld-valid-v1-pro-mixed-cap3-v2",
        "artifact": "artifacts/trace2tower/towers/alfworld-mixed-family-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse3.yaml",
    },
    "mixed_cap8": {
        "method": "trace2tower_static",
        "run_id": "alfworld-valid-v1-pro-mixed-cap8-v2",
        "artifact": "artifacts/trace2tower/towers/alfworld-mixed-family-cap8-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse8.yaml",
    },
}


def build_command(name: str, protocol_path: Path) -> list[str]:
    condition = CONDITIONS[name]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = protocol["validation"]
    command = [
        sys.executable,
        "-m",
        "scripts.experiments.run.run_matrix",
        "--benchmark",
        "alfworld",
        "--split",
        "dev",
        "--method",
        condition["method"],
        "--shard-id",
        "all",
        "--num-shards",
        "10",
        "--run-id",
        condition["run_id"],
        "--agent-model",
        "deepseek-v4-pro",
    ]
    for repeat_id in validation["repeat_ids"]:
        command.extend(("--repeat-id", str(repeat_id)))
    for sample_id in validation["sample_ids"]:
        command.extend(("--sample-id", sample_id))
    if artifact := condition.get("artifact"):
        command.extend(("--artifact", artifact))
    if method_config := condition.get("method_config"):
        command.extend(("--method-config", method_config))
    return command


def main(options: argparse.Namespace) -> int:
    selected = options.condition or list(CONDITIONS)
    for name in selected:
        command = build_command(name, options.protocol)
        print(f"running {name}: {CONDITIONS[name]['run_id']}", flush=True)
        if options.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        subprocess.run(command, check=True, cwd=Path(__file__).parents[3])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        choices=tuple(CONDITIONS),
        help="Only run the selected condition; may be repeated.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/alfworld_protocol_v1.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
