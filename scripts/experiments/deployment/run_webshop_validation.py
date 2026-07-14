from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "noskill": {
        "method": "no_skill",
        "run_id": "webshop-deploy-valid-v1-pro-noskill",
    },
    "success_v0": {
        "method": "trace2tower_static",
        "run_id": "webshop-deploy-valid-v1-pro-success-v0",
        "artifact": "artifacts/trace2tower/towers/webshop-flash50-repeat4-success-only-task-support10-cap3-v4.json",
        "method_config": "configs/experiments/trace2tower_static_diverse3.yaml",
    },
    "success_v1": {
        "method": "trace2tower_static",
        "run_id": "webshop-deploy-valid-v1-pro-success-v1",
        "artifact": "artifacts/trace2tower/towers/webshop-flash50-repeat4-success-only-task-support10-cap3-v4.json",
        "method_config": "artifacts/deployment/configs/webshop-success-cap3-v1.yaml",
    },
    "mixed_v0": {
        "method": "trace2tower_static",
        "run_id": "webshop-deploy-valid-v1-pro-mixed-v0",
        "artifact": "artifacts/trace2tower/towers/webshop-flash50-repeat4-mixed-task-support10-cap3-v4.json",
        "method_config": "configs/experiments/trace2tower_static_diverse3.yaml",
    },
    "mixed_v1": {
        "method": "trace2tower_static",
        "run_id": "webshop-deploy-valid-v1-pro-mixed-v1",
        "artifact": "artifacts/trace2tower/towers/webshop-flash50-repeat4-mixed-task-support10-cap3-v4.json",
        "method_config": "artifacts/deployment/configs/webshop-mixed-cap3-v1.yaml",
    },
}


def build_command(name: str) -> list[str]:
    condition = CONDITIONS[name]
    command = [
        sys.executable,
        "-m",
        "scripts.experiments.run.run_matrix",
        "--benchmark",
        "webshop",
        "--split",
        "test",
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
    for repeat_id in range(3):
        command.extend(("--repeat-id", str(repeat_id)))
    for task_id in range(50, 150):
        command.extend(("--sample-id", f"webshop:{task_id}"))
    if artifact := condition.get("artifact"):
        command.extend(("--artifact", artifact))
    if method_config := condition.get("method_config"):
        command.extend(("--method-config", method_config))
    return command


def main(options: argparse.Namespace) -> int:
    selected = options.condition or list(CONDITIONS)
    for name in selected:
        command = build_command(name)
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
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
