from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "noskill": {
        "method": "no_skill",
    },
    "success": {
        "method": "trace2tower_static",
        "artifact": "artifacts/trace2tower/towers/alfworld-success-only-family-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse3.yaml",
    },
    "mixed": {
        "method": "trace2tower_static",
        "artifact": "artifacts/trace2tower/towers/alfworld-mixed-family-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse3.yaml",
    },
    "success_mid": {
        "method": "trace2tower_static",
        "artifact": "artifacts/trace2tower/towers/alfworld-success-only-family-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse3_mid_only.yaml",
    },
    "mixed_mid": {
        "method": "trace2tower_static",
        "artifact": "artifacts/trace2tower/towers/alfworld-mixed-family-v2.json",
        "method_config": "configs/experiments/trace2tower_alfworld_family_diverse3_mid_only.yaml",
    },
    "flat": {
        "method": "flat_skill_summary",
        "artifact": "artifacts/flat/alfworld-success-family-v1/combined/library.json",
        "method_config": "configs/experiments/flat_skill_summary_alfworld_family3.yaml",
    },
    "skillx": {
        "method": "skillx",
        "artifact": "artifacts/skillx/alfworld-success-family-v1/combined/library.json",
        "method_config": "configs/experiments/skillx_alfworld_family.yaml",
    },
}

MODELS = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


def build_command(
    name: str,
    model: str,
    protocol_path: Path,
    *,
    max_episodes: int | None,
    server_url: str | None,
    episode_concurrency: int | None,
) -> list[str]:
    condition = CONDITIONS[name]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    final_test = protocol["test"]
    command = [
        sys.executable,
        "-m",
        "scripts.experiments.run.run_matrix",
        "--benchmark",
        "alfworld",
        "--split",
        "test",
        "--method",
        condition["method"],
        "--shard-id",
        "all",
        "--num-shards",
        "10",
        "--run-id",
        f"alfworld-final-v1-{model}-{name}",
        "--agent-model",
        MODELS[model],
    ]
    for repeat_id in final_test["repeat_ids"]:
        command.extend(("--repeat-id", str(repeat_id)))
    for sample_id in final_test["sample_ids"]:
        command.extend(("--sample-id", sample_id))
    if max_episodes is not None:
        command.extend(("--max-episodes", str(max_episodes)))
    if server_url is not None:
        command.extend(("--alfworld-server-url", server_url))
    if episode_concurrency is not None:
        command.extend(("--episode-concurrency", str(episode_concurrency)))
    if artifact := condition.get("artifact"):
        command.extend(("--artifact", f"alfworld={artifact}"))
    if method_config := condition.get("method_config"):
        command.extend(("--method-config", method_config))
    return command


def main(options: argparse.Namespace) -> int:
    selected = options.condition or list(CONDITIONS)
    for name in selected:
        command = build_command(
            name,
            options.model,
            options.protocol,
            max_episodes=options.max_episodes,
            server_url=options.server_url,
            episode_concurrency=options.episode_concurrency,
        )
        print(f"running {options.model} {name}", flush=True)
        if options.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        subprocess.run(command, check=True, cwd=Path(__file__).parents[3])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
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
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--server-url")
    parser.add_argument("--episode-concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
