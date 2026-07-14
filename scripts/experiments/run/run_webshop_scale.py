from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json


ROOT = Path(__file__).parents[3]
PROTOCOL_PATH = Path("configs/experiments/webshop_scale_v1.json")
ARTIFACT_ROOT = Path("artifacts/experiments/webshop-scale-v1")
TRAJECTORY_ROOT = Path("artifacts/trajectories/webshop/scale-v1")
TOWER_CONFIG = Path(
    "configs/experiments/trace2tower_webshop_repeat4_event_stratified_cap3.yaml"
)
TOWER_RUNTIME_CONFIG = Path("configs/experiments/trace2tower_static_diverse3.yaml")
POOL_ADDITION_RUN_IDS = {
    "p100": "webshop-scale-v1-flash-p100-add50",
    "p200": "webshop-scale-v1-flash-p200-add100",
}


def module_command(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", module, *arguments]


def run_command(command: list[str], label: str, *, dry_run: bool) -> dict:
    logs = ARTIFACT_ROOT / "logs"
    ledger = ARTIFACT_ROOT / "ledger"
    logs.mkdir(parents=True, exist_ok=True)
    ledger.mkdir(parents=True, exist_ok=True)
    record = {
        "label": label,
        "command": command,
    }
    if dry_run:
        print(subprocess.list2cmdline(command))
        return {**record, "dry_run": True}

    with (logs / f"{label}.out.log").open("a", encoding="utf-8") as stdout, (
        logs / f"{label}.err.log"
    ).open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    record.update(
        {
            "return_code": completed.returncode,
        }
    )
    ledger_path = ledger / f"{label}.json"
    attempts = []
    if ledger_path.exists():
        previous = json.loads(ledger_path.read_text(encoding="utf-8"))
        attempts = list(previous.get("attempts", (previous,)))
    attempts.append(record)
    write_json(
        ledger_path,
        {
            "label": label,
            "attempts": attempts,
        },
    )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return record


def protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_p200_decision() -> None:
    decision_path = Path("experiments/webshop/scale-study/p200-decision.json")
    if not decision_path.exists():
        raise ValueError("P200 requires a recorded P100 stage-gate decision")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("continue") is not True:
        raise ValueError("the recorded stage-gate decision does not authorize P200")


def collect(options: argparse.Namespace) -> None:
    if options.pool == "p200":
        require_p200_decision()
    scale = protocol(options.protocol)
    pool = scale["training"]["pools"][options.pool]
    command = module_command(
        "scripts.experiments.run.run_matrix",
        "--benchmark",
        "webshop",
        "--split",
        "train",
        "--method",
        "no_skill",
        "--shard-id",
        "all",
        "--num-shards",
        "10",
        "--run-id",
        POOL_ADDITION_RUN_IDS[options.pool],
        "--agent-model",
        scale["training"]["agent_model"],
        "--episode-concurrency",
        str(options.episode_concurrency),
        "--api-concurrency",
        str(options.api_concurrency),
    )
    for repeat_id in scale["training"]["repeat_ids"]:
        command.extend(("--repeat-id", str(repeat_id)))
    for sample_id in pool["new_sample_ids"]:
        command.extend(("--sample-id", sample_id))
    if options.max_episodes is not None:
        command.extend(("--max-episodes", str(options.max_episodes)))
    run_command(command, f"collect-{options.pool}", dry_run=options.dry_run)


def materialize(options: argparse.Namespace) -> None:
    if options.pool == "p200":
        require_p200_decision()
    output = TRAJECTORY_ROOT / f"webshop-scale-v1-{options.pool}.jsonl"
    command = module_command(
        "scripts.experiments.data.materialize_webshop_scale_pool",
        "--pool",
        options.pool,
        "--protocol",
        options.protocol.as_posix(),
        "--output",
        output.as_posix(),
    )
    run_command(command, f"materialize-{options.pool}", dry_run=options.dry_run)


def evidence_paths(pool: str) -> tuple[Path, Path]:
    root = TRAJECTORY_ROOT / "evidence"
    return (
        root / f"webshop-scale-v1-{pool}-success-only.jsonl",
        root / f"webshop-scale-v1-{pool}-mixed.jsonl",
    )


def prepare_evidence(pool: str, dry_run: bool) -> tuple[Path, Path]:
    source = TRAJECTORY_ROOT / f"webshop-scale-v1-{pool}.jsonl"
    success, mixed = evidence_paths(pool)
    for policy, output in (
        ("success-only", success),
        ("success-prioritized-contrastive-v1", mixed),
    ):
        run_command(
            module_command(
                "scripts.experiments.data.select_evidence_pool",
                "--input",
                source.as_posix(),
                "--output",
                output.as_posix(),
                "--policy",
                policy,
            ),
            f"evidence-{pool}-{policy}",
            dry_run=dry_run,
        )
    return success, mixed


def build_flat(pool: str, success: Path, dry_run: bool) -> dict:
    output = Path("artifacts/flat_skill_summary/webshop-scale-v1") / pool
    return run_command(
        module_command(
            "scripts.experiments.build.build_flat_skill_summary",
            "--benchmark",
            "webshop",
            "--trajectory-glob",
            success.as_posix(),
            "--output-dir",
            output.as_posix(),
            "--config",
            "configs/experiments/flat_skill_summary.yaml",
        ),
        f"build-{pool}-flat",
        dry_run=dry_run,
    )


def build_skillx(pool: str, success: Path, dry_run: bool) -> dict:
    root = Path("artifacts/skillx/webshop-scale-v1") / pool
    run_command(
        module_command(
            "scripts.experiments.run.run_skillx_minimal",
            "--benchmark",
            "webshop",
            "--trajectory",
            success.as_posix(),
            "--all-successful",
            "--output-dir",
            (root / "upstream-parallel4-recoverable").as_posix(),
            "--config",
            "configs/experiments/skillx_scale_parallel4_recoverable.yaml",
        ),
        f"build-{pool}-skillx-upstream",
        dry_run=dry_run,
    )
    record = run_command(
        module_command(
            "scripts.experiments.build.build_skillx_index",
            "--benchmark",
            "webshop",
            "--source-library",
            (root / "upstream-parallel4-recoverable/library.json").as_posix(),
            "--output-dir",
            (root / "execution").as_posix(),
        ),
        f"build-{pool}-skillx-index",
        dry_run=dry_run,
    )
    return record


def build_tower(pool: str, policy: str, evidence: Path, dry_run: bool) -> dict:
    root = Path("artifacts/trace2tower/scale-v1") / pool / policy
    preprocessed = root / "preprocessed.jsonl"
    graph = root / "graph"
    skills = root / "skills"
    snapshot = Path("artifacts/trace2tower/towers") / (
        f"webshop-scale-v1-{pool}-{policy}.json"
    )
    stages = (
        (
            "preprocess",
            module_command(
                "scripts.experiments.build.preprocess_trajectories",
                "--benchmark",
                "webshop",
                "--trajectory-glob",
                evidence.as_posix(),
                "--output",
                preprocessed.as_posix(),
                "--embedding-concurrency",
                "1",
            ),
        ),
        (
            "graph",
            module_command(
                "scripts.experiments.build.build_trace2tower_graph",
                "--input",
                preprocessed.as_posix(),
                "--config",
                TOWER_CONFIG.as_posix(),
                "--output-dir",
                graph.as_posix(),
            ),
        ),
        (
            "skills",
            module_command(
                "scripts.experiments.build.build_trace2tower_skills",
                "--benchmark",
                "webshop",
                "--input",
                preprocessed.as_posix(),
                "--clusters",
                (graph / "clusters.json").as_posix(),
                "--config",
                TOWER_CONFIG.as_posix(),
                "--output-dir",
                skills.as_posix(),
                "--render-all-mid",
            ),
        ),
        (
            "index",
            module_command(
                "scripts.experiments.build.build_trace2tower_index",
                "--cards",
                (skills / "rendered-cards.json").as_posix(),
                "--config",
                TOWER_CONFIG.as_posix(),
                "--output",
                (skills / "retrieval-index.json").as_posix(),
            ),
        ),
        (
            "snapshot",
            module_command(
                "scripts.experiments.build.build_tower_snapshot",
                "--benchmark",
                "webshop",
                "--input",
                preprocessed.as_posix(),
                "--clusters",
                (graph / "clusters.json").as_posix(),
                "--high-paths",
                (skills / "high-paths.json").as_posix(),
                "--cards",
                (skills / "rendered-cards.json").as_posix(),
                "--index",
                (skills / "retrieval-index.json").as_posix(),
                "--config",
                TOWER_CONFIG.as_posix(),
                "--output",
                snapshot.as_posix(),
            ),
        ),
    )
    record = {}
    for stage, command in stages:
        record = run_command(
            command,
            f"build-{pool}-tower-{policy}-{stage}",
            dry_run=dry_run,
        )
    return record


def build(options: argparse.Namespace) -> None:
    if options.pool == "p200":
        require_p200_decision()
    if options.parallel_builds > 4:
        raise ValueError("GPT-backed build concurrency cannot exceed 4")
    success, mixed = prepare_evidence(options.pool, options.dry_run)
    selected = set(
        options.method
        or ("flat", "skillx", "tower_success", "tower_mixed")
    )
    available = {
        "flat": (build_flat, (options.pool, success, options.dry_run)),
        "skillx": (build_skillx, (options.pool, success, options.dry_run)),
        "tower_success": (
            build_tower,
            (options.pool, "success", success, options.dry_run),
        ),
        "tower_mixed": (
            build_tower,
            (options.pool, "mixed", mixed, options.dry_run),
        ),
    }
    workflows = tuple(available[name] for name in available if name in selected)
    with ThreadPoolExecutor(max_workers=options.parallel_builds) as executor:
        records = [executor.submit(function, *arguments) for function, arguments in workflows]
        for future in records:
            future.result()
    write_json(
        ARTIFACT_ROOT / f"{options.pool}-build-workflows.json",
        {
            "pool": options.pool,
            "status": "complete",
            "methods": sorted(selected),
        },
    )


def evaluation_conditions(pool: str) -> dict[str, tuple[str, Path | None, Path | None]]:
    if pool == "p50":
        return {
            "noskill": ("no_skill", None, None),
            "p50-flat": (
                "flat_skill_summary",
                Path(
                    "artifacts/flat_skill_summary/"
                    "webshop-flash50-repeat4-mmr-v1/library.json"
                ),
                Path("configs/experiments/flat_skill_summary.yaml"),
            ),
            "p50-skillx": (
                "skillx",
                Path(
                    "artifacts/skillx/"
                    "webshop-success94-official-execution-v1/library.json"
                ),
                Path("configs/experiments/skillx.yaml"),
            ),
            "p50-success": (
                "trace2tower_static",
                Path(
                    "artifacts/trace2tower/towers/"
                    "webshop-flash50-repeat4-success-only-task-support10-cap3-v4.json"
                ),
                TOWER_RUNTIME_CONFIG,
            ),
            "p50-mixed": (
                "trace2tower_static",
                Path(
                    "artifacts/trace2tower/towers/"
                    "webshop-flash50-repeat4-mixed-task-support10-cap3-v4.json"
                ),
                TOWER_RUNTIME_CONFIG,
            ),
        }
    return {
        "noskill": ("no_skill", None, None),
        f"{pool}-flat": (
            "flat_skill_summary",
            Path("artifacts/flat_skill_summary/webshop-scale-v1") / pool / "library.json",
            Path("configs/experiments/flat_skill_summary.yaml"),
        ),
        f"{pool}-skillx": (
            "skillx",
            Path("artifacts/skillx/webshop-scale-v1") / pool / "execution/library.json",
            Path("configs/experiments/skillx.yaml"),
        ),
        f"{pool}-success": (
            "trace2tower_static",
            Path("artifacts/trace2tower/towers")
            / f"webshop-scale-v1-{pool}-success.json",
            TOWER_RUNTIME_CONFIG,
        ),
        f"{pool}-mixed": (
            "trace2tower_static",
            Path("artifacts/trace2tower/towers")
            / f"webshop-scale-v1-{pool}-mixed.json",
            TOWER_RUNTIME_CONFIG,
        ),
    }


def evaluate(options: argparse.Namespace) -> None:
    if options.pool == "p200":
        require_p200_decision()
    scale = protocol(options.protocol)
    evaluation = scale["evaluation"]
    commands = []
    selected = set(options.condition or ("noskill", "flat", "skillx", "success", "mixed"))
    for label, (method, artifact, method_config) in evaluation_conditions(options.pool).items():
        method_label = "noskill" if label == "noskill" else label.rsplit("-", 1)[1]
        if method_label not in selected:
            continue
        command = module_command(
            "scripts.experiments.run.run_matrix",
            "--benchmark",
            "webshop",
            "--split",
            "test",
            "--method",
            method,
            "--shard-id",
            "all",
            "--num-shards",
            "10",
            "--run-id",
            f"webshop-scale-v1-pro-{label}",
            "--agent-model",
            evaluation["agent_model"],
            "--episode-concurrency",
            str(options.episode_concurrency),
            "--api-concurrency",
            str(options.api_concurrency),
        )
        for repeat_id in evaluation["repeat_ids"]:
            command.extend(("--repeat-id", str(repeat_id)))
        for sample_id in evaluation["sample_ids"]:
            command.extend(("--sample-id", sample_id))
        if artifact is not None:
            command.extend(("--artifact", f"webshop={artifact.as_posix()}"))
        if method_config is not None:
            command.extend(("--method-config", method_config.as_posix()))
        commands.append((label, command))

    with ThreadPoolExecutor(max_workers=options.parallel_conditions) as executor:
        futures = [
            executor.submit(
                run_command,
                command,
                f"evaluate-{label}",
                dry_run=options.dry_run,
            )
            for label, command in commands
        ]
        for future in futures:
            future.result()


def main(options: argparse.Namespace) -> int:
    if options.command == "collect":
        collect(options)
    elif options.command == "materialize":
        materialize(options)
    elif options.command == "build":
        build(options)
    else:
        evaluate(options)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--pool", choices=("p100", "p200"), required=True)
    collect_parser.add_argument("--episode-concurrency", type=int, default=20)
    collect_parser.add_argument("--api-concurrency", type=int, default=20)
    collect_parser.add_argument("--max-episodes", type=int)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument(
        "--pool", choices=("p50", "p100", "p200"), required=True
    )

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--pool", choices=("p50", "p100", "p200"), required=True)
    build_parser.add_argument("--parallel-builds", type=int, default=4)
    build_parser.add_argument(
        "--method",
        action="append",
        choices=("flat", "skillx", "tower_success", "tower_mixed"),
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--pool", choices=("p50", "p100", "p200"), required=True
    )
    evaluate_parser.add_argument("--parallel-conditions", type=int, default=4)
    evaluate_parser.add_argument("--episode-concurrency", type=int, default=10)
    evaluate_parser.add_argument("--api-concurrency", type=int, default=10)
    evaluate_parser.add_argument(
        "--condition",
        action="append",
        choices=("noskill", "flat", "skillx", "success", "mixed"),
    )
    raise SystemExit(main(parser.parse_args()))
