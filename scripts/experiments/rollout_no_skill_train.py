from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from trace2tower.agent import AgentEvaluator
from trace2tower.benchmarks.alfworld import AlfworldEnvironment
from trace2tower.benchmarks.webshop import WebShopEnvironment
from trace2tower.checkpoint import EpisodeCheckpoint
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, read_manifest
from trace2tower.results import EpisodeResultWriter, MethodName
from trace2tower.runner import run_shard
from trace2tower.trajectory import TrajectoryWriter, materialize_trajectory_shard


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as output_file:
        temporary_path = Path(output_file.name)
        yaml.safe_dump(payload, output_file, sort_keys=True, allow_unicode=True)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as output_file:
        temporary_path = Path(output_file.name)
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


async def collect_benchmark(
    benchmark: Benchmark,
    options: argparse.Namespace,
    common: dict,
    benchmark_config: dict,
    runtime: CommonLLMRuntime | None,
) -> dict:
    manifest_path = Path(common["manifests_dir"]) / f"{benchmark}_train.jsonl"
    entries = read_manifest(manifest_path)
    shard_name = f"shard-{options.shard_id:02d}"
    run_dir = (
        Path(common["runs_dir"])
        / options.run_id
        / benchmark
        / "no_skill_train"
        / shard_name
    )
    checkpoint = EpisodeCheckpoint(run_dir / "results.jsonl", run_dir / "errors.jsonl")
    writer = EpisodeResultWriter(checkpoint)

    if runtime is None:
        async def unavailable_executor(entry, shard_id):
            raise RuntimeError("dry-run executor must not be called")

        summary = await run_shard(
            entries,
            method=MethodName.NO_SKILL,
            shard_id=options.shard_id,
            num_shards=options.num_shards,
            writer=writer,
            executor=unavailable_executor,
            max_concurrency=common["episode_concurrency"],
            max_episodes=options.max_episodes,
            dry_run=True,
        )
        return asdict(summary)

    episode_dir = run_dir / "episodes"
    evaluator = AgentEvaluator(
        runtime,
        TrajectoryWriter(episode_dir),
        temperature=common["agent_temperature"],
        max_output_tokens=common["agent_max_output_tokens"],
    )

    async def execute(entry, shard_id):
        if benchmark is Benchmark.ALFWORLD:
            environment = AlfworldEnvironment(
                Path(benchmark_config["dataset_root"]),
                benchmark_config["server_url"],
            )
        else:
            environment = WebShopEnvironment(
                Path(benchmark_config["dataset_root"]),
                Path(benchmark_config["source_root"]),
            )
        return await evaluator.run_episode(
            entry=entry,
            environment=environment,
            run_id=options.run_id,
            method=MethodName.NO_SKILL,
            skill_context=None,
            shard_id=shard_id,
            max_steps=benchmark_config["max_steps"],
        )

    started_at = datetime.now(timezone.utc)
    summary = await run_shard(
        entries,
        method=MethodName.NO_SKILL,
        shard_id=options.shard_id,
        num_shards=options.num_shards,
        writer=writer,
        executor=execute,
        max_concurrency=common["episode_concurrency"],
        max_episodes=options.max_episodes,
    )
    pool_path = (
        Path(common["trajectories_dir"])
        / benchmark
        / "no_skill_train"
        / f"{shard_name}.jsonl"
    )
    trajectory_count = materialize_trajectory_shard(episode_dir, pool_path)
    metadata = {
        "run_id": options.run_id,
        "benchmark": benchmark.value,
        "method": MethodName.NO_SKILL.value,
        "shard_id": options.shard_id,
        "num_shards": options.num_shards,
        "manifest_path": manifest_path.as_posix(),
        "trajectory_path": pool_path.as_posix(),
        "trajectory_count": trajectory_count,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "invocation_summary": asdict(summary),
    }
    write_json(run_dir / "run-metadata.json", metadata)
    return metadata


async def main(options: argparse.Namespace) -> None:
    config_root = options.config_root
    common = load_yaml(config_root / "common.yaml")
    configs = {
        benchmark: load_yaml(config_root / f"{benchmark}.yaml")
        for benchmark in Benchmark
    }
    resolved_config = {
        "common": common,
        "benchmarks": {
            benchmark.value: config for benchmark, config in configs.items()
        },
        "methods": {"no_skill": load_yaml(config_root / "no_skill.yaml")},
    }
    invocation = {
        "benchmark": options.benchmark,
        "method": str(options.method),
        "shard_id": options.shard_id,
        "num_shards": options.num_shards,
        "max_episodes": options.max_episodes,
        "dry_run": options.dry_run,
        "run_id": options.run_id,
    }
    print(yaml.safe_dump({"resolved_config": resolved_config, "invocation": invocation}))

    benchmarks = tuple(Benchmark) if options.benchmark == "all" else (Benchmark(options.benchmark),)
    if options.dry_run:
        for benchmark in benchmarks:
            print(benchmark, await collect_benchmark(
                benchmark, options, common, configs[benchmark], None
            ))
        return

    load_dotenv(options.env)
    write_yaml(Path(common["runs_dir"]) / options.run_id / "resolved-config.yaml", resolved_config)
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    try:
        for benchmark in benchmarks:
            print(benchmark, await collect_benchmark(
                benchmark, options, common, configs[benchmark], runtime
            ))
    finally:
        await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", *Benchmark), default="all")
    parser.add_argument("--method", choices=(MethodName.NO_SKILL,), default=MethodName.NO_SKILL)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--run-id", default="no-skill-train-v1")
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
