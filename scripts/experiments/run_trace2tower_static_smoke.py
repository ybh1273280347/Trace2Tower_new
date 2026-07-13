from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rollout_no_skill_train import load_yaml, write_json

from trace2tower.agent import AgentEvaluator
from trace2tower.benchmarks.alfworld import AlfworldEnvironment
from trace2tower.benchmarks.webshop import WebShopEnvironment
from trace2tower.checkpoint import EpisodeCheckpoint
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, ExperimentSplit, read_manifest
from trace2tower.methods.trace2tower.provider import Trace2TowerSkillProvider
from trace2tower.results import EpisodeResultWriter, MethodName
from trace2tower.runner import run_shard
from trace2tower.trajectory import TrajectoryWriter


async def main(options: argparse.Namespace) -> int:
    if options.split is ExperimentSplit.TEST:
        raise ValueError("smoke validation must not use the frozen test split")
    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    benchmark_config = load_yaml(options.config_root / f"{options.benchmark}.yaml")
    entries = read_manifest(
        Path(common["manifests_dir"])
        / f"{options.benchmark}_{options.split}.jsonl"
    )
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    provider = Trace2TowerSkillProvider.from_path(runtime, options.snapshot)
    if provider.snapshot.benchmark is not options.benchmark:
        raise ValueError("Tower snapshot benchmark does not match the smoke benchmark")
    evaluator = AgentEvaluator(
        runtime,
        TrajectoryWriter(options.output / "trajectories"),
        temperature=common["agent_temperature"],
        max_output_tokens=common["agent_max_output_tokens"],
    )
    checkpoint = EpisodeCheckpoint(
        options.output / "results.jsonl",
        options.output / "errors.jsonl",
    )

    async def execute(entry, shard_id):
        environment = (
            AlfworldEnvironment(
                Path(benchmark_config["dataset_root"]),
                benchmark_config["server_url"],
            )
            if options.benchmark is Benchmark.ALFWORLD
            else WebShopEnvironment(
                Path(benchmark_config["dataset_root"]),
                Path(benchmark_config["source_root"]),
            )
        )
        return await evaluator.run_episode(
            entry=entry,
            environment=environment,
            run_id=options.run_id,
            method=MethodName.TRACE2TOWER_STATIC,
            skill_context=None,
            shard_id=shard_id,
            max_steps=benchmark_config["max_steps"],
            skill_provider=provider.select,
        )

    try:
        summary = await run_shard(
            entries,
            method=MethodName.TRACE2TOWER_STATIC,
            shard_id=0,
            num_shards=1,
            writer=EpisodeResultWriter(checkpoint),
            executor=execute,
            max_concurrency=1,
            max_episodes=options.max_episodes,
        )
    finally:
        await runtime.close()
    report = {
        "run_id": options.run_id,
        "benchmark": options.benchmark.value,
        "split": options.split.value,
        "method": MethodName.TRACE2TOWER_STATIC.value,
        "agent_model": os.getenv("AGENT_MODEL"),
        "snapshot_id": provider.snapshot.snapshot_id,
        "summary": asdict(summary),
    }
    write_json(options.output / "report.json", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument(
        "--split",
        type=ExperimentSplit,
        choices=(ExperimentSplit.TRAIN, ExperimentSplit.DEV),
        required=True,
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
