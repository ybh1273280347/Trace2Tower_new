from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml
from dotenv import load_dotenv

from trace2tower.benchmarks.alfworld import AlfworldEnvironment
from trace2tower.benchmarks.webshop import WebShopEnvironment
from trace2tower.components.agent import AgentEvaluator
from trace2tower.components.llm_runtime import CommonLLMRuntime
from trace2tower.core.manifests import Benchmark, ManifestEntry, read_manifest
from trace2tower.core.results import MethodName
from trace2tower.core.trajectory import TrajectoryWriter
from trace2tower.experiments.checkpoint import EpisodeCheckpoint
from trace2tower.experiments.result_writer import EpisodeResultWriter
from trace2tower.experiments.runner import run_shard


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


async def run_benchmark(
    benchmark: Benchmark,
    entry: ManifestEntry,
    config: dict,
    evaluator: AgentEvaluator,
    writer: EpisodeResultWriter,
    run_id: str,
) -> None:
    async def execute(current_entry: ManifestEntry, shard_id: int):
        if benchmark is Benchmark.ALFWORLD:
            environment = AlfworldEnvironment(Path(config["dataset_root"]), config["server_url"])
        else:
            environment = WebShopEnvironment(
                Path(config["dataset_root"]), Path(config["source_root"])
            )
        return await evaluator.run_episode(
            entry=current_entry,
            environment=environment,
            run_id=run_id,
            method=MethodName.NO_SKILL,
            skill_context=None,
            shard_id=shard_id,
            max_steps=config["max_steps"],
        )

    summary = await run_shard(
        [entry],
        method=MethodName.NO_SKILL,
        shard_id=0,
        num_shards=1,
        writer=writer,
        executor=execute,
        max_concurrency=1,
    )
    print(f"{benchmark}: {summary}")


async def main(options: argparse.Namespace) -> None:
    load_dotenv(options.env)
    common = load_yaml(Path("configs/experiments/common.yaml"))
    output = options.output
    checkpoint = EpisodeCheckpoint(output / "results.jsonl", output / "errors.jsonl")
    writer = EpisodeResultWriter(checkpoint)
    evaluator = AgentEvaluator(
        CommonLLMRuntime(
            max_concurrency=common["global_api_concurrency"],
            max_attempts=common["provider_max_attempts"],
            timeout_seconds=common["provider_timeout_seconds"],
            retry_base_seconds=common["retry_base_seconds"],
        ),
        TrajectoryWriter(output / "trajectories"),
        temperature=common["agent_temperature"],
        max_output_tokens=common["agent_max_output_tokens"],
    )
    runtime = evaluator.runtime
    try:
        for benchmark in (Benchmark.ALFWORLD, Benchmark.WEBSHOP):
            config = load_yaml(Path(f"configs/experiments/{benchmark}.yaml"))
            entry = read_manifest(Path(common["manifests_dir"]) / f"{benchmark}_test.jsonl")[0]
            await run_benchmark(benchmark, entry, config, evaluator, writer, options.run_id)
    finally:
        await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/smoke/benchmark-adapters-v1")
    )
    parser.add_argument("--run-id", default="benchmark-adapters-v1")
    asyncio.run(main(parser.parse_args()))
