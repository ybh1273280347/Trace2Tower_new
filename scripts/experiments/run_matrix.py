from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rollout_no_skill_train import collect_benchmark, load_yaml, write_json, write_yaml
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.results import MethodName


def parse_shard_ids(value: str, num_shards: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(num_shards))
    shard_ids = tuple(sorted({int(item) for item in value.split(",")}))
    if not shard_ids or any(shard_id < 0 or shard_id >= num_shards for shard_id in shard_ids):
        raise ValueError(f"shard IDs must be in [0, {num_shards})")
    return shard_ids


async def main(options: argparse.Namespace) -> None:
    common = load_yaml(options.config_root / "common.yaml")
    configs = {
        benchmark: load_yaml(options.config_root / f"{benchmark}.yaml")
        for benchmark in Benchmark
    }
    resolved_config = {
        "common": common,
        "benchmarks": {
            benchmark.value: config for benchmark, config in configs.items()
        },
        "methods": {
            "no_skill": load_yaml(options.config_root / "no_skill.yaml")
        },
    }
    agent_model = options.agent_model or resolved_config["methods"]["no_skill"]["agent_model"]
    benchmarks = (
        tuple(Benchmark)
        if options.benchmark == "all"
        else (Benchmark(options.benchmark),)
    )
    shard_ids = parse_shard_ids(options.shard_id, options.num_shards)
    invocation = {
        "benchmark": options.benchmark,
        "method": str(options.method),
        "shard_ids": list(shard_ids),
        "num_shards": options.num_shards,
        "max_episodes": options.max_episodes,
        "dry_run": options.dry_run,
        "run_id": options.run_id,
        "agent_model": agent_model,
    }
    print(yaml.safe_dump({"resolved_config": resolved_config, "invocation": invocation}))

    if options.dry_run:
        tasks = []
        for benchmark in benchmarks:
            for shard_id in shard_ids:
                shard_options = argparse.Namespace(**vars(options))
                shard_options.shard_id = shard_id
                tasks.append(
                    collect_benchmark(
                        benchmark,
                        shard_options,
                        common,
                        configs[benchmark],
                        None,
                    )
                )
        print(json_records(await asyncio.gather(*tasks)))
        return

    load_dotenv(options.env)
    os.environ["AGENT_MODEL"] = agent_model
    write_yaml(
        Path(common["runs_dir"]) / options.run_id / "resolved-config.yaml",
        resolved_config,
    )
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    episode_semaphore = asyncio.Semaphore(common["episode_concurrency"])
    started_at = datetime.now(timezone.utc)
    try:
        tasks = []
        for benchmark in benchmarks:
            for shard_id in shard_ids:
                shard_options = argparse.Namespace(**vars(options))
                shard_options.shard_id = shard_id
                tasks.append(
                    collect_benchmark(
                        benchmark,
                        shard_options,
                        common,
                        configs[benchmark],
                        runtime,
                        episode_semaphore,
                    )
                )
        records = await asyncio.gather(*tasks)
    finally:
        await runtime.close()

    metadata = {
        "run_id": options.run_id,
        "method": MethodName.NO_SKILL.value,
        "benchmarks": [benchmark.value for benchmark in benchmarks],
        "shard_ids": list(shard_ids),
        "num_shards": options.num_shards,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "shards": records,
    }
    write_json(Path(common["runs_dir"]) / options.run_id / "matrix-metadata.json", metadata)
    print(json_records(records))


def json_records(records: list[dict] | tuple[dict, ...]) -> str:
    return yaml.safe_dump({"shards": list(records)}, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", *Benchmark), default="all")
    parser.add_argument("--method", choices=(MethodName.NO_SKILL,), default=MethodName.NO_SKILL)
    parser.add_argument("--shard-id", default="all")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--run-id", default="no-skill-train-v1")
    parser.add_argument("--agent-model")
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
