from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rollout_no_skill_train import collect_benchmark, load_yaml, write_json

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, read_manifest, select_shard
from trace2tower.results import MethodName
from trace2tower.trajectory import TrajectoryReader
from trace2tower.trajectory_pool import audit_training_shard

NO_SKILL_COMMON_FIELDS = (
    "manifests_dir",
    "runs_dir",
    "trajectories_dir",
    "repeat_ids",
    "num_shards",
    "global_api_concurrency",
    "episode_concurrency",
    "provider_max_attempts",
    "provider_timeout_seconds",
    "retry_base_seconds",
    "agent_temperature",
    "agent_max_output_tokens",
)


def common_execution_contract(config: dict) -> dict:
    return {field: config[field] for field in NO_SKILL_COMMON_FIELDS}


def validate_extension_contract(
    *,
    base_config: dict,
    current_common: dict,
    current_benchmark_config: dict,
    run_metadata: dict,
    run_id: str,
    benchmark: Benchmark,
    agent_model: str,
    shard_id: int,
    num_shards: int,
    current_count: int,
    target_count: int,
    pool_path: Path,
) -> None:
    pilot = base_config.get("pilot", {})
    checks = (
        base_config.get("agent_model") == agent_model,
        common_execution_contract(base_config.get("common", {}))
        == common_execution_contract(current_common),
        base_config.get("benchmarks", {}).get(benchmark.value)
        == current_benchmark_config,
        base_config.get("method") == MethodName.NO_SKILL.value,
        int(pilot.get("shard_id", -1)) == shard_id,
        int(pilot.get("num_shards", -1)) == num_shards,
        0 < int(pilot.get("max_episodes", -1)) <= current_count,
        run_metadata.get("run_id") == run_id,
        run_metadata.get("benchmark") == benchmark.value,
        run_metadata.get("method") == MethodName.NO_SKILL.value,
        run_metadata.get("agent_model") == agent_model,
        int(run_metadata.get("shard_id", -1)) == shard_id,
        int(run_metadata.get("num_shards", -1)) == num_shards,
        Path(run_metadata.get("trajectory_path", "")) == pool_path,
        int(run_metadata.get("trajectory_count", -1)) == current_count,
        target_count > current_count,
    )
    if not all(checks):
        raise ValueError("No-Skill extension does not match the immutable pilot contract")


async def main(options: argparse.Namespace) -> int:
    benchmark = Benchmark(options.benchmark)
    common = load_yaml(options.config_root / "common.yaml")
    benchmark_config = load_yaml(options.config_root / f"{benchmark}.yaml")
    base_config_path = Path(common["runs_dir"]) / options.run_id / "resolved-config.yaml"
    base_config = load_yaml(base_config_path)
    run_dir = (
        Path(common["runs_dir"])
        / options.run_id
        / benchmark
        / "no_skill_train"
        / f"shard-{options.shard_id:02d}"
    )
    run_metadata_path = run_dir / "run-metadata.json"
    run_metadata = load_yaml(run_metadata_path)
    pool_path = (
        Path(common["trajectories_dir"])
        / benchmark
        / options.trajectory_pool_name
        / f"shard-{options.shard_id:02d}.jsonl"
    )
    current_trajectories = TrajectoryReader.read_jsonl(pool_path)
    current_count = len(current_trajectories)
    agent_model = options.agent_model or base_config["agent_model"]
    validate_extension_contract(
        base_config=base_config,
        current_common=common,
        current_benchmark_config=benchmark_config,
        run_metadata=run_metadata,
        run_id=options.run_id,
        benchmark=benchmark,
        agent_model=agent_model,
        shard_id=options.shard_id,
        num_shards=options.num_shards,
        current_count=current_count,
        target_count=options.target_episodes,
        pool_path=pool_path,
    )
    entries = read_manifest(
        Path(common["manifests_dir"]) / f"{benchmark}_train.jsonl"
    )
    shard_size = len(select_shard(entries, options.shard_id, options.num_shards))
    if options.target_episodes > shard_size:
        raise ValueError(
            f"target exceeds shard size: {options.target_episodes} > {shard_size}"
        )
    before = audit_training_shard(
        entries,
        run_id=options.run_id,
        benchmark=benchmark,
        method=MethodName.NO_SKILL,
        shard_id=options.shard_id,
        num_shards=options.num_shards,
        run_dir=run_dir,
        pool_path=pool_path,
        max_episodes=current_count,
    )
    if not before.complete:
        raise ValueError("existing pilot prefix is incomplete; repair it before extension")
    extension_config = {
        "base_resolved_config_sha256": hashlib.sha256(
            base_config_path.read_bytes()
        ).hexdigest(),
        "run_id": options.run_id,
        "benchmark": benchmark.value,
        "method": MethodName.NO_SKILL.value,
        "agent_model": agent_model,
        "shard_id": options.shard_id,
        "num_shards": options.num_shards,
        "current_episode_count": current_count,
        "target_episode_count": options.target_episodes,
        "trajectory_pool_name": options.trajectory_pool_name,
        "trajectory_path": pool_path.as_posix(),
    }
    print(
        yaml.safe_dump(
            {
                "extension_config": extension_config,
                "before_audit": before.to_record(),
            },
            sort_keys=False,
        )
    )
    if options.dry_run:
        return 0

    load_dotenv(options.env)
    os.environ["AGENT_MODEL"] = agent_model
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    collect_options = argparse.Namespace(
        shard_id=options.shard_id,
        num_shards=options.num_shards,
        max_episodes=options.target_episodes,
        run_id=options.run_id,
        agent_model=agent_model,
        trajectory_pool_name=options.trajectory_pool_name,
    )
    try:
        invocation = await collect_benchmark(
            benchmark,
            collect_options,
            common,
            benchmark_config,
            runtime,
            asyncio.Semaphore(common["episode_concurrency"]),
        )
    finally:
        await runtime.close()
    after = audit_training_shard(
        entries,
        run_id=options.run_id,
        benchmark=benchmark,
        method=MethodName.NO_SKILL,
        shard_id=options.shard_id,
        num_shards=options.num_shards,
        run_dir=run_dir,
        pool_path=pool_path,
        max_episodes=options.target_episodes,
    )
    if not after.complete:
        raise ValueError("extended No-Skill pool failed its completeness audit")
    report = {
        **extension_config,
        "before_audit": before.to_record(),
        "after_audit": after.to_record(),
        "invocation": invocation,
        "trajectory_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
    }
    output = run_dir / "extensions" / f"target-{options.target_episodes:04d}.json"
    write_json(output, report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trajectory-pool-name", required=True)
    parser.add_argument("--target-episodes", type=int, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--agent-model")
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
