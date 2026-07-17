from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments.run import extend_no_skill_pool


def extension_module():
    return extend_no_skill_pool


def base_config() -> dict:
    common = {
        "manifests_dir": "manifests",
        "runs_dir": "runs",
        "trajectories_dir": "trajectories",
        "repeat_ids": [0],
        "num_shards": 10,
        "global_api_concurrency": 10,
        "episode_concurrency": 10,
        "provider_max_attempts": 5,
        "provider_timeout_seconds": 120,
        "retry_base_seconds": 1,
        "agent_temperature": 0,
        "agent_max_output_tokens": 512,
    }
    return {
        "agent_model": "flash",
        "common": common,
        "benchmarks": {"webshop": {"max_steps": 20}},
        "method": "no_skill",
        "pilot": {"shard_id": 0, "num_shards": 10, "max_episodes": 5},
    }


def metadata(pool_path: Path) -> dict:
    return {
        "run_id": "pilot-flash",
        "benchmark": "webshop",
        "method": "no_skill",
        "agent_model": "flash",
        "shard_id": 0,
        "num_shards": 10,
        "trajectory_path": pool_path.as_posix(),
        "trajectory_count": 5,
    }


def test_extension_contract_accepts_only_same_pilot_prefix(tmp_path: Path) -> None:
    validate = extension_module().validate_extension_contract
    pool_path = tmp_path / "pool.jsonl"
    validate(
        base_config=base_config(),
        current_common={**base_config()["common"], "embedding_dimension": 4096},
        current_benchmark_config={"max_steps": 20},
        run_metadata=metadata(pool_path),
        run_id="pilot-flash",
        benchmark=extension_module().Benchmark.WEBSHOP,
        agent_model="flash",
        shard_id=0,
        num_shards=10,
        current_count=5,
        target_count=20,
        pool_path=pool_path,
    )
    with pytest.raises(ValueError, match="immutable pilot contract"):
        validate(
            base_config=base_config(),
            current_common=base_config()["common"],
            current_benchmark_config={"max_steps": 20},
            run_metadata=metadata(pool_path),
            run_id="pilot-flash",
            benchmark=extension_module().Benchmark.WEBSHOP,
            agent_model="pro",
            shard_id=0,
            num_shards=10,
            current_count=5,
            target_count=20,
            pool_path=pool_path,
        )


def test_extension_contract_rejects_non_growing_target(tmp_path: Path) -> None:
    validate = extension_module().validate_extension_contract
    pool_path = tmp_path / "pool.jsonl"
    with pytest.raises(ValueError, match="immutable pilot contract"):
        validate(
            base_config=base_config(),
            current_common=base_config()["common"],
            current_benchmark_config={"max_steps": 20},
            run_metadata=metadata(pool_path),
            run_id="pilot-flash",
            benchmark=extension_module().Benchmark.WEBSHOP,
            agent_model="flash",
            shard_id=0,
            num_shards=10,
            current_count=5,
            target_count=5,
            pool_path=pool_path,
        )


def test_extension_contract_rejects_agent_execution_config_change(
    tmp_path: Path,
) -> None:
    validate = extension_module().validate_extension_contract
    pool_path = tmp_path / "pool.jsonl"
    changed_common = {**base_config()["common"], "agent_temperature": 0.2}
    with pytest.raises(ValueError, match="immutable pilot contract"):
        validate(
            base_config=base_config(),
            current_common=changed_common,
            current_benchmark_config={"max_steps": 20},
            run_metadata=metadata(pool_path),
            run_id="pilot-flash",
            benchmark=extension_module().Benchmark.WEBSHOP,
            agent_model="flash",
            shard_id=0,
            num_shards=10,
            current_count=5,
            target_count=20,
            pool_path=pool_path,
        )
