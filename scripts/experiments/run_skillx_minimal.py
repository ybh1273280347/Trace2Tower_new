from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from check_skillx_upstream import inspect_skillx
from dotenv import load_dotenv
from rollout_no_skill_train import load_yaml, write_json

from trace2tower.benchmarks.alfworld import AlfworldEnvironment
from trace2tower.benchmarks.webshop import WebShopEnvironment
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.methods.skillx.embedding_adapter import SkillXEmbeddingAdapter
from trace2tower.methods.skillx.llm_adapter import SkillXLLMAdapter
from trace2tower.methods.skillx.trajectory_adapter import (
    adapt_tool_schemas,
    adapt_trajectory,
)
from trace2tower.results import MethodName
from trace2tower.trajectory import EpisodeTrajectory, TrajectoryReader


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_trajectory(
    path: Path,
    benchmark: Benchmark,
    threshold: float,
    sample_id: str | None,
) -> EpisodeTrajectory:
    candidates = tuple(
        trajectory
        for trajectory in TrajectoryReader.read_jsonl(path)
        if trajectory.benchmark is benchmark
        and trajectory.split is ExperimentSplit.TRAIN
        and trajectory.method is MethodName.NO_SKILL
        and trajectory.primary_score >= threshold
        and (sample_id is None or trajectory.sample_id == sample_id)
    )
    if not candidates:
        requested = f" for sample {sample_id}" if sample_id else ""
        raise ValueError(f"no fully successful shared training trajectory{requested}")
    return min(candidates, key=lambda trajectory: trajectory.trajectory_id)


def tool_schemas_for(benchmark: Benchmark) -> dict[str, dict]:
    schemas = {
        Benchmark.ALFWORLD: AlfworldEnvironment.tool_schemas,
        Benchmark.WEBSHOP: WebShopEnvironment.tool_schemas,
    }
    return adapt_tool_schemas(schemas[benchmark])


async def main(options: argparse.Namespace) -> int:
    report_path = options.output_dir / "report.json"
    if report_path.exists() and not options.force:
        raise FileExistsError(
            f"{report_path} already exists; pass --force to spend on a fresh run"
        )
    config = load_yaml(options.config)
    if config["method"] != "skillx":
        raise ValueError("SkillX minimal run requires the SkillX config")
    upstream = inspect_skillx(options.skillx_root)
    trajectory = select_trajectory(
        options.trajectory,
        options.benchmark,
        float(config["filter_threshold"]),
        options.sample_id,
    )
    adapted = adapt_trajectory(trajectory)
    config_sha256 = hashlib.sha256(options.config.read_bytes()).hexdigest()

    skillx_parent = str(options.skillx_root.parent.resolve())
    if skillx_parent not in sys.path:
        sys.path.insert(0, skillx_parent)
    from SkillX.pipeline import IterativeSkillPipeline

    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    llm = SkillXLLMAdapter(
        runtime,
        max_output_tokens=int(config["llm_max_output_tokens"]),
        temperature=float(config["llm_temperature"]),
        max_validation_attempts=int(config["llm_validation_attempts"]),
        retry_delay_seconds=float(config["llm_retry_delay_seconds"]),
    )
    embedding = SkillXEmbeddingAdapter(runtime)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = IterativeSkillPipeline(
        llm=llm,
        benchmark=options.benchmark.value,
        skill_type=config["skill_type"],
        plan_strategy=config["plan_strategy"],
        atomic_mode=config["atomic_mode"],
        expansion_strategy=None,
        tool_schemas=tool_schemas_for(options.benchmark),
        output_dir=str(options.output_dir / "upstream"),
        verbose=True,
    )
    pipeline.clusterer.embedding_service = embedding
    started_at = datetime.now(UTC)
    try:
        results = await pipeline.run(
            [adapted],
            num_epochs=int(config["num_epochs"]),
            filter_threshold=float(config["filter_threshold"]),
            batch_size=int(config["batch_size"]),
            max_concurrent=int(config["max_concurrent"]),
            enable_clustering=bool(config["enable_clustering"]),
            filter_timing=config["filter_timing"],
        )
    finally:
        await runtime.close()
    finished_at = datetime.now(UTC)

    library = results["skill_library"].to_dict()
    write_json(options.output_dir / "library.json", library)
    epoch_statistics = [epoch["statistics"] for epoch in results["epochs"]]
    report = {
        "benchmark": options.benchmark.value,
        "source_trajectory_id": trajectory.trajectory_id,
        "source_trajectory_sha256": canonical_sha256(trajectory.to_record()),
        "source_step_count": len(trajectory.steps),
        "source_score": trajectory.primary_score,
        "skillx_commit": upstream["commit"],
        "skillx_protected_file_count": upstream["protected_file_count"],
        "config_sha256": config_sha256,
        "config": config,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "statistics": results["statistics"],
        "epoch_statistics": epoch_statistics,
        "llm_usage": asdict(llm.usage),
        "embedding_input_tokens": embedding.input_tokens,
        "library_sha256": canonical_sha256(library),
    }
    write_json(report_path, report)
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/skillx.yaml")
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--skillx-root", type=Path, default=Path("third_party/SkillX"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--force", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
