from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rollout_no_skill_train import load_yaml, write_json, write_yaml

from trace2tower.agent import AgentEvaluator
from trace2tower.benchmarks.alfworld import AlfworldEnvironment
from trace2tower.benchmarks.webshop import WebShopEnvironment
from trace2tower.checkpoint import EpisodeCheckpoint
from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    ManifestEntry,
    expand_manifest_repeats,
    read_manifest,
)
from trace2tower.methods.flat_skill_summary.models import FlatSkillLibrary
from trace2tower.methods.flat_skill_summary.provider import FlatSkillProvider
from trace2tower.methods.skillx.models import SkillXExecutionLibrary
from trace2tower.methods.skillx.provider import SkillXProvider
from trace2tower.methods.trace2tower.provider import Trace2TowerSkillProvider
from trace2tower.methods.trace2tower.tower import TowerSnapshot
from trace2tower.results import EpisodeResultWriter, MethodName
from trace2tower.runner import run_shard
from trace2tower.trajectory import TrajectoryWriter

EXECUTABLE_METHODS = (
    MethodName.NO_SKILL,
    MethodName.FLAT_SKILL_SUMMARY,
    MethodName.SKILLX,
    MethodName.TRACE2TOWER_STATIC,
)
METHOD_CONFIG_FILES = {
    MethodName.NO_SKILL: "no_skill.yaml",
    MethodName.FLAT_SKILL_SUMMARY: "flat_skill_summary.yaml",
    MethodName.SKILLX: "skillx.yaml",
    MethodName.TRACE2TOWER_STATIC: "trace2tower_static.yaml",
}


@dataclass(frozen=True, slots=True)
class MethodArtifact:
    benchmark: Benchmark
    method: MethodName
    path: Path
    artifact_id: str
    sha256: str

    def to_record(self) -> dict:
        return {
            "benchmark": self.benchmark.value,
            "method": self.method.value,
            "path": self.path.as_posix(),
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
        }


class DryRunWriter:
    @staticmethod
    def is_completed(entry: ManifestEntry, method: MethodName) -> bool:
        return False


def parse_shard_ids(value: str, num_shards: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(num_shards))
    shard_ids = tuple(sorted({int(item) for item in value.split(",")}))
    if not shard_ids or any(
        shard_id < 0 or shard_id >= num_shards for shard_id in shard_ids
    ):
        raise ValueError(f"shard IDs must be in [0, {num_shards})")
    return shard_ids


def parse_artifact_paths(values: list[str]) -> dict[Benchmark, Path]:
    paths = {}
    for value in values:
        benchmark_name, separator, raw_path = value.partition("=")
        if not separator or not raw_path:
            raise ValueError(f"expected BENCHMARK=PATH artifact assignment: {value}")
        benchmark = Benchmark(benchmark_name)
        if benchmark in paths:
            raise ValueError(f"duplicate artifact assignment for {benchmark}")
        paths[benchmark] = Path(raw_path)
    return paths


def load_method_artifact(
    benchmark: Benchmark,
    method: MethodName,
    path: Path,
) -> MethodArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method is MethodName.FLAT_SKILL_SUMMARY:
        library = FlatSkillLibrary.from_record(payload)
        artifact_benchmark = library.benchmark
        artifact_id = library.library_id
    elif method is MethodName.SKILLX:
        library = SkillXExecutionLibrary.from_record(payload)
        artifact_benchmark = library.benchmark
        artifact_id = library.library_id
    elif method is MethodName.TRACE2TOWER_STATIC:
        snapshot = TowerSnapshot.from_record(payload)
        snapshot.require_complete()
        artifact_benchmark = snapshot.benchmark
        artifact_id = snapshot.snapshot_id
    else:
        raise ValueError(f"method does not consume an artifact: {method}")
    if artifact_benchmark is not benchmark:
        raise ValueError("method artifact benchmark does not match its assignment")
    return MethodArtifact(
        benchmark,
        method,
        path,
        artifact_id,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def environment_type(benchmark: Benchmark):
    return AlfworldEnvironment if benchmark is Benchmark.ALFWORLD else WebShopEnvironment


def create_provider(
    runtime: CommonLLMRuntime,
    artifact: MethodArtifact,
    method_config: dict,
):
    if artifact.method is MethodName.FLAT_SKILL_SUMMARY:
        retrieval_strategy = method_config.get("retrieval_strategy", "diverse")
        return FlatSkillProvider.from_path(
            runtime,
            artifact.path,
            candidate_top_k=int(method_config.get("flat_candidate_top_k", 100)),
            similarity_threshold=float(method_config.get("flat_similarity_threshold", 0.45)),
            relative_margin=float(method_config.get("flat_relative_margin", 0.08)),
            dedup_similarity_threshold=float(
                method_config.get("flat_dedup_similarity_threshold", 0.95)
            ),
            mmr_lambda=float(method_config.get("flat_mmr_lambda", 0.75)),
            max_skills=int(method_config["flat_top_k"]),
            retrieval_strategy=retrieval_strategy,
        )
    if artifact.method is MethodName.SKILLX:
        schemas = environment_type(artifact.benchmark).tool_schemas
        allowed_tools = {schema["function"]["name"] for schema in schemas}
        return SkillXProvider.from_path(
            runtime,
            artifact.path,
            allowed_tools=allowed_tools,
            similarity_threshold=float(method_config["similarity_threshold"]),
            plan_top_k=int(method_config["plan_top_k"]),
            skills_per_step=int(method_config["skills_per_step"]),
            max_skills=int(method_config["max_skills"]),
        )
    if artifact.method is MethodName.TRACE2TOWER_STATIC:
        diverse = method_config.get("retrieval_strategy", "legacy") == "diverse"
        provider = Trace2TowerSkillProvider.from_path(
            runtime,
            artifact.path,
            high_similarity_threshold=float(
                method_config["high_similarity_threshold"]
            ),
            include_high_child_context=method_config[
                "include_high_child_context"
            ],
            direct_mid_candidate_top_k=(
                int(method_config["direct_mid_candidate_top_k"])
                if diverse
                else None
            ),
            direct_mid_similarity_threshold=float(
                method_config.get("direct_mid_similarity_threshold", 0.45)
            ),
            direct_mid_relative_margin=float(
                method_config.get("direct_mid_relative_margin", 0.08)
            ),
            direct_mid_dedup_similarity_threshold=float(
                method_config.get("direct_mid_dedup_similarity_threshold", 0.95)
            ),
            direct_mid_mmr_lambda=float(
                method_config.get("direct_mid_mmr_lambda", 0.75)
            ),
            lifecycle_report_path=(
                Path(method_config["lifecycle_report"])
                if method_config.get("lifecycle_report")
                else None
            ),
            status_tie_epsilon=float(
                method_config.get("status_tie_epsilon", 0.0)
            ),
        )
        if (
            provider.snapshot.config.high_top_k != int(method_config["high_top_k"])
            or provider.snapshot.config.direct_mid_top_k
            != int(method_config["direct_mid_top_k"])
        ):
            raise ValueError("Static retrieval config differs from the Tower snapshot")
        return provider
    raise ValueError(f"unsupported provider method: {artifact.method}")


def select_entries(
    entries: list[ManifestEntry],
    sample_ids: tuple[str, ...],
    repeat_ids: tuple[int, ...] = (),
) -> list[ManifestEntry]:
    selected = (
        [entry for entry in entries if entry.sample_id in set(sample_ids)]
        if sample_ids
        else entries
    )
    if sample_ids and {entry.sample_id for entry in selected} != set(sample_ids):
        raise ValueError("one or more selected sample IDs are absent from the manifest")
    return expand_manifest_repeats(selected, repeat_ids)


async def run_matrix_shard(
    *,
    benchmark: Benchmark,
    split: ExperimentSplit,
    entries: list[ManifestEntry],
    method: MethodName,
    shard_id: int,
    options: argparse.Namespace,
    common: dict,
    benchmark_config: dict,
    runtime: CommonLLMRuntime | None,
    provider,
    episode_semaphore: asyncio.Semaphore | None,
) -> dict:
    shard_name = f"shard-{shard_id:02d}"
    run_dir = (
        Path(common["runs_dir"])
        / options.run_id
        / benchmark
        / split
        / method
        / shard_name
    )
    writer = (
        DryRunWriter()
        if options.dry_run
        else EpisodeResultWriter(
            EpisodeCheckpoint(run_dir / "results.jsonl", run_dir / "errors.jsonl")
        )
    )

    if runtime is None:
        async def unavailable_executor(entry, current_shard_id):
            raise RuntimeError("dry-run executor must not be called")

        summary = await run_shard(
            entries,
            method=method,
            shard_id=shard_id,
            num_shards=options.num_shards,
            writer=writer,
            executor=unavailable_executor,
            max_concurrency=common["episode_concurrency"],
            max_episodes=options.max_episodes,
            dry_run=True,
        )
        return {
            "benchmark": benchmark.value,
            "split": split.value,
            "method": method.value,
            "shard_id": shard_id,
            **asdict(summary),
        }

    evaluator = AgentEvaluator(
        runtime,
        TrajectoryWriter(run_dir / "trajectories"),
        temperature=common["agent_temperature"],
        max_output_tokens=common["agent_max_output_tokens"],
        endpoint_role=ModelRole(options.agent_endpoint_role),
    )

    async def execute(entry: ManifestEntry, current_shard_id: int):
        environment = (
            AlfworldEnvironment(
                Path(benchmark_config["dataset_root"]),
                benchmark_config["server_url"],
            )
            if benchmark is Benchmark.ALFWORLD
            else WebShopEnvironment(
                Path(benchmark_config["dataset_root"]),
                Path(benchmark_config["source_root"]),
            )
        )
        return await evaluator.run_episode(
            entry=entry,
            environment=environment,
            run_id=options.run_id,
            method=method,
            skill_context=None,
            shard_id=current_shard_id,
            max_steps=benchmark_config["max_steps"],
            skill_provider=provider.select if provider else None,
        )

    summary = await run_shard(
        entries,
        method=method,
        shard_id=shard_id,
        num_shards=options.num_shards,
        writer=writer,
        executor=execute,
        max_concurrency=common["episode_concurrency"],
        episode_semaphore=episode_semaphore,
        max_episodes=options.max_episodes,
    )
    result_path = run_dir / "results.jsonl"
    error_path = run_dir / "errors.jsonl"
    official_result_count = (
        len(result_path.read_text(encoding="utf-8").splitlines())
        if result_path.exists()
        else 0
    )
    metadata = {
        "run_id": options.run_id,
        "benchmark": benchmark.value,
        "split": split.value,
        "method": method.value,
        "agent_model": os.getenv("AGENT_MODEL"),
        "agent_endpoint_role": options.agent_endpoint_role,
        "shard_id": shard_id,
        "num_shards": options.num_shards,
        "manifest_path": (
            Path(common["manifests_dir"]) / f"{benchmark}_{split}.jsonl"
        ).as_posix(),
        "result_path": result_path.as_posix(),
        "result_sha256": (
            hashlib.sha256(result_path.read_bytes()).hexdigest()
            if result_path.exists()
            else None
        ),
        "official_result_count": official_result_count,
        "error_attempt_count": (
            len(error_path.read_text(encoding="utf-8").splitlines())
            if error_path.exists()
            else 0
        ),
        "trajectory_count": len(tuple((run_dir / "trajectories").glob("*.json"))),
        "invocation_summary": asdict(summary),
    }
    write_json(run_dir / "run-metadata.json", metadata)
    return metadata


async def main(options: argparse.Namespace) -> int:
    method = MethodName(options.method)
    split = ExperimentSplit(options.split)
    common = load_yaml(options.config_root / "common.yaml")
    benchmark_configs = {
        benchmark: load_yaml(options.config_root / f"{benchmark}.yaml")
        for benchmark in Benchmark
    }
    method_config = load_yaml(
        getattr(options, "method_config", None)
        or options.config_root / METHOD_CONFIG_FILES[method]
    )
    no_skill_config = load_yaml(options.config_root / "no_skill.yaml")
    if method_config["method"] != method.value:
        raise ValueError("method config does not match the requested method")
    agent_model = options.agent_model or no_skill_config["agent_model"]
    benchmarks = (
        tuple(Benchmark)
        if options.benchmark == "all"
        else (Benchmark(options.benchmark),)
    )
    shard_ids = parse_shard_ids(options.shard_id, options.num_shards)
    artifact_paths = parse_artifact_paths(options.artifact)
    if method is MethodName.NO_SKILL:
        if artifact_paths:
            raise ValueError("No-Skill does not accept method artifacts")
        artifacts = {}
    else:
        if set(artifact_paths) != set(benchmarks):
            raise ValueError("every selected benchmark requires one method artifact")
        artifacts = {
            benchmark: load_method_artifact(
                benchmark, method, artifact_paths[benchmark]
            )
            for benchmark in benchmarks
        }
    entries_by_benchmark = {
        benchmark: select_entries(
            read_manifest(
                Path(common["manifests_dir"]) / f"{benchmark}_{split}.jsonl"
            ),
            tuple(options.sample_id),
            tuple(options.repeat_id),
        )
        for benchmark in benchmarks
    }
    resolved_config = {
        "common": common,
        "benchmarks": {
            benchmark.value: benchmark_configs[benchmark] for benchmark in benchmarks
        },
        "method": method_config,
        "agent_model": agent_model,
        "artifacts": {
            benchmark.value: artifact.to_record()
            for benchmark, artifact in artifacts.items()
        },
    }
    if options.agent_endpoint_role != ModelRole.AGENT:
        resolved_config["agent_endpoint_role"] = options.agent_endpoint_role
    if options.repeat_id:
        resolved_config["selection"] = {
            "repeat_ids": sorted(options.repeat_id),
        }
    invocation = {
        "benchmark": options.benchmark,
        "split": split.value,
        "method": method.value,
        "shard_ids": list(shard_ids),
        "num_shards": options.num_shards,
        "max_episodes": options.max_episodes,
        "sample_ids": list(options.sample_id),
        "repeat_ids": sorted(options.repeat_id),
        "dry_run": options.dry_run,
        "run_id": options.run_id,
        "agent_model": agent_model,
        "agent_endpoint_role": options.agent_endpoint_role,
    }
    print(yaml.safe_dump({"resolved_config": resolved_config, "invocation": invocation}))

    if options.dry_run:
        records = await asyncio.gather(
            *(
                run_matrix_shard(
                    benchmark=benchmark,
                    split=split,
                    entries=entries_by_benchmark[benchmark],
                    method=method,
                    shard_id=shard_id,
                    options=options,
                    common=common,
                    benchmark_config=benchmark_configs[benchmark],
                    runtime=None,
                    provider=None,
                    episode_semaphore=None,
                )
                for benchmark in benchmarks
                for shard_id in shard_ids
            )
        )
        print(json_records(records))
        return 0

    load_dotenv(options.env)
    endpoint_role = ModelRole(options.agent_endpoint_role)
    if endpoint_role is ModelRole.AGENT:
        os.environ["AGENT_MODEL"] = agent_model
    elif os.environ["RENDERER_MODEL"] != agent_model:
        raise ValueError("agent model does not match the selected endpoint role")
    resolved_path = Path(common["runs_dir"]) / options.run_id / "resolved-config.yaml"
    if resolved_path.exists() and load_yaml(resolved_path) != resolved_config:
        raise ValueError("run ID already has a different resolved configuration")
    write_yaml(resolved_path, resolved_config)
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    providers = {
        benchmark: create_provider(
            runtime,
            artifacts[benchmark],
            method_config,
        )
        for benchmark in artifacts
    }
    episode_semaphore = asyncio.Semaphore(common["episode_concurrency"])
    started_at = datetime.now(UTC)
    try:
        records = await asyncio.gather(
            *(
                run_matrix_shard(
                    benchmark=benchmark,
                    split=split,
                    entries=entries_by_benchmark[benchmark],
                    method=method,
                    shard_id=shard_id,
                    options=options,
                    common=common,
                    benchmark_config=benchmark_configs[benchmark],
                    runtime=runtime,
                    provider=providers.get(benchmark),
                    episode_semaphore=episode_semaphore,
                )
                for benchmark in benchmarks
                for shard_id in shard_ids
            )
        )
    finally:
        await runtime.close()
    metadata = {
        "run_id": options.run_id,
        "split": split.value,
        "method": method.value,
        "agent_model": agent_model,
        "agent_endpoint_role": options.agent_endpoint_role,
        "benchmarks": [benchmark.value for benchmark in benchmarks],
        "benchmark": benchmarks[0].value if len(benchmarks) == 1 else None,
        "sample_ids": list(options.sample_id),
        "artifacts": {
            benchmark.value: artifact.to_record()
            for benchmark, artifact in artifacts.items()
        },
        "snapshot_id": (
            next(iter(artifacts.values())).artifact_id
            if len(artifacts) == 1
            and method is MethodName.TRACE2TOWER_STATIC
            else None
        ),
        "shard_ids": list(shard_ids),
        "num_shards": options.num_shards,
        "repeat_ids": sorted(options.repeat_id),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "shards": records,
    }
    write_json(Path(common["runs_dir"]) / options.run_id / "matrix-metadata.json", metadata)
    print(json_records(records))
    return 0


def json_records(records: list[dict] | tuple[dict, ...]) -> str:
    return yaml.safe_dump({"shards": list(records)}, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", *Benchmark), default="all")
    parser.add_argument("--split", choices=tuple(ExperimentSplit), default="train")
    parser.add_argument("--method", choices=EXECUTABLE_METHODS, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--repeat-id", action="append", type=int, default=[])
    parser.add_argument("--shard-id", default="all")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--agent-model")
    parser.add_argument(
        "--agent-endpoint-role",
        choices=(ModelRole.AGENT.value, ModelRole.RENDERER.value),
        default=ModelRole.AGENT.value,
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
