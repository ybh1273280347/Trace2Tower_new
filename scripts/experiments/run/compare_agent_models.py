from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import (
    collect_benchmark,
    load_yaml,
    write_json,
    write_yaml,
)
from trace2tower.components.llm_runtime import CommonLLMRuntime
from trace2tower.core.manifests import Benchmark
from trace2tower.core.trajectory import TrajectoryReader
from trace2tower.data.trajectory_quality import summarize_trajectory_quality


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def run_model(
    model: str,
    options: argparse.Namespace,
    common: dict,
    configs: dict[Benchmark, dict],
) -> dict:
    os.environ["AGENT_MODEL"] = model
    run_id = f"{options.pilot_id}-{model}"
    model_options = argparse.Namespace(
        shard_id=options.shard_id,
        num_shards=options.num_shards,
        max_episodes=options.max_episodes,
        run_id=run_id,
        agent_model=model,
        trajectory_pool_name=f"model_pilot/{options.pilot_id}/{model}",
    )
    resolved_config = {
        "common": common,
        "benchmarks": {benchmark.value: config for benchmark, config in configs.items()},
        "method": "no_skill",
        "agent_model": model,
        "pilot": {
            "shard_id": options.shard_id,
            "num_shards": options.num_shards,
            "max_episodes": options.max_episodes,
        },
    }
    write_yaml(Path(common["runs_dir"]) / run_id / "resolved-config.yaml", resolved_config)
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    episode_semaphore = asyncio.Semaphore(common["episode_concurrency"])
    benchmarks = tuple(Benchmark) if options.benchmark == "all" else (Benchmark(options.benchmark),)
    try:
        await asyncio.gather(
            *(
                collect_benchmark(
                    benchmark,
                    model_options,
                    common,
                    configs[benchmark],
                    runtime,
                    episode_semaphore,
                )
                for benchmark in benchmarks
            )
        )
    finally:
        await runtime.close()

    benchmark_reports = {}
    all_trajectories = []
    all_results = []
    for benchmark in benchmarks:
        pool_path = (
            Path(common["trajectories_dir"])
            / benchmark
            / "model_pilot"
            / options.pilot_id
            / model
            / f"shard-{options.shard_id:02d}.jsonl"
        )
        run_dir = (
            Path(common["runs_dir"])
            / run_id
            / benchmark
            / "no_skill_train"
            / f"shard-{options.shard_id:02d}"
        )
        trajectories = TrajectoryReader.read_jsonl(pool_path)
        results = read_jsonl(run_dir / "results.jsonl")
        errors = read_jsonl(run_dir / "errors.jsonl")
        summary = summarize_trajectory_quality(trajectories, results)
        benchmark_reports[benchmark.value] = {
            **summary.to_record(),
            "attempt_error_count": len(errors),
        }
        all_trajectories.extend(trajectories)
        all_results.extend(results)
    aggregate = summarize_trajectory_quality(all_trajectories, all_results)
    return {
        "run_id": run_id,
        "agent_model": model,
        "benchmarks": benchmark_reports,
        "aggregate": aggregate.to_record(),
    }


def paired_comparison(
    model_reports: dict[str, dict], common: dict, options: argparse.Namespace
) -> dict:
    scores = {}
    for model, report in model_reports.items():
        run_id = report["run_id"]
        model_scores = {}
        benchmarks = (
            tuple(Benchmark) if options.benchmark == "all" else (Benchmark(options.benchmark),)
        )
        for benchmark in benchmarks:
            results_path = (
                Path(common["runs_dir"])
                / run_id
                / benchmark
                / "no_skill_train"
                / f"shard-{options.shard_id:02d}"
                / "results.jsonl"
            )
            model_scores[benchmark] = {
                (record["sample_id"], int(record["repeat_id"])): float(record["primary_score"])
                for record in read_jsonl(results_path)
            }
        scores[model] = model_scores

    baseline, challenger = options.models
    paired = {}
    for benchmark in benchmarks:
        baseline_scores = scores[baseline][benchmark]
        challenger_scores = scores[challenger][benchmark]
        if set(baseline_scores) != set(challenger_scores):
            raise ValueError(f"paired episode keys differ for {benchmark}")
        differences = [
            challenger_scores[key] - baseline_scores[key] for key in sorted(baseline_scores)
        ]
        paired[benchmark.value] = {
            "challenger": challenger,
            "baseline": baseline,
            "challenger_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "challenger_losses": sum(value < 0 for value in differences),
            "mean_score_delta": sum(differences) / len(differences),
        }
    return paired


async def main(options: argparse.Namespace) -> None:
    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    configs = {
        benchmark: load_yaml(options.config_root / f"{benchmark}.yaml") for benchmark in Benchmark
    }
    print(
        yaml.safe_dump(
            {
                "pilot_id": options.pilot_id,
                "benchmark": options.benchmark,
                "models": list(options.models),
                "shard_id": options.shard_id,
                "num_shards": options.num_shards,
                "max_episodes_per_benchmark": options.max_episodes,
                "agent_temperature": common["agent_temperature"],
            }
        )
    )
    reports = {}
    for model in options.models:
        reports[model] = await run_model(model, options, common, configs)
    report = {
        "pilot_id": options.pilot_id,
        "models": reports,
        "paired": paired_comparison(reports, common, options),
    }
    output = Path(common["runs_dir"]) / options.pilot_id / "model-comparison.json"
    write_json(output, report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=lambda value: tuple(item.strip() for item in value.split(",")),
        default=("deepseek-v4-flash", "deepseek-v4-pro"),
    )
    parser.add_argument("--pilot-id", default="agent-model-pilot-v1")
    parser.add_argument("--benchmark", choices=("all", *Benchmark), default="all")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    options = parser.parse_args()
    if len(options.models) != 2:
        parser.error("exactly two models are required")
    asyncio.run(main(options))
