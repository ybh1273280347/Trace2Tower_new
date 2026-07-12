from __future__ import annotations

import argparse
import asyncio
import glob
import os
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rollout_no_skill_train import load_yaml, write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.segmentation import (
    calibrate_segmentation_penalty,
    segment_boundaries,
)
from trace2tower.methods.trace2tower.transition_encoder import TransitionEncoder
from trace2tower.methods.trace2tower.transitions import build_transitions, transition_text
from trace2tower.trajectory import TrajectoryReader


async def main(options: argparse.Namespace) -> None:
    common = load_yaml(options.config_root / "common.yaml")
    paths = tuple(Path(path) for path in sorted(glob.glob(options.trajectory_glob)))
    if not paths:
        raise FileNotFoundError(f"no trajectories match: {options.trajectory_glob}")
    trajectories = tuple(
        trajectory
        for path in paths
        for trajectory in TrajectoryReader.read_jsonl(path)
        if trajectory.benchmark is Benchmark.ALFWORLD
    )
    transition_groups = tuple(
        build_transitions(trajectory) for trajectory in trajectories
    )
    invocation = {
        "trajectory_glob": options.trajectory_glob,
        "trajectory_files": [path.as_posix() for path in paths],
        "trajectory_count": len(trajectories),
        "transition_count": sum(len(group) for group in transition_groups),
        "target_segment_length": options.target_segment_length,
        "max_segment_length": options.max_segment_length,
        "output": options.output.as_posix(),
        "dry_run": options.dry_run,
    }
    print(yaml.safe_dump({"common": common, "invocation": invocation}))
    if options.dry_run:
        return

    load_dotenv(options.env)
    model = os.environ["EMBEDDING_MODEL"]
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    encoder = TransitionEncoder(
        runtime,
        cache_path=Path(common["transition_embedding_cache"]),
        model=model,
        dimension=common["embedding_dimension"],
        batch_size=common["embedding_batch_size"],
    )
    texts = [
        transition_text(transition)
        for group in transition_groups
        for transition in group
    ]
    try:
        vectors = await encoder.embed(texts)
    finally:
        await runtime.close()

    grouped_vectors = []
    offset = 0
    for group in transition_groups:
        grouped_vectors.append(vectors[offset : offset + len(group)])
        offset += len(group)
    calibration = calibrate_segmentation_penalty(
        grouped_vectors,
        target_segment_length=options.target_segment_length,
        max_segment_length=options.max_segment_length,
    )
    lengths = [
        end - start + 1
        for group in grouped_vectors
        for start, end in segment_boundaries(
            group,
            penalty=calibration.penalty,
            max_segment_length=options.max_segment_length,
        )
    ]
    report = {
        **invocation,
        "embedding_model": model,
        "embedding_dimension": common["embedding_dimension"],
        "penalty": calibration.penalty,
        "median_segment_length": calibration.median_segment_length,
        "segment_count": calibration.segment_count,
        "segment_length_distribution": dict(sorted(Counter(lengths).items())),
    }
    write_json(options.output, report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory-glob",
        default="artifacts/trajectories/alfworld/no_skill_train/shard-*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/trace2tower/segmentation-calibration.json"),
    )
    parser.add_argument("--target-segment-length", type=int, default=3)
    parser.add_argument("--max-segment-length", type=int, default=6)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
