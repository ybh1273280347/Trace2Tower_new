from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.alfworld_events import (
    alfworld_segment_signature,
    segment_alfworld_trajectory,
)
from trace2tower.methods.trace2tower.transition_encoder import TransitionEncoder
from trace2tower.methods.trace2tower.transitions import build_transitions
from trace2tower.methods.trace2tower.webshop_events import (
    segment_webshop_trajectory,
    webshop_segment_signature,
)
from trace2tower.trajectory import TrajectoryReader


async def main(options: argparse.Namespace) -> None:
    if options.embedding_concurrency is not None and options.embedding_concurrency <= 0:
        raise ValueError("embedding concurrency must be positive")
    common = load_yaml(options.config_root / "common.yaml")
    benchmark = Benchmark(options.benchmark)
    paths = tuple(Path(path) for path in sorted(glob.glob(options.trajectory_glob)))
    if not paths:
        raise FileNotFoundError(f"no trajectories match: {options.trajectory_glob}")
    trajectories = tuple(
        trajectory
        for path in paths
        for trajectory in TrajectoryReader.read_jsonl(path)
        if trajectory.benchmark is benchmark
    )
    transition_groups = tuple(build_transitions(trajectory) for trajectory in trajectories)
    segment_groups = tuple(
        (
            segment_alfworld_trajectory(trajectory, transitions)
            if benchmark is Benchmark.ALFWORLD
            else segment_webshop_trajectory(trajectory, transitions)
        )
        for trajectory, transitions in zip(trajectories, transition_groups)
    )
    invocation = {
        "benchmark": benchmark.value,
        "trajectory_glob": options.trajectory_glob,
        "trajectory_files": [path.as_posix() for path in paths],
        "trajectory_count": len(trajectories),
        "transition_count": sum(len(group) for group in transition_groups),
        "output": options.output.as_posix(),
        "embedding_concurrency": options.embedding_concurrency,
        "dry_run": options.dry_run,
    }
    print(yaml.safe_dump({"common": common, "invocation": invocation}))
    if options.dry_run:
        return

    load_dotenv(options.env)
    runtime = CommonLLMRuntime(
        max_concurrency=(
            options.embedding_concurrency
            if options.embedding_concurrency is not None
            else common["global_api_concurrency"]
        ),
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    encoder = TransitionEncoder(
        runtime,
        cache_path=Path(common["transition_embedding_cache"]),
        model=os.environ["EMBEDDING_MODEL"],
        dimension=common["embedding_dimension"],
        batch_size=common["embedding_batch_size"],
    )
    texts = (
        [alfworld_segment_signature(segment) for group in segment_groups for segment in group]
        if benchmark is Benchmark.ALFWORLD
        else [webshop_segment_signature(segment) for group in segment_groups for segment in group]
    )
    try:
        vectors = await encoder.embed(texts)
    finally:
        await runtime.close()

    records = []
    primitive_counts = Counter()
    event_counts = Counter()
    segment_lengths = Counter()
    offset = 0
    for trajectory_index, (trajectory, transitions) in enumerate(
        zip(trajectories, transition_groups)
    ):
        raw_segments = segment_groups[trajectory_index]
        segment_vectors = vectors[offset : offset + len(raw_segments)]
        offset += len(raw_segments)
        segments = tuple(
            replace(segment, embedding=tuple(vector))
            for segment, vector in zip(raw_segments, segment_vectors)
        )
        if any(segment.event_type is None for segment in segments):
            raise ValueError("event preprocessing produced an unlabeled segment")
        primitive_counts.update(transition.primitive_action for transition in transitions)
        event_counts.update(
            segment.event_type for segment in segments if segment.event_type is not None
        )
        segment_lengths.update(segment.end_step - segment.start_step + 1 for segment in segments)
        records.append(
            {
                "run_id": trajectory.run_id,
                "benchmark": trajectory.benchmark.value,
                "split": trajectory.split.value,
                "trajectory_method": trajectory.method.value,
                "trajectory_id": trajectory.trajectory_id,
                "sample_id": trajectory.sample_id,
                "repeat_id": trajectory.repeat_id,
                "primary_score": trajectory.primary_score,
                "transitions": [transition.to_record() for transition in transitions],
                "segments": [segment.to_record() for segment in segments],
            }
        )

    options.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=options.output.parent,
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        temporary_path = Path(output_file.name)
        for record in records:
            json.dump(record, output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, options.output)

    report = {
        **invocation,
        "embedding_model": os.environ["EMBEDDING_MODEL"],
        "embedding_dimension": common["embedding_dimension"],
        "embedding_input": "compact_event_segment_signature",
        "embedding_request_count": encoder.embedding_request_count,
        "embedding_input_tokens": encoder.embedding_input_tokens,
        "cached_unique_text_count": encoder.cached_unique_text_count,
        "embedded_unique_text_count": encoder.embedded_unique_text_count,
        "segment_count": sum(segment_lengths.values()),
        "segment_length_distribution": dict(sorted(segment_lengths.items())),
        "primitive_action_distribution": {
            action.value: count for action, count in sorted(primitive_counts.items())
        },
        "event_distribution": {event.value: count for event, count in sorted(event_counts.items())},
    }
    write_json(options.output.with_suffix(".metadata.json"), report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--trajectory-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--embedding-concurrency", type=int)
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
