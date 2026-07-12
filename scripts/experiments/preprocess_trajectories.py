from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rollout_no_skill_train import load_yaml, write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.segmentation import segment_alfworld_trajectory
from trace2tower.methods.trace2tower.transition_encoder import TransitionEncoder
from trace2tower.methods.trace2tower.transitions import build_transitions, transition_text
from trace2tower.methods.trace2tower.webshop_events import segment_webshop_trajectory
from trace2tower.trajectory import TrajectoryReader


async def main(options: argparse.Namespace) -> None:
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
    invocation = {
        "benchmark": benchmark.value,
        "trajectory_glob": options.trajectory_glob,
        "trajectory_files": [path.as_posix() for path in paths],
        "trajectory_count": len(trajectories),
        "transition_count": sum(len(group) for group in transition_groups),
        "output": options.output.as_posix(),
        "calibration": options.calibration.as_posix(),
        "dry_run": options.dry_run,
    }
    print(yaml.safe_dump({"common": common, "invocation": invocation}))
    if options.dry_run:
        return

    penalty = None
    max_segment_length = None
    if benchmark is Benchmark.ALFWORLD:
        calibration = json.loads(options.calibration.read_text(encoding="utf-8"))
        penalty = float(calibration["penalty"])
        max_segment_length = int(calibration["max_segment_length"])

    load_dotenv(options.env)
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
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
    texts = [
        transition_text(transition)
        for group in transition_groups
        for transition in group
    ]
    try:
        vectors = await encoder.embed(texts)
    finally:
        await runtime.close()

    records = []
    primitive_counts = Counter()
    event_counts = Counter()
    segment_lengths = Counter()
    offset = 0
    for trajectory, transitions in zip(trajectories, transition_groups):
        embeddings = vectors[offset : offset + len(transitions)]
        offset += len(transitions)
        if benchmark is Benchmark.ALFWORLD:
            segments = segment_alfworld_trajectory(
                trajectory,
                transitions,
                embeddings,
                penalty=penalty,
                max_segment_length=max_segment_length,
            )
        else:
            segments = segment_webshop_trajectory(
                trajectory,
                transitions,
                embeddings,
            )
        primitive_counts.update(transition.primitive_action for transition in transitions)
        event_counts.update(
            segment.event_type for segment in segments if segment.event_type is not None
        )
        segment_lengths.update(
            segment.end_step - segment.start_step + 1 for segment in segments
        )
        records.append(
            {
                "run_id": trajectory.run_id,
                "benchmark": trajectory.benchmark.value,
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
        "segment_count": sum(segment_lengths.values()),
        "segment_length_distribution": dict(sorted(segment_lengths.items())),
        "primitive_action_distribution": {
            action.value: count for action, count in sorted(primitive_counts.items())
        },
        "event_distribution": {
            event.value: count for event, count in sorted(event_counts.items())
        },
    }
    write_json(options.output.with_suffix(".metadata.json"), report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--trajectory-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/trace2tower/segmentation-calibration.json"),
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(main(parser.parse_args()))
