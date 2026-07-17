from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from scripts.experiments.run.run_matrix import parse_shard_ids
from trace2tower.core.manifests import Benchmark, read_manifest
from trace2tower.core.results import MethodName
from trace2tower.core.trajectory import TrajectoryReader, write_trajectory_jsonl
from trace2tower.data.trajectory_pool import audit_training_shard


def main(options: argparse.Namespace) -> int:
    common = load_yaml(options.config_root / "common.yaml")
    if options.benchmark == "all":
        raise ValueError("merge_shards requires one benchmark")
    benchmark = Benchmark(options.benchmark)
    shard_ids = parse_shard_ids(options.shard_id, options.num_shards)
    invocation = {
        "benchmark": benchmark.value,
        "method": str(options.method),
        "shard_ids": list(shard_ids),
        "num_shards": options.num_shards,
        "max_episodes": options.max_episodes,
        "run_id": options.run_id,
        "trajectory_pool_name": options.trajectory_pool_name,
        "dry_run": options.dry_run,
    }
    print(yaml.safe_dump({"common": common, "invocation": invocation}))

    entries = read_manifest(Path(common["manifests_dir"]) / f"{benchmark}_train.jsonl")
    audits = []
    paths = []
    for shard_id in shard_ids:
        shard_name = f"shard-{shard_id:02d}"
        pool_path = (
            Path(common["trajectories_dir"])
            / benchmark
            / options.trajectory_pool_name
            / f"{shard_name}.jsonl"
        )
        paths.append(pool_path)
        audits.append(
            audit_training_shard(
                entries,
                run_id=options.run_id,
                benchmark=benchmark,
                method=MethodName.NO_SKILL,
                shard_id=shard_id,
                num_shards=options.num_shards,
                run_dir=(
                    Path(common["runs_dir"])
                    / options.run_id
                    / benchmark
                    / "no_skill_train"
                    / shard_name
                ),
                pool_path=pool_path,
                max_episodes=options.max_episodes,
            )
        )
    if not all(audit.complete for audit in audits):
        print(yaml.safe_dump({"shards": [audit.to_record() for audit in audits]}))
        return 1

    trajectories = [
        trajectory for path in paths for trajectory in TrajectoryReader.read_jsonl(path)
    ]
    suffix = (
        "all"
        if shard_ids == tuple(range(options.num_shards))
        else "shards-" + "-".join(f"{shard_id:02d}" for shard_id in shard_ids)
    )
    output = options.output or (
        Path(common["trajectories_dir"])
        / benchmark
        / options.trajectory_pool_name
        / f"{suffix}.jsonl"
    )
    if not options.dry_run:
        count = write_trajectory_jsonl(trajectories, output)
        metadata = {
            "run_id": options.run_id,
            "benchmark": benchmark.value,
            "method": MethodName.NO_SKILL.value,
            "shard_ids": list(shard_ids),
            "trajectory_count": count,
            "output": output.as_posix(),
        }
        write_json(
            Path(common["runs_dir"])
            / options.run_id
            / benchmark
            / "no_skill_train"
            / f"merge-{suffix}.json",
            metadata,
        )
        print(yaml.safe_dump(metadata, sort_keys=False))
    else:
        print(yaml.safe_dump({"trajectory_count": len(trajectories), "output": output.as_posix()}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--method", choices=(MethodName.NO_SKILL,), default=MethodName.NO_SKILL)
    parser.add_argument("--shard-id", default="all")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--run-id", default="no-skill-train-v1")
    parser.add_argument("--trajectory-pool-name", default="no_skill_train")
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    raise SystemExit(main(parser.parse_args()))
