from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from scripts.experiments.build.build_trace2tower_skills import load_skill_records
from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.action_parser import parse_alfworld_action
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    PRIMITIVE_EVENTS,
    alfworld_applicable_events,
    alfworld_goal_events,
)
from trace2tower.methods.trace2tower.graph_retrieval import TowerGraphProfile
from trace2tower.methods.trace2tower.models import MidCluster, PrimitiveAction
from trace2tower.methods.trace2tower.three_signal_retrieval import (
    build_mid_transition_signal_profile,
    retrieve_mid_three_signal,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    tower = TowerSnapshot.from_record(
        json.loads(options.tower.read_text(encoding="utf-8"))
    )
    event_profile = TowerGraphProfile.from_record(
        json.loads(options.event_profile.read_text(encoding="utf-8"))
    )
    records = load_skill_records(options.preprocessed)
    clusters = tuple(
        MidCluster.from_record(item)
        for item in json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    )
    signal_profile = build_mid_transition_signal_profile(records, clusters)
    trajectory_paths = tuple(sorted(options.trajectories.rglob("*.json")))
    trajectories = [
        json.loads(path.read_text(encoding="utf-8")) for path in trajectory_paths
    ]
    queries = [
        f"{trajectory['task_goal']}\n{step['observation']}"
        for trajectory in trajectories
        for step in trajectory["steps"]
    ]

    runtime = CommonLLMRuntime(
        max_concurrency=options.concurrency,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    vectors = []
    try:
        for start in range(0, len(queries), options.batch_size):
            result = await runtime.embed(queries[start : start + options.batch_size])
            vectors.extend(result.vectors)
    finally:
        await runtime.close()

    threshold_stats = {
        threshold: Counter() for threshold in options.threshold
    }
    baseline = Counter()
    vector_index = 0
    for trajectory in trajectories:
        required_events = alfworld_goal_events(trajectory["task_goal"])
        for step in trajectory["steps"]:
            vector = vectors[vector_index]
            vector_index += 1
            allowed_events = alfworld_applicable_events(step["admissible_actions"])
            candidate_ids = frozenset(
                mid_id
                for mid_id in tower.mid_index.skill_ids
                if event_profile.compatibility(mid_id, allowed_events) >= 0.1
                and frozenset(
                    event
                    for event in ALFWORLD_EXCLUSIVE_PATH_EVENTS
                    if event_profile.compatibility(mid_id, frozenset((event,))) >= 0.1
                )
                <= required_events
            )
            primitive = parse_alfworld_action(
                step["action_name"],
                step["action_arguments"],
            )
            expected_event = (
                PRIMITIVE_EVENTS[primitive]
                if primitive is not PrimitiveAction.INVALID
                else None
            )

            baseline_ids = tuple(
                skill_id
                for skill_id in step["retrieved_skill_ids"]
                if skill_id.startswith("mid_")
            )
            baseline["steps"] += 1
            if baseline_ids:
                baseline["injected_steps"] += 1
                if expected_event is not None and any(
                    event_profile.compatibility(mid_id, frozenset((expected_event,))) >= 0.5
                    for mid_id in baseline_ids
                ):
                    baseline["aligned_steps"] += 1

            for threshold in options.threshold:
                matches = retrieve_mid_three_signal(
                    vector,
                    tower.mid_index,
                    candidate_ids,
                    signal_profile,
                    top_k=options.top_k,
                    score_threshold=threshold,
                )
                stats = threshold_stats[threshold]
                stats["steps"] += 1
                if matches:
                    stats["injected_steps"] += 1
                    stats["selected_cards"] += len(matches)
                    if expected_event is not None and any(
                        event_profile.compatibility(
                            match.skill_id,
                            frozenset((expected_event,)),
                        )
                        >= 0.5
                        for match in matches
                    ):
                        stats["aligned_steps"] += 1

    def report(stats: Counter) -> dict:
        injected = stats["injected_steps"]
        return {
            **dict(stats),
            "coverage": injected / stats["steps"],
            "aligned_precision": stats["aligned_steps"] / injected if injected else None,
        }

    output = {
        "trajectory_count": len(trajectories),
        "step_count": len(queries),
        "baseline": report(baseline),
        "three_signal": {
            str(threshold): report(stats)
            for threshold, stats in threshold_stats.items()
        },
        "signal_profile": signal_profile.to_record(),
    }
    write_json(options.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--event-profile", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", action="append", type=float, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
