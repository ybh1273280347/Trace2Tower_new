from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    alfworld_applicable_events,
    alfworld_goal_events,
)
from trace2tower.methods.trace2tower.graph_retrieval import (
    TowerGraphProfile,
    retrieve_tower_graph,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "artifacts/runs/"
            "alfworld-dev-v1-flash-operator-mid-graph-high-cap3-r0"
        ),
    )
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--tower", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/alfworld/official/validation/"
            "operator-mid-v4-offline-retrieval-audit.json"
        ),
    )
    return parser.parse_args()


def read_trajectories(run_dir: Path) -> list[dict]:
    trajectories = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in run_dir.rglob("trajectories/*.json")
    ]
    if not trajectories:
        raise ValueError(f"no trajectories found under {run_dir}")
    return sorted(trajectories, key=lambda item: item["sample_id"])


async def embed_texts(
    runtime: CommonLLMRuntime,
    texts: set[str],
    batch_size: int,
) -> dict[str, tuple[float, ...]]:
    ordered = sorted(texts)
    batches = [
        ordered[start : start + batch_size]
        for start in range(0, len(ordered), batch_size)
    ]
    results = await asyncio.gather(*(runtime.embed(batch) for batch in batches))
    vectors = [vector for result in results for vector in result.vectors]
    return dict(zip(ordered, vectors, strict=True))


def represented_exclusive_events(
    mid_id: str,
    profile: TowerGraphProfile,
    minimum_compatibility: float,
) -> frozenset:
    return frozenset(
        event
        for event in ALFWORLD_EXCLUSIVE_PATH_EVENTS
        if profile.compatibility(mid_id, frozenset((event,)))
        >= minimum_compatibility
    )


async def audit(options: argparse.Namespace) -> dict:
    if options.embedding_batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    resolved = yaml.safe_load(
        (options.run_dir / "resolved-config.yaml").read_text(encoding="utf-8")
    )
    method = resolved["method"]
    tower_path = options.tower or Path(resolved["artifacts"]["alfworld"]["path"])
    profile_path = options.profile or Path(method["graph_profile"])
    snapshot = TowerSnapshot.from_record(
        json.loads(
            tower_path.read_text(encoding="utf-8")
        )
    )
    profile = TowerGraphProfile.from_record(
        json.loads(profile_path.read_text(encoding="utf-8"))
    )
    trajectories = read_trajectories(options.run_dir)
    texts = {trajectory["task_goal"] for trajectory in trajectories}
    texts.update(
        step["observation"]
        for trajectory in trajectories
        for step in trajectory["steps"]
    )

    runtime = CommonLLMRuntime(
        max_concurrency=5,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    try:
        embeddings = await embed_texts(
            runtime,
            texts,
            options.embedding_batch_size,
        )
    finally:
        await runtime.close()

    high_cards = {card.skill_id: card for card in snapshot.high_cards}
    mid_cards = {card.skill_id: card for card in snapshot.mid_cards}
    high_paths = {path.path_id: path for path in snapshot.high_paths}
    minimum_compatibility = float(method["min_event_compatibility"])
    counts = Counter()
    initial_records = []

    for trajectory in trajectories:
        goal = trajectory["task_goal"]
        required_events = alfworld_goal_events(goal)
        for step in trajectory["steps"]:
            allowed_events = alfworld_applicable_events(step["admissible_actions"])
            result = retrieve_tower_graph(
                embeddings[goal],
                embeddings[step["observation"]],
                snapshot.high_index,
                snapshot.mid_index,
                high_cards,
                mid_cards,
                high_paths,
                profile,
                allowed_events,
                mid_context_budget=int(method["mid_context_budget"]),
                high_similarity_threshold=float(method["high_similarity_threshold"]),
                direct_mid_dedup_similarity_threshold=float(
                    method["direct_mid_dedup_similarity_threshold"]
                ),
                direct_mid_mmr_lambda=float(method["direct_mid_mmr_lambda"]),
                min_event_compatibility=minimum_compatibility,
                required_path_events=required_events,
                exclusive_path_events=ALFWORLD_EXCLUSIVE_PATH_EVENTS,
            )
            counts["steps"] += 1
            selected_mid_ids = [
                skill_id
                for skill_id in result.retrieval.context_skill_ids
                if skill_id.startswith("mid_")
            ]
            selected_exclusive_events = frozenset().union(
                *(
                    represented_exclusive_events(
                        mid_id,
                        profile,
                        minimum_compatibility,
                    )
                    for mid_id in selected_mid_ids
                ),
                frozenset(),
            )
            if selected_exclusive_events - allowed_events:
                counts["selected_unavailable_exclusive_event"] += 1
            if selected_exclusive_events - required_events:
                counts["selected_wrong_goal_exclusive_event"] += 1
            if required_events & allowed_events:
                counts["goal_operator_actionable_steps"] += 1
                if required_events <= selected_exclusive_events:
                    counts["goal_operator_mid_retrieved_when_actionable"] += 1
            elif selected_exclusive_events & required_events:
                counts["goal_operator_mid_retrieved_before_actionable"] += 1

            high_match = result.graph_high_match
            if high_match:
                counts["steps_with_high"] += 1
                high_events = frozenset().union(
                    *(
                        represented_exclusive_events(
                            mid_id,
                            profile,
                            minimum_compatibility,
                        )
                        for mid_id in high_cards[
                            high_match.skill_id
                        ].ordered_mid_ids
                    ),
                    frozenset(),
                )
                if high_events != required_events:
                    counts["high_goal_event_mismatch"] += 1
            elif required_events:
                counts["operator_steps_without_high"] += 1

            if step["step_index"] == 0:
                counts["episodes"] += 1
                old_ids = step.get("retrieved_context_skill_ids", ())
                old_first_mid = next(
                    (skill_id for skill_id in old_ids if skill_id.startswith("mid_")),
                    None,
                )
                old_events = (
                    represented_exclusive_events(
                        old_first_mid,
                        profile,
                        minimum_compatibility,
                    )
                    if old_first_mid
                    else frozenset()
                )
                if old_events & required_events:
                    counts["old_initial_goal_operator_mid"] += 1
                if selected_exclusive_events & required_events:
                    counts["new_initial_goal_operator_mid"] += 1
                initial_records.append(
                    {
                        "sample_id": trajectory["sample_id"],
                        "goal": goal,
                        "admissible_event_types": sorted(allowed_events),
                        "old_context_skill_ids": old_ids,
                        "new_context_skill_ids": result.retrieval.context_skill_ids,
                        "selected_high_name": (
                            result.retrieval.high_card.name
                            if result.retrieval.high_card
                            else None
                        ),
                        "active_mid_id": (
                            high_match.active_mid_id if high_match else None
                        ),
                        "active_mid_name": (
                            mid_cards[high_match.active_mid_id].name
                            if high_match
                            else None
                        ),
                        "first_action": step["action_arguments"].get("action"),
                    }
                )

    return {
        "source_run": str(options.run_dir),
        "tower": str(tower_path),
        "profile": str(profile_path),
        "tower_snapshot_id": snapshot.snapshot_id,
        "embedding_text_count": len(texts),
        "counts": dict(counts),
        "initial_records": initial_records,
    }


async def main() -> int:
    options = parse_args()
    load_dotenv(options.env)
    report = await audit(options)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"wrote {options.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
