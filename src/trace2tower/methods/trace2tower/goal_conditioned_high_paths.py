from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from trace2tower.methods.trace2tower.high_paths import trajectory_mid_sequences
from trace2tower.methods.trace2tower.models import HighPath, MidCluster


def mine_goal_conditioned_high_paths(
    records: Sequence[Mapping],
    clusters: Sequence[MidCluster],
    *,
    success_threshold: float = 0.999,
) -> tuple[HighPath, ...]:
    """从每个有成功证据的任务目标中保留一条完整 Mid 组合。"""
    sequences = trajectory_mid_sequences(records, clusters)
    records_by_goal: dict[str, list[Mapping]] = defaultdict(list)
    for record in records:
        goal = next(
            str(transition["goal"]).strip()
            for transition in record["transitions"]
            if transition.get("goal")
        )
        records_by_goal[goal].append(record)

    successful_goal_count = sum(
        any(float(record["primary_score"]) >= success_threshold for record in goal_records)
        for goal_records in records_by_goal.values()
    )
    if successful_goal_count == 0:
        return ()

    paths = []
    for goal, goal_records in sorted(records_by_goal.items()):
        successful_records = [
            record
            for record in goal_records
            if float(record["primary_score"]) >= success_threshold
        ]
        eligible_sequences = [
            sequences[str(record["trajectory_id"])]
            for record in successful_records
            if len(sequences[str(record["trajectory_id"])]) >= 2
            and len(set(sequences[str(record["trajectory_id"])])) >= 2
        ]
        if not eligible_sequences:
            continue

        sequence_counts = Counter(eligible_sequences)
        ordered_mid_ids = min(
            sequence_counts,
            key=lambda sequence: (
                -sequence_counts[sequence],
                len(sequence),
                sequence,
            ),
        )
        supporting_ids = tuple(
            sorted(
                str(record["trajectory_id"])
                for record in successful_records
                if sequences[str(record["trajectory_id"])] == ordered_mid_ids
            )
        )
        failed_records = [
            record
            for record in goal_records
            if float(record["primary_score"]) < success_threshold
        ]
        matching_failure_count = sum(
            sequences[str(record["trajectory_id"])] == ordered_mid_ids
            for record in failed_records
        )
        positive_support = 1.0 / successful_goal_count
        negative_support = (
            matching_failure_count / len(failed_records) if failed_records else 0.0
        )
        path_key = "\x1f".join((goal.casefold(), *ordered_mid_ids)).encode("utf-8")
        paths.append(
            HighPath(
                path_id=f"high_{hashlib.sha256(path_key).hexdigest()[:12]}",
                ordered_mid_ids=ordered_mid_ids,
                positive_support=positive_support,
                negative_support=negative_support,
                contrastive_score=positive_support - negative_support,
                supporting_trajectory_ids=supporting_ids,
                task_condition=goal,
            )
        )
    return tuple(paths)
