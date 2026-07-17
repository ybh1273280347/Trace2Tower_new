from __future__ import annotations

from collections.abc import Mapping, Sequence


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right) or not left:
        raise ValueError("Pareto vectors must have the same non-zero dimension")
    differences = tuple(left_value - right_value for left_value, right_value in zip(left, right))
    return all(value >= 0 for value in differences) and any(value > 0 for value in differences)


def rank_fronts(objectives: Mapping[str, Sequence[float]]) -> dict[str, int]:
    if not objectives:
        return {}
    dimensions = {len(values) for values in objectives.values()}
    if len(dimensions) != 1 or dimensions == {0}:
        raise ValueError("Pareto vectors must have the same non-zero dimension")

    remaining = set(objectives)
    ranks = {}
    rank = 1
    while remaining:
        front = {
            item_id
            for item_id in remaining
            if not any(
                dominates(objectives[other_id], objectives[item_id])
                for other_id in remaining
                if other_id != item_id
            )
        }
        for item_id in front:
            ranks[item_id] = rank
        remaining -= front
        rank += 1
    return ranks
