from __future__ import annotations

import math
from statistics import median
from typing import Sequence

from trace2tower.methods.trace2tower.models import (
    SegmentInstance,
    SegmentationCalibration,
    StepTransition,
)
from trace2tower.trajectory import EpisodeTrajectory


def segment_boundaries(
    embeddings: Sequence[Sequence[float]],
    *,
    penalty: float,
    max_segment_length: int = 6,
) -> tuple[tuple[int, int], ...]:
    if penalty < 0 or max_segment_length <= 0:
        raise ValueError("penalty must be non-negative and max segment length positive")
    if not embeddings:
        return ()
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(vector) != dimension for vector in embeddings):
        raise ValueError("embeddings must have one nonzero fixed dimension")

    segment_costs = _prepare_segment_costs(embeddings, max_segment_length)
    return _boundaries_from_costs(segment_costs, penalty)


def _prepare_segment_costs(
    embeddings: Sequence[Sequence[float]], max_segment_length: int
) -> tuple[tuple[float, ...], ...]:
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(vector) != dimension for vector in embeddings):
        raise ValueError("embeddings must have one nonzero fixed dimension")
    normalized = []
    for vector in embeddings:
        norm = math.sqrt(sum(value * value for value in vector))
        normalized.append(
            [value / norm for value in vector] if norm else [0.0] * dimension
        )
    prefix = [[0.0] * dimension]
    prefix_squared = [0.0]
    for vector in normalized:
        prefix.append([left + right for left, right in zip(prefix[-1], vector)])
        prefix_squared.append(prefix_squared[-1] + sum(value * value for value in vector))

    segment_costs = [()]
    for end in range(1, len(normalized) + 1):
        current = []
        for length in range(1, min(max_segment_length, end) + 1):
            start = end - length
            sums = [right - left for left, right in zip(prefix[start], prefix[end])]
            current.append(
                max(
                    0.0,
                    prefix_squared[end]
                    - prefix_squared[start]
                    - sum(value * value for value in sums) / length,
                )
            )
        segment_costs.append(tuple(current))
    return tuple(segment_costs)


def _boundaries_from_costs(
    segment_costs: Sequence[Sequence[float]], penalty: float
) -> tuple[tuple[int, int], ...]:
    transition_count = len(segment_costs) - 1
    costs = [math.inf] * (transition_count + 1)
    previous = [-1] * (transition_count + 1)
    costs[0] = 0.0
    for end in range(1, transition_count + 1):
        for length, within_cost in enumerate(segment_costs[end], start=1):
            start = end - length
            candidate = costs[start] + within_cost + penalty
            if candidate < costs[end] - 1e-12:
                costs[end] = candidate
                previous[end] = start

    boundaries = []
    end = transition_count
    while end:
        start = previous[end]
        if start < 0:
            raise RuntimeError("segmentation dynamic program has no valid path")
        boundaries.append((start, end - 1))
        end = start
    return tuple(reversed(boundaries))


def calibrate_segmentation_penalty(
    trajectory_embeddings: Sequence[Sequence[Sequence[float]]],
    *,
    target_segment_length: int = 3,
    max_segment_length: int = 6,
    iterations: int = 40,
) -> SegmentationCalibration:
    sequences = tuple(sequence for sequence in trajectory_embeddings if sequence)
    if (
        not sequences
        or not 1 <= target_segment_length <= max_segment_length
        or iterations <= 0
    ):
        raise ValueError("calibration requires trajectories, valid lengths, and iterations")

    prepared_costs = tuple(
        _prepare_segment_costs(sequence, max_segment_length)
        for sequence in sequences
    )

    def evaluate(penalty: float) -> tuple[float, list[int]]:
        lengths = [
            end - start + 1
            for costs in prepared_costs
            for start, end in _boundaries_from_costs(costs, penalty)
        ]
        return float(median(lengths)), lengths

    low = 0.0
    high = 1.0
    low_median, low_lengths = evaluate(low)
    candidates = [
        (
            abs(low_median - target_segment_length),
            low,
            low_median,
            low_lengths,
        )
    ]
    for _ in range(40):
        high_median, high_lengths = evaluate(high)
        candidates.append(
            (
                abs(high_median - target_segment_length),
                high,
                high_median,
                high_lengths,
            )
        )
        if high_median >= target_segment_length:
            break
        high *= 2
    else:
        high_median = low_median

    if high_median >= target_segment_length:
        for _ in range(iterations):
            current = (low + high) / 2
            current_median, lengths = evaluate(current)
            candidates.append(
                (
                    abs(current_median - target_segment_length),
                    current,
                    current_median,
                    lengths,
                )
            )
            if current_median < target_segment_length:
                low = current
            else:
                high = current
    _, penalty, observed_median, lengths = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return SegmentationCalibration(
        penalty=penalty,
        target_segment_length=target_segment_length,
        median_segment_length=observed_median,
        trajectory_count=len(sequences),
        segment_count=len(lengths),
    )


def segment_alfworld_trajectory(
    trajectory: EpisodeTrajectory,
    transitions: Sequence[StepTransition],
    embeddings: Sequence[Sequence[float]],
    *,
    penalty: float,
    max_segment_length: int = 6,
) -> tuple[SegmentInstance, ...]:
    if len(transitions) != len(trajectory.steps) or len(embeddings) != len(transitions):
        raise ValueError("trajectory, transitions, and embeddings must align")
    segments = []
    for start, end in segment_boundaries(
        embeddings,
        penalty=penalty,
        max_segment_length=max_segment_length,
    ):
        segment_vectors = embeddings[start : end + 1]
        embedding = tuple(
            sum(vector[index] for vector in segment_vectors) / len(segment_vectors)
            for index in range(len(segment_vectors[0]))
        )
        segments.append(
            SegmentInstance(
                segment_id=f"{trajectory.trajectory_id}:segment:{start}-{end}",
                trajectory_id=trajectory.trajectory_id,
                start_step=start,
                end_step=end,
                transition_ids=tuple(
                    transition.transition_id for transition in transitions[start : end + 1]
                ),
                embedding=embedding,
                trajectory_score=trajectory.primary_score,
                event_type=None,
                raw_actions=tuple(
                    transition.raw_action for transition in transitions[start : end + 1]
                ),
                observation_before=transitions[start].observation_before,
                observation_after=transitions[end].observation_after,
            )
        )
    return tuple(segments)
