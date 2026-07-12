from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Mapping

from trace2tower.results import FinishReason
from trace2tower.trajectory import EpisodeTrajectory


@dataclass(frozen=True, slots=True)
class TrajectoryQualitySummary:
    episode_count: int
    full_success_count: int
    full_success_rate: float
    successful_valid_count: int
    mean_primary_score: float
    completion_rate: float
    valid_action_rate: float
    mean_steps: float
    mean_success_steps: float | None
    mean_repeated_action_rate: float
    mean_observation_change_rate: float
    input_tokens: int
    output_tokens: int
    token_usage_episodes: int
    tokens_per_success: float | None
    mean_latency_ms: float

    def to_record(self) -> dict:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def summarize_trajectory_quality(
    trajectories: Iterable[EpisodeTrajectory],
    results: Iterable[Mapping],
    *,
    success_threshold: float = 0.999,
) -> TrajectoryQualitySummary:
    episodes = tuple(trajectories)
    result_records = tuple(results)
    if not episodes or len(episodes) != len(result_records):
        raise ValueError("trajectory and result counts must match and be nonzero")

    result_by_key = {
        (record["sample_id"], int(record["repeat_id"])): record
        for record in result_records
    }
    if len(result_by_key) != len(result_records):
        raise ValueError("duplicate result episode key")
    if {
        (episode.sample_id, episode.repeat_id) for episode in episodes
    } != set(result_by_key):
        raise ValueError("trajectory and result episode keys do not match")

    successful = [episode for episode in episodes if episode.primary_score >= success_threshold]
    total_steps = sum(len(episode.steps) for episode in episodes)
    valid_steps = sum(
        step.valid_action for episode in episodes for step in episode.steps
    )
    repeated_rates = []
    observation_change_rates = []
    for episode in episodes:
        actions = [
            (step.action_name, json.dumps(step.action_arguments, sort_keys=True))
            for step in episode.steps
        ]
        repeated_rates.append(
            sum(left == right for left, right in zip(actions, actions[1:]))
            / max(len(actions) - 1, 1)
        )
        observation_change_rates.append(
            sum(step.observation != step.next_observation for step in episode.steps)
            / max(len(episode.steps), 1)
        )

    usage_records = [
        record
        for record in result_records
        if record.get("input_tokens") is not None
        and record.get("output_tokens") is not None
    ]
    input_tokens = sum(int(record["input_tokens"]) for record in usage_records)
    output_tokens = sum(int(record["output_tokens"]) for record in usage_records)
    return TrajectoryQualitySummary(
        episode_count=len(episodes),
        full_success_count=len(successful),
        full_success_rate=len(successful) / len(episodes),
        successful_valid_count=sum(
            all(step.valid_action for step in episode.steps) for episode in successful
        ),
        mean_primary_score=fmean(episode.primary_score for episode in episodes),
        completion_rate=sum(
            episode.finish_reason is FinishReason.COMPLETED for episode in episodes
        )
        / len(episodes),
        valid_action_rate=valid_steps / max(total_steps, 1),
        mean_steps=fmean(len(episode.steps) for episode in episodes),
        mean_success_steps=(
            fmean(len(episode.steps) for episode in successful)
            if successful
            else None
        ),
        mean_repeated_action_rate=fmean(repeated_rates),
        mean_observation_change_rate=fmean(observation_change_rates),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_usage_episodes=len(usage_records),
        tokens_per_success=(
            (input_tokens + output_tokens) / len(successful) if successful else None
        ),
        mean_latency_ms=fmean(int(record["latency_ms"]) for record in result_records),
    )
