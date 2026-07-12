from __future__ import annotations

import json

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.action_parser import (
    parse_alfworld_action,
    parse_webshop_action,
)
from trace2tower.methods.trace2tower.models import StepTransition
from trace2tower.trajectory import EpisodeTrajectory, StepRecord


def build_transitions(trajectory: EpisodeTrajectory) -> tuple[StepTransition, ...]:
    transitions = []
    for step in trajectory.steps:
        primitive = (
            parse_alfworld_action(step.action_name, step.action_arguments)
            if trajectory.benchmark is Benchmark.ALFWORLD
            else parse_webshop_action(step.action_name, step.action_arguments)
        )
        transitions.append(
            StepTransition(
                transition_id=f"{trajectory.trajectory_id}:step:{step.step_index}",
                trajectory_id=trajectory.trajectory_id,
                step_index=step.step_index,
                goal=trajectory.task_goal,
                observation_before=step.observation,
                raw_action=raw_action(step),
                primitive_action=primitive,
                observation_after=step.next_observation,
                trajectory_score=trajectory.primary_score,
            )
        )
    return tuple(transitions)


def raw_action(step: StepRecord) -> str:
    if step.action_name == "take_action" and step.action_arguments:
        action = step.action_arguments.get("action")
        if isinstance(action, str):
            return action
    return json.dumps(
        {"name": step.action_name, "arguments": step.action_arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def transition_text(transition: StepTransition) -> str:
    return (
        f"Goal:\n{transition.goal}\n\n"
        f"Observation Before:\n{transition.observation_before}\n\n"
        f"Action:\n{transition.primitive_action} | {transition.raw_action}\n\n"
        f"Observation After:\n{transition.observation_after}"
    )
