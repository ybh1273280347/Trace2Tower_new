from __future__ import annotations

from trace2tower.core.manifests import Benchmark
from trace2tower.core.trajectory import EpisodeTrajectory
from trace2tower.methods.trace2tower.adapters.alfworld.actions import (
    parse_alfworld_action,
)
from trace2tower.methods.trace2tower.adapters.webshop.actions import (
    parse_webshop_action,
)
from trace2tower.methods.trace2tower.core.models import StepTransition
from trace2tower.methods.trace2tower.preprocessing.transitions import build_transitions


def build_benchmark_transitions(
    trajectory: EpisodeTrajectory,
) -> tuple[StepTransition, ...]:
    action_parser = (
        parse_alfworld_action
        if trajectory.benchmark is Benchmark.ALFWORLD
        else parse_webshop_action
    )
    return build_transitions(trajectory, action_parser)
