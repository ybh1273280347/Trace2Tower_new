from __future__ import annotations

import re

from trace2tower.benchmarks.models import EnvironmentState

_GOAL_RE = re.compile(r"your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)


class AlfworldPlanRewriteAdapter:
    domain = "alfworld"
    plan_rewrite_instructions = (
        "ALFWorld plan semantics:\n"
        "- If the target's source is not explicitly observed, do not copy or guess a source "
        "location from a reference plan. Search each candidate location at most once and "
        "prefer already visible or open locations before closed ones.\n"
        "- Avoid initial look or inventory actions when the initial state already provides "
        "that information. Take the exact target immediately when it appears and never "
        "substitute a related object type.\n"
        "- Preserve navigation, receptacle state, possession, appliance or tool, "
        "transformation, quantity, and final-placement prerequisites required by the current "
        "task.\n"
        "- Do not close an empty searched container unless closure is required by a later "
        "demonstrated action. Stop when the environment confirms the current objective.\n"
    )

    def task_text(self, task_goal: str, state: EnvironmentState) -> str:
        match = _GOAL_RE.search(state.observation)
        observation_goal = " ".join((match.group(1) if match else "").split())
        return observation_goal or task_goal
