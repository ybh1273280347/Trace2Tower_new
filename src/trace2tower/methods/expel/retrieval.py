from __future__ import annotations

from trace2tower.core.manifests import Benchmark


def task_scope(benchmark: Benchmark, task_goal: str) -> str:
    if benchmark is Benchmark.WEBSHOP:
        return "webshop"

    goal = task_goal.casefold()
    if "two " in goal or "two of" in goal:
        return "pick_two_obj"
    if any(word in goal for word in ("clean", "washed")):
        return "pick_clean_then_place"
    if any(word in goal for word in ("heat", "hot")):
        return "pick_heat_then_place"
    if any(word in goal for word in ("cool", "cold")):
        return "pick_cool_then_place"
    if any(word in goal for word in ("look at", "examine", "desklamp")):
        return "look_at_obj"
    return "pick_and_place"
