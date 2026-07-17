from __future__ import annotations

from typing import Any

from trace2tower.methods.trace2tower.core.models import PrimitiveAction


def parse_alfworld_action(
    tool_name: str | None,
    arguments: dict[str, Any] | None,
) -> PrimitiveAction:
    if tool_name != "take_action" or not arguments or not isinstance(
        arguments.get("action"),
        str,
    ):
        return PrimitiveAction.INVALID

    action = arguments["action"].strip().casefold()
    exact_actions = {
        "inventory": PrimitiveAction.INVENTORY,
        "look": PrimitiveAction.LOOK,
    }
    if action in exact_actions:
        return exact_actions[action]

    prefixes = (
        ("go to ", PrimitiveAction.GOTO),
        ("take ", PrimitiveAction.PICK),
        ("pick ", PrimitiveAction.PICK),
        ("put ", PrimitiveAction.PUT),
        ("move ", PrimitiveAction.PUT),
        ("open ", PrimitiveAction.OPEN),
        ("close ", PrimitiveAction.CLOSE),
        ("use ", PrimitiveAction.TOGGLE),
        ("toggle ", PrimitiveAction.TOGGLE),
        ("heat ", PrimitiveAction.HEAT),
        ("clean ", PrimitiveAction.CLEAN),
        ("cool ", PrimitiveAction.COOL),
        ("slice ", PrimitiveAction.SLICE),
        ("examine ", PrimitiveAction.EXAMINE),
    )
    for prefix, primitive in prefixes:
        if action.startswith(prefix) and len(action) > len(prefix):
            return primitive
    return PrimitiveAction.INVALID
