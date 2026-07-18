from __future__ import annotations

from typing import Any

from trace2tower.core.tower_models import PrimitiveAction


def parse_webshop_action(
    tool_name: str | None,
    arguments: dict[str, Any] | None,
) -> PrimitiveAction:
    if not arguments:
        return PrimitiveAction.INVALID
    if tool_name == "search_action" and isinstance(arguments.get("keywords"), str):
        return PrimitiveAction.SEARCH
    if tool_name == "click_action" and isinstance(arguments.get("value"), str):
        return PrimitiveAction.CLICK
    return PrimitiveAction.INVALID
