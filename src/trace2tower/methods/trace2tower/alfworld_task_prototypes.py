from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from trace2tower.methods.trace2tower.action_parser import parse_alfworld_action
from trace2tower.methods.trace2tower.models import PrimitiveAction


_GOAL_RE = re.compile(r"your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FROM_RE = re.compile(r"^(?:take|pick)\s+(.+?)\s+from\s+(.+?)$", re.IGNORECASE)
_TO_RE = re.compile(r"^(?:put|move)\s+(.+?)\s+to\s+(.+?)$", re.IGNORECASE)
_WITH_RE = re.compile(
    r"^(clean|heat|cool|slice)\s+(.+?)\s+with\s+(.+?)$", re.IGNORECASE
)
_TOGGLE_RE = re.compile(r"^(?:use|toggle)\s+(.+?)$", re.IGNORECASE)
_GOAL_DESTINATION_RE = re.compile(
    r"\b(?:put|place)\b.+?\b(?:in|on|under)\s+"
    r"(?:the\s+|a\s+|an\s+)?([a-z][a-z ]*?)[.!]?$",
    re.IGNORECASE,
)
_GOAL_OBJECT_PATTERNS = (
    re.compile(r"^(?:examine the|look at)\s+([a-z]+)\s+(?:with|under)\b", re.IGNORECASE),
    re.compile(
        r"^(?:heat|cool|clean|slice)\s+(?:some\s+|a\s+|an\s+|the\s+)?([a-z]+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:put|place)\s+(?:a\s+|an\s+|some\s+|two\s+|the\s+)?"
        r"(?:hot\s+|cold\s+|cool\s+|clean\s+|rinsed\s+|wet\s+|chilled\s+|sliced\s+)?"
        r"([a-z]+)\s+(?:in|on|under)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^find\s+two\s+([a-z]+)\b", re.IGNORECASE),
)


def normalize_entity(value: str) -> str:
    """只去掉环境实例编号，保留对象、容器和设备类别。"""
    normalized = " ".join(value.casefold().strip().split())
    return re.sub(r"\s+\d+$", "", normalized)


def canonical_goal(observation: str | None) -> str:
    if not observation:
        return ""
    match = _GOAL_RE.search(observation)
    return " ".join((match.group(1) if match else "").split())


def goal_target_object(goal: str) -> str:
    for pattern in _GOAL_OBJECT_PATTERNS:
        match = pattern.search(goal)
        if match:
            return normalize_entity(match.group(1))
    return ""


def goal_transformation(goal: str) -> str:
    normalized = goal.casefold()
    phrases = (
        (PrimitiveAction.CLEAN, ("clean", "rinsed", "rinse", "wet", "washed")),
        (PrimitiveAction.COOL, ("cool", "cold", "chilled", "chill", "frozen")),
        (PrimitiveAction.HEAT, ("heat", "hot", "warm", "cooked")),
        (PrimitiveAction.SLICE, ("slice", "sliced", "cut")),
        (PrimitiveAction.TOGGLE, ("desklamp", "lamp", "light")),
    )
    for primitive, candidates in phrases:
        if any(candidate in normalized for candidate in candidates):
            return primitive.value
    return "NONE"


def goal_destination(goal: str) -> str:
    match = _GOAL_DESTINATION_RE.search(goal)
    return normalize_entity(match.group(1)) if match else ""


def goal_target_count(goal: str) -> int:
    return 2 if re.search(r"\b(?:two|2)\b", goal, re.IGNORECASE) else 1


@dataclass(frozen=True, slots=True)
class AlfworldTaskPrototype:
    canonical_goal: str
    target_object: str
    source_receptacles: tuple[str, ...]
    transformation: str
    transformation_devices: tuple[str, ...]
    destination_receptacles: tuple[str, ...]
    event_chain: tuple[str, ...]
    target_count: int

    @property
    def exact_key(self) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
        return (
            self.transformation,
            self.target_object,
            self.source_receptacles,
            self.destination_receptacles,
        )

    @property
    def relaxed_key(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.transformation, self.target_object, self.destination_receptacles)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _action_text(step: Mapping[str, Any]) -> str:
    arguments = step.get("action_arguments")
    if isinstance(arguments, Mapping) and isinstance(arguments.get("action"), str):
        return arguments["action"]
    return ""


def _parsed_entities(action: str) -> tuple[PrimitiveAction, tuple[str, ...]]:
    primitive = parse_alfworld_action("take_action", {"action": action})
    normalized = " ".join(action.casefold().strip().split())
    if primitive is PrimitiveAction.PICK:
        match = _FROM_RE.match(normalized)
        if match:
            return primitive, (normalize_entity(match.group(1)), normalize_entity(match.group(2)))
    if primitive is PrimitiveAction.PUT:
        match = _TO_RE.match(normalized)
        if match:
            return primitive, (normalize_entity(match.group(1)), normalize_entity(match.group(2)))
    if primitive in {
        PrimitiveAction.CLEAN,
        PrimitiveAction.HEAT,
        PrimitiveAction.COOL,
        PrimitiveAction.SLICE,
    }:
        match = _WITH_RE.match(normalized)
        if match:
            return primitive, (normalize_entity(match.group(2)), normalize_entity(match.group(3)))
    if primitive is PrimitiveAction.TOGGLE:
        match = _TOGGLE_RE.match(normalized)
        if match:
            return primitive, (normalize_entity(match.group(1)),)
    return primitive, ()


def extract_task_prototype(record: Mapping[str, Any]) -> AlfworldTaskPrototype:
    steps = record.get("steps", ())
    if not isinstance(steps, Sequence):
        raise ValueError("ALFWorld trajectory steps must be a sequence")

    picks: list[tuple[str, str]] = []
    puts: list[tuple[str, str]] = []
    transforms: list[tuple[PrimitiveAction, str, str]] = []
    toggle_devices: list[str] = []
    event_chain: list[str] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        action = _action_text(step)
        primitive, entities = _parsed_entities(action)
        if primitive is PrimitiveAction.INVALID:
            continue
        event_chain.append(primitive.value)
        if primitive is PrimitiveAction.PICK and len(entities) == 2:
            picks.append((entities[0], entities[1]))
        elif primitive is PrimitiveAction.PUT and len(entities) == 2:
            puts.append((entities[0], entities[1]))
        elif primitive in {
            PrimitiveAction.CLEAN,
            PrimitiveAction.HEAT,
            PrimitiveAction.COOL,
            PrimitiveAction.SLICE,
        } and len(entities) == 2:
            transforms.append((primitive, entities[0], entities[1]))
        elif primitive is PrimitiveAction.TOGGLE and len(entities) == 1:
            toggle_devices.append(entities[0])

    goal = canonical_goal(
        str((steps[0] if steps and isinstance(steps[0], Mapping) else {}).get("observation", ""))
    )
    action_target = (
        transforms[-1][1]
        if transforms
        else (puts[-1][0] if puts else (picks[-1][0] if picks else ""))
    )
    target = goal_target_object(goal) or action_target
    target_picks = [source for object_name, source in picks if object_name == target]
    target_puts = [destination for object_name, destination in puts if object_name == target]
    target_transforms = [device for primitive, object_name, device in transforms if object_name == target]
    if transforms:
        transformation = transforms[-1][0].value
    elif toggle_devices:
        transformation = PrimitiveAction.TOGGLE.value
    else:
        transformation = "NONE"
    if not target_puts and "put" in goal.casefold():
        destination_match = _GOAL_DESTINATION_RE.search(goal)
        if destination_match:
            target_puts.append(normalize_entity(destination_match.group(1)))
    return AlfworldTaskPrototype(
        canonical_goal=goal,
        target_object=target,
        source_receptacles=tuple(dict.fromkeys(target_picks)),
        transformation=transformation,
        transformation_devices=tuple(dict.fromkeys((*target_transforms, *toggle_devices))),
        destination_receptacles=tuple(dict.fromkeys(target_puts)),
        event_chain=tuple(dict.fromkeys(event_chain)),
        target_count=max(1, sum(1 for object_name, _ in puts if object_name == target)),
    )
