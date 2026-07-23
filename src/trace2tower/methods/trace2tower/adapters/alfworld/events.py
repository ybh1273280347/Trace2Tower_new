from __future__ import annotations

import re
from collections.abc import Sequence

from trace2tower.core.trajectory import EpisodeTrajectory
from trace2tower.methods.trace2tower.adapters.alfworld.actions import (
    parse_alfworld_action,
)
from trace2tower.methods.trace2tower.core.models import (
    AlfworldEventType,
    PrimitiveAction,
    SegmentInstance,
    StepTransition,
)

PRIMITIVE_EVENTS = {
    PrimitiveAction.GOTO: AlfworldEventType.GOTO_LOCATION,
    PrimitiveAction.PICK: AlfworldEventType.PICKUP_OBJECT,
    PrimitiveAction.PUT: AlfworldEventType.PUT_OBJECT,
    PrimitiveAction.OPEN: AlfworldEventType.OPEN_OBJECT,
    PrimitiveAction.CLOSE: AlfworldEventType.CLOSE_OBJECT,
    PrimitiveAction.TOGGLE: AlfworldEventType.TOGGLE_OBJECT,
    PrimitiveAction.SLICE: AlfworldEventType.SLICE_OBJECT,
    PrimitiveAction.CLEAN: AlfworldEventType.CLEAN_OBJECT,
    PrimitiveAction.HEAT: AlfworldEventType.HEAT_OBJECT,
    PrimitiveAction.COOL: AlfworldEventType.COOL_OBJECT,
    PrimitiveAction.INVENTORY: AlfworldEventType.SCAN,
    PrimitiveAction.EXAMINE: AlfworldEventType.SCAN,
    PrimitiveAction.LOOK: AlfworldEventType.SCAN,
    PrimitiveAction.INVALID: AlfworldEventType.INVALID_ACTION,
}

EVENT_ACTION_TEMPLATES = {
    AlfworldEventType.GOTO_LOCATION: "GO_TO(receptacle)",
    AlfworldEventType.PICKUP_OBJECT: "PICK_UP(object, source)",
    AlfworldEventType.PUT_OBJECT: "PUT(object, destination)",
    AlfworldEventType.OPEN_OBJECT: "OPEN(receptacle)",
    AlfworldEventType.CLOSE_OBJECT: "CLOSE(receptacle)",
    AlfworldEventType.TOGGLE_OBJECT: "TOGGLE(object)",
    AlfworldEventType.SLICE_OBJECT: "SLICE(object, tool)",
    AlfworldEventType.CLEAN_OBJECT: "CLEAN(object, appliance)",
    AlfworldEventType.HEAT_OBJECT: "HEAT(object, appliance)",
    AlfworldEventType.COOL_OBJECT: "COOL(object, appliance)",
    AlfworldEventType.INVALID_ACTION: "INVALID_ACTION",
}

ALFWORLD_GOAL_EVENT_PHRASES = {
    AlfworldEventType.CLEAN_OBJECT: (
        "clean",
        "wash",
        "washed",
        "wet",
        "rinse",
        "rinsed",
    ),
    AlfworldEventType.HEAT_OBJECT: (
        "heat",
        "heated",
        "hot",
        "warm",
        "microwaved",
        "cooked",
    ),
    AlfworldEventType.COOL_OBJECT: (
        "cool",
        "cooled",
        "cold",
        "chill",
        "chilled",
        "frozen",
    ),
    AlfworldEventType.SLICE_OBJECT: ("slice", "sliced", "cut"),
    AlfworldEventType.TOGGLE_OBJECT: (
        "lamp",
        "light",
        "turn on",
        "toggle",
        "illuminate",
    ),
}
ALFWORLD_EXCLUSIVE_PATH_EVENTS = frozenset(ALFWORLD_GOAL_EVENT_PHRASES)

_OBSERVED_GOAL_RE = re.compile(r"your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_INSTANCE_RE = re.compile(r"\s+\d+$")
_FROM_RE = re.compile(r"^(?:take|pick)\s+(.+?)\s+from\s+(.+?)$", re.IGNORECASE)
_TO_RE = re.compile(
    r"^(?:put|move)\s+(.+?)\s+(?:to|in/on|in|on)\s+(.+?)$",
    re.IGNORECASE,
)
_WITH_RE = re.compile(
    r"^(clean|heat|cool|slice)\s+(.+?)\s+with\s+(.+?)$",
    re.IGNORECASE,
)
_UNARY_RE = re.compile(
    r"^(go to|open|close|use|toggle|examine)\s+(.+?)$",
    re.IGNORECASE,
)


def classify_alfworld_transitions(
    transitions: Sequence[StepTransition],
) -> tuple[AlfworldEventType, ...]:
    return tuple(PRIMITIVE_EVENTS[transition.primitive_action] for transition in transitions)


def segment_alfworld_trajectory(
    trajectory: EpisodeTrajectory,
    transitions: Sequence[StepTransition],
    embeddings: Sequence[Sequence[float]] | None = None,
) -> tuple[SegmentInstance, ...]:
    if len(transitions) != len(trajectory.steps):
        raise ValueError("trajectory and transitions must align")
    if embeddings is not None and len(embeddings) != len(transitions):
        raise ValueError("trajectory and embeddings must align")

    events = classify_alfworld_transitions(transitions)
    if not events:
        return ()

    boundaries = []
    start = 0
    for index in range(1, len(events) + 1):
        if index == len(events) or events[index] is not events[start]:
            boundaries.append((start, index - 1, events[start]))
            start = index

    segments = []
    for start, end, event in boundaries:
        if embeddings is None:
            embedding = ()
        else:
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
                event_type=event,
                raw_actions=tuple(
                    transition.raw_action for transition in transitions[start : end + 1]
                ),
                observation_before=transitions[start].observation_before,
                observation_after=transitions[end].observation_after,
            )
        )
    return tuple(segments)


def alfworld_goal_from_observation(observation: str) -> str:
    match = _OBSERVED_GOAL_RE.search(observation)
    return _normalize_text(match.group(1)) if match else ""


def alfworld_segment_signature(
    segment: SegmentInstance,
    *,
    goal: str = "",
    previous_event: AlfworldEventType | None = None,
    next_event: AlfworldEventType | None = None,
) -> str:
    if not isinstance(segment.event_type, AlfworldEventType):
        raise ValueError("ALFWorld segment signature requires an ALFWorld event type")

    actions = [_normalized_action_signature(action) for action in segment.raw_actions]
    return "\n".join(
        (
            f"Task: {_normalize_text(goal) or 'unknown'}",
            "Event context: "
            f"{previous_event.value if previous_event else 'START'} -> "
            f"{segment.event_type.value} -> "
            f"{next_event.value if next_event else 'END'}",
            f"Event: {segment.event_type.value}",
            f"Length: {segment.end_step - segment.start_step + 1}",
            f"Actions: {' -> '.join(actions)}",
        )
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def _normalize_entity(value: str) -> str:
    return _INSTANCE_RE.sub("", _normalize_text(value))


def _normalized_action_signature(raw_action: str) -> str:
    action = _normalize_text(raw_action)
    if action in {"inventory", "look", "help"}:
        return action.upper()
    match = _FROM_RE.match(action)
    if match:
        return f"PICK_UP({_normalize_entity(match.group(1))}, {_normalize_entity(match.group(2))})"
    match = _TO_RE.match(action)
    if match:
        return f"PUT({_normalize_entity(match.group(1))}, {_normalize_entity(match.group(2))})"
    match = _WITH_RE.match(action)
    if match:
        return (
            f"{match.group(1).upper()}({_normalize_entity(match.group(2))}, "
            f"{_normalize_entity(match.group(3))})"
        )
    match = _UNARY_RE.match(action)
    if match:
        operation = match.group(1).replace(" ", "_").upper()
        return f"{operation}({_normalize_entity(match.group(2))})"
    return re.sub(r"\s+\d+\b", "", action).upper()


def alfworld_applicable_events(
    admissible_actions: Sequence[str],
) -> frozenset[AlfworldEventType]:
    events = {
        PRIMITIVE_EVENTS[primitive]
        for action in admissible_actions
        if (primitive := parse_alfworld_action("take_action", {"action": action}))
        is not PrimitiveAction.INVALID
    }
    if not events:
        raise ValueError("ALFWorld graph retrieval requires admissible primitive actions")
    return frozenset(events)


def alfworld_goal_events(task_goal: str) -> frozenset[AlfworldEventType]:
    goal = task_goal.casefold()
    return frozenset(
        event
        for event, phrases in ALFWORLD_GOAL_EVENT_PHRASES.items()
        if any(phrase in goal for phrase in phrases)
    )
