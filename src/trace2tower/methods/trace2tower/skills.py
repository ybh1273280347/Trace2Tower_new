from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.models import (
    MidCluster,
    PrimitiveAction,
    SegmentInstance,
    StepTransition,
)


@dataclass(frozen=True, slots=True)
class LowSkill:
    primitive_action: PrimitiveAction
    action_template: str


LOW_SKILLS: dict[Benchmark, tuple[LowSkill, ...]] = {
    Benchmark.ALFWORLD: (
        LowSkill(PrimitiveAction.GOTO, "go to {receptacle}"),
        LowSkill(PrimitiveAction.PICK, "take {object} from {receptacle}"),
        LowSkill(PrimitiveAction.PUT, "put {object} in/on {receptacle}"),
        LowSkill(PrimitiveAction.OPEN, "open {receptacle}"),
        LowSkill(PrimitiveAction.CLOSE, "close {receptacle}"),
        LowSkill(PrimitiveAction.TOGGLE, "use {object}"),
        LowSkill(PrimitiveAction.HEAT, "heat {object} with {appliance}"),
        LowSkill(PrimitiveAction.CLEAN, "clean {object} with {receptacle}"),
        LowSkill(PrimitiveAction.COOL, "cool {object} with {receptacle}"),
        LowSkill(PrimitiveAction.SLICE, "slice {object} with {tool}"),
        LowSkill(PrimitiveAction.INVENTORY, "inventory"),
        LowSkill(PrimitiveAction.EXAMINE, "examine {object}"),
        LowSkill(PrimitiveAction.LOOK, "look"),
    ),
    Benchmark.WEBSHOP: (
        LowSkill(PrimitiveAction.SEARCH, "search_action(keywords)"),
        LowSkill(PrimitiveAction.CLICK, "click_action(value)"),
    ),
}


@dataclass(frozen=True, slots=True)
class SegmentEvidence:
    segment_id: str
    trajectory_id: str
    goal: str
    raw_actions: tuple[str, ...]
    primitive_actions: tuple[PrimitiveAction, ...]
    observation_before: str
    observation_after: str
    trajectory_score: float
    event_type: str | None

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MidRenderInput:
    cluster_id: str
    member_segment_ids: tuple[str, ...]
    segment_evidence: tuple[SegmentEvidence, ...]
    support_count: int
    primitive_action_distribution: dict[str, int]

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MidSkillCard:
    skill_id: str
    member_segment_ids: tuple[str, ...]
    name: str
    description: str
    procedure: tuple[str, ...]
    constraints: tuple[str, ...]
    grounding_actions: tuple[PrimitiveAction, ...]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> MidSkillCard:
        return cls(
            skill_id=str(record["skill_id"]),
            member_segment_ids=tuple(record["member_segment_ids"]),
            name=str(record["name"]),
            description=str(record["description"]),
            procedure=tuple(record["procedure"]),
            constraints=tuple(record["constraints"]),
            grounding_actions=tuple(
                PrimitiveAction(value) for value in record["grounding_actions"]
            ),
        )


@dataclass(frozen=True, slots=True)
class HighSkillCard:
    skill_id: str
    ordered_mid_ids: tuple[str, ...]
    name: str
    description: str
    procedure: tuple[str, ...]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> HighSkillCard:
        return cls(
            skill_id=str(record["skill_id"]),
            ordered_mid_ids=tuple(record["ordered_mid_ids"]),
            name=str(record["name"]),
            description=str(record["description"]),
            procedure=tuple(record["procedure"]),
        )


def build_mid_render_inputs(
    records: Sequence[Mapping], clusters: Iterable[MidCluster]
) -> tuple[MidRenderInput, ...]:
    segments: dict[str, SegmentInstance] = {}
    transitions: dict[str, StepTransition] = {}
    goals: dict[str, str] = {}
    for record in records:
        trajectory_transitions = tuple(
            StepTransition.from_record(item) for item in record["transitions"]
        )
        for transition in trajectory_transitions:
            transitions[transition.transition_id] = transition
            goals[transition.trajectory_id] = transition.goal
        for item in record["segments"]:
            segment = SegmentInstance.from_record(item)
            if segment.segment_id in segments:
                raise ValueError(f"duplicate segment ID: {segment.segment_id}")
            segments[segment.segment_id] = segment

    inputs = []
    assigned_segment_ids = set()
    for cluster in sorted(clusters, key=lambda item: item.cluster_id):
        evidence = []
        action_counts: Counter[str] = Counter()
        for segment_id in cluster.member_segment_ids:
            if segment_id in assigned_segment_ids:
                raise ValueError(f"segment belongs to multiple Mid clusters: {segment_id}")
            if segment_id not in segments:
                raise ValueError(f"cluster references unknown segment: {segment_id}")
            segment = segments[segment_id]
            segment_transitions = tuple(
                transitions[transition_id] for transition_id in segment.transition_ids
            )
            primitive_actions = tuple(
                transition.primitive_action
                for transition in segment_transitions
                if transition.primitive_action is not PrimitiveAction.INVALID
            )
            action_counts.update(action.value for action in primitive_actions)
            evidence.append(
                SegmentEvidence(
                    segment_id=segment.segment_id,
                    trajectory_id=segment.trajectory_id,
                    goal=goals[segment.trajectory_id],
                    raw_actions=segment.raw_actions,
                    primitive_actions=primitive_actions,
                    observation_before=segment.observation_before,
                    observation_after=segment.observation_after,
                    trajectory_score=segment.trajectory_score,
                    event_type=segment.event_type.value if segment.event_type else None,
                )
            )
            assigned_segment_ids.add(segment_id)
        inputs.append(
            MidRenderInput(
                cluster_id=cluster.cluster_id,
                member_segment_ids=cluster.member_segment_ids,
                segment_evidence=tuple(evidence),
                support_count=len(cluster.member_segment_ids),
                primitive_action_distribution=dict(sorted(action_counts.items())),
            )
        )
    if assigned_segment_ids != set(segments):
        raise ValueError("Mid clusters must partition every preprocessed segment")
    return tuple(inputs)


def legal_grounding_actions(
    benchmark: Benchmark, render_input: MidRenderInput
) -> tuple[PrimitiveAction, ...]:
    official_actions = {skill.primitive_action for skill in LOW_SKILLS[benchmark]}
    return tuple(
        action
        for action in PrimitiveAction
        if action in official_actions
        and render_input.primitive_action_distribution.get(action.value, 0) > 0
    )
