from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class PrimitiveAction(StrEnum):
    GOTO = "GOTO"
    PICK = "PICK"
    PUT = "PUT"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    TOGGLE = "TOGGLE"
    HEAT = "HEAT"
    CLEAN = "CLEAN"
    COOL = "COOL"
    SLICE = "SLICE"
    INVENTORY = "INVENTORY"
    EXAMINE = "EXAMINE"
    LOOK = "LOOK"
    SEARCH = "SEARCH"
    CLICK = "CLICK"
    INVALID = "INVALID"


class AlfworldEventType(StrEnum):
    GOTO_LOCATION = "GotoLocation"
    PICKUP_OBJECT = "PickupObject"
    PUT_OBJECT = "PutObject"
    OPEN_OBJECT = "OpenObject"
    CLOSE_OBJECT = "CloseObject"
    TOGGLE_OBJECT = "ToggleObject"
    SLICE_OBJECT = "SliceObject"
    CLEAN_OBJECT = "CleanObject"
    HEAT_OBJECT = "HeatObject"
    COOL_OBJECT = "CoolObject"
    SCAN = "Scan"
    INVALID_ACTION = "InvalidAction"


class WebShopPageType(StrEnum):
    SEARCH = "SEARCH"
    RESULTS = "RESULTS"
    ITEM = "ITEM"
    ITEM_DETAIL = "ITEM_DETAIL"
    TERMINAL = "TERMINAL"
    UNKNOWN = "UNKNOWN"


class WebShopEventType(StrEnum):
    QUERY_FORMULATION = "QUERY_FORMULATION"
    QUERY_REFINEMENT = "QUERY_REFINEMENT"
    RESULT_NAVIGATION = "RESULT_NAVIGATION"
    CANDIDATE_SELECTION = "CANDIDATE_SELECTION"
    OPTION_SELECTION = "OPTION_SELECTION"
    ATTRIBUTE_INSPECTION = "ATTRIBUTE_INSPECTION"
    DETAIL_BACKTRACKING = "DETAIL_BACKTRACKING"
    CANDIDATE_BACKTRACKING = "CANDIDATE_BACKTRACKING"
    SEARCH_BACKTRACKING = "SEARCH_BACKTRACKING"
    PURCHASE_DECISION = "PURCHASE_DECISION"
    OTHER_CLICK = "OTHER_CLICK"


EventType = AlfworldEventType | WebShopEventType


def event_type_from_value(value: str) -> EventType:
    try:
        return AlfworldEventType(value)
    except ValueError:
        return WebShopEventType(value)


@dataclass(frozen=True, slots=True)
class StepTransition:
    transition_id: str
    trajectory_id: str
    step_index: int
    goal: str
    observation_before: str
    raw_action: str
    primitive_action: PrimitiveAction
    observation_after: str
    trajectory_score: float

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> StepTransition:
        return cls(
            transition_id=str(record["transition_id"]),
            trajectory_id=str(record["trajectory_id"]),
            step_index=int(record["step_index"]),
            goal=str(record["goal"]),
            observation_before=str(record["observation_before"]),
            raw_action=str(record["raw_action"]),
            primitive_action=PrimitiveAction(record["primitive_action"]),
            observation_after=str(record["observation_after"]),
            trajectory_score=float(record["trajectory_score"]),
        )


@dataclass(frozen=True, slots=True)
class SegmentInstance:
    segment_id: str
    trajectory_id: str
    start_step: int
    end_step: int
    transition_ids: tuple[str, ...]
    embedding: tuple[float, ...]
    trajectory_score: float
    event_type: EventType | None
    raw_actions: tuple[str, ...]
    observation_before: str
    observation_after: str

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> SegmentInstance:
        event_type = record["event_type"]
        return cls(
            segment_id=str(record["segment_id"]),
            trajectory_id=str(record["trajectory_id"]),
            start_step=int(record["start_step"]),
            end_step=int(record["end_step"]),
            transition_ids=tuple(record["transition_ids"]),
            embedding=tuple(float(value) for value in record["embedding"]),
            trajectory_score=float(record["trajectory_score"]),
            event_type=event_type_from_value(event_type) if event_type is not None else None,
            raw_actions=tuple(record["raw_actions"]),
            observation_before=str(record["observation_before"]),
            observation_after=str(record["observation_after"]),
        )


@dataclass(frozen=True, slots=True)
class SegmentationCalibration:
    penalty: float
    target_segment_length: int
    median_segment_length: float
    trajectory_count: int
    segment_count: int


@dataclass(frozen=True, slots=True)
class MidCluster:
    cluster_id: str
    member_segment_ids: tuple[str, ...]
    centroid: tuple[float, ...]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> MidCluster:
        return cls(
            cluster_id=str(record["cluster_id"]),
            member_segment_ids=tuple(record["member_segment_ids"]),
            centroid=tuple(float(value) for value in record["centroid"]),
        )


@dataclass(frozen=True, slots=True)
class HighPath:
    path_id: str
    ordered_mid_ids: tuple[str, ...]
    positive_support: float
    negative_support: float
    contrastive_score: float
    supporting_trajectory_ids: tuple[str, ...]
    task_condition: str = ""

    def to_record(self) -> dict:
        record = asdict(self)
        if not self.task_condition:
            record.pop("task_condition")
        return record

    @classmethod
    def from_record(cls, record: dict) -> HighPath:
        return cls(
            path_id=str(record["path_id"]),
            ordered_mid_ids=tuple(record["ordered_mid_ids"]),
            positive_support=float(record["positive_support"]),
            negative_support=float(record["negative_support"]),
            contrastive_score=float(record["contrastive_score"]),
            supporting_trajectory_ids=tuple(record["supporting_trajectory_ids"]),
            task_condition=str(record.get("task_condition", "")),
        )


@dataclass(frozen=True, slots=True)
class HighCommunity:
    community_id: str
    member_mid_ids: tuple[str, ...]
    member_path_ids: tuple[str, ...]
    supporting_trajectory_ids: tuple[str, ...]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> HighCommunity:
        return cls(
            community_id=str(record["community_id"]),
            member_mid_ids=tuple(record["member_mid_ids"]),
            member_path_ids=tuple(record["member_path_ids"]),
            supporting_trajectory_ids=tuple(record["supporting_trajectory_ids"]),
        )
