from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RefinementAction(StrEnum):
    SPLIT = "split"
    MERGE = "merge"
    PROMOTE = "promote"
    DOWNWEIGHT = "downweight"


class LineageKind(StrEnum):
    CONTINUATION = "continuation"
    SPLIT = "split"
    MERGE = "merge"
    RECOMPOSED = "recomposed"
    NEW_MID = "new_mid"
    DISAPPEARED_MID = "disappeared_mid"


@dataclass(frozen=True, slots=True)
class DeploymentObjectives:
    performance_level: float
    paired_success_gain: float
    guarded_step_saving: float

    @property
    def values(self) -> tuple[float, float, float]:
        return (
            self.performance_level,
            self.paired_success_gain,
            self.guarded_step_saving,
        )


@dataclass(frozen=True, slots=True)
class BundleMetrics:
    primary_high_id: str
    exposure_count: int
    objectives: DeploymentObjectives


@dataclass(frozen=True, slots=True)
class BundleParetoEstimate:
    metrics: BundleMetrics
    pareto_front_rank: int
    front_1_probability: float
    dominated_probability: float


@dataclass(frozen=True, slots=True)
class LineageOverlap:
    old_mid_id: str
    new_mid_id: str
    shared_member_count: int
    old_retention: float
    new_historical_purity: float


@dataclass(frozen=True, slots=True)
class LineageComponent:
    component_id: str
    kind: LineageKind
    old_mid_ids: tuple[str, ...]
    new_mid_ids: tuple[str, ...]
    overlaps: tuple[LineageOverlap, ...]
