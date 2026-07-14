from __future__ import annotations

from dataclasses import asdict, dataclass

from trace2tower.manifests import Benchmark
from trace2tower.results import MethodName


@dataclass(frozen=True, slots=True)
class Trace2TowerConfig:
    method: MethodName
    semantic_only: bool
    use_transition_edge: bool
    use_outcome_edge: bool
    use_contrastive_decomposition: bool
    failure_penalty: float
    min_mid_clusters: int
    max_mid_clusters: int
    random_state: int
    max_high_path_length: int = 4
    high_min_support_ratio: float = 0.02
    high_path_epsilon: float = 1e-6
    event_type_stratification: bool = False

    def __post_init__(self) -> None:
        if self.failure_penalty < 0:
            raise ValueError("failure penalty must be non-negative")
        if not 1 <= self.min_mid_clusters <= self.max_mid_clusters:
            raise ValueError("invalid Mid cluster range")
        if self.semantic_only != (
            self.method is MethodName.TRACE2TOWER_SEMANTIC_ONLY
        ):
            raise ValueError("semantic-only switch and method must agree")
        if self.semantic_only and (
            self.use_transition_edge
            or self.use_outcome_edge
            or self.use_contrastive_decomposition
            or self.event_type_stratification
        ):
            raise ValueError("semantic-only ablation cannot use graph or event structure")
        if self.max_high_path_length < 2:
            raise ValueError("max High path length must be at least two")
        if not 0 <= self.high_min_support_ratio <= 1:
            raise ValueError("High path support ratio must be in [0, 1]")
        if self.high_path_epsilon <= 0:
            raise ValueError("High path epsilon must be positive")

    def validate_for_benchmark(self, benchmark: Benchmark) -> None:
        if benchmark is not Benchmark.WEBSHOP:
            return
        event_aware_methods = {
            MethodName.TRACE2TOWER,
            MethodName.TRACE2TOWER_MID_ONLY,
            MethodName.TRACE2TOWER_NO_MIXED,
        }
        if self.method in event_aware_methods and not self.event_type_stratification:
            raise ValueError(
                "formal WebShop Trace2Tower requires event-type stratification"
            )

    def to_record(self) -> dict:
        record = asdict(self)
        if not self.event_type_stratification:
            record.pop("event_type_stratification")
        return record

    @classmethod
    def from_record(cls, record: dict) -> Trace2TowerConfig:
        boolean_fields = (
            "semantic_only",
            "use_transition_edge",
            "use_outcome_edge",
            "use_contrastive_decomposition",
        )
        if any(not isinstance(record[field], bool) for field in boolean_fields):
            raise ValueError("Trace2Tower switches must be booleans")
        if "event_type_stratification" in record and not isinstance(
            record["event_type_stratification"], bool
        ):
            raise ValueError("event-type stratification switch must be boolean")
        return cls(
            method=MethodName(record["method"]),
            semantic_only=record["semantic_only"],
            use_transition_edge=record["use_transition_edge"],
            use_outcome_edge=record["use_outcome_edge"],
            use_contrastive_decomposition=record["use_contrastive_decomposition"],
            failure_penalty=float(record["failure_penalty"]),
            min_mid_clusters=int(record["min_mid_clusters"]),
            max_mid_clusters=int(record["max_mid_clusters"]),
            random_state=int(record["random_state"]),
            max_high_path_length=int(record.get("max_high_path_length", 4)),
            high_min_support_ratio=float(record.get("high_min_support_ratio", 0.02)),
            high_path_epsilon=float(record.get("high_path_epsilon", 1e-6)),
            event_type_stratification=record.get(
                "event_type_stratification", False
            ),
        )
