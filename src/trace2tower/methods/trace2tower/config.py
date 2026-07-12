from __future__ import annotations

from dataclasses import dataclass

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

    def __post_init__(self) -> None:
        if self.failure_penalty < 0:
            raise ValueError("failure penalty must be non-negative")
        if not 1 <= self.min_mid_clusters <= self.max_mid_clusters:
            raise ValueError("invalid Mid cluster range")
        if self.semantic_only and self.method is not MethodName.SEMANTIC_CLUSTERING:
            raise ValueError("semantic_only requires the semantic clustering method")

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
        )
