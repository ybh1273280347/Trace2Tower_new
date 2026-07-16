from __future__ import annotations

from dataclasses import asdict, dataclass

from trace2tower.results import MethodName
from trace2tower.methods.trace2tower.models import (
    GraphOutcomeMode,
    HighPathDiscovery,
    SemanticNeighborScope,
)


@dataclass(frozen=True, slots=True)
class Trace2TowerConfig:
    method: MethodName
    semantic_only: bool
    use_transition_edge: bool
    use_outcome_edge: bool
    use_contrastive_decomposition: bool
    failure_penalty: float
    min_mid_clusters: int
    max_mid_clusters: int | None
    random_state: int
    max_high_path_length: int = 4
    high_min_support_ratio: float = 0.02
    high_path_epsilon: float = 1e-6
    success_threshold: float = 0.999
    collapse_duplicate_embeddings: bool = False
    high_min_success_count: int = 0
    high_path_discovery: HighPathDiscovery = HighPathDiscovery.CONTRASTIVE_SUBSEQUENCE
    semantic_neighbor_scope: SemanticNeighborScope = SemanticNeighborScope.GLOBAL
    graph_outcome_mode: GraphOutcomeMode = GraphOutcomeMode.BINARY_CONTRASTIVE
    continuous_residual_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.failure_penalty < 0:
            raise ValueError("failure penalty must be non-negative")
        if self.min_mid_clusters < 1 or (
            self.max_mid_clusters is not None
            and self.min_mid_clusters > self.max_mid_clusters
        ):
            raise ValueError("invalid Mid cluster range")
        if self.semantic_only != (self.method is MethodName.SEMANTIC_CLUSTERING):
            raise ValueError("semantic-only switch and method must agree")
        if self.semantic_only and (
            self.use_transition_edge or self.use_outcome_edge or self.use_contrastive_decomposition
        ):
            raise ValueError("semantic clustering cannot use graph structure")
        if self.max_high_path_length < 2:
            raise ValueError("max High path length must be at least two")
        if not 0 <= self.high_min_support_ratio <= 1:
            raise ValueError("High path support ratio must be in [0, 1]")
        if self.high_min_success_count < 0:
            raise ValueError("High path success count must be non-negative")
        if self.high_path_epsilon <= 0:
            raise ValueError("High path epsilon must be positive")
        if not 0 < self.success_threshold <= 1:
            raise ValueError("success threshold must be in (0, 1]")
        if not 0 <= self.continuous_residual_weight <= 1:
            raise ValueError("continuous residual weight must be in [0, 1]")

    def to_record(self) -> dict:
        record = asdict(self)
        if not self.collapse_duplicate_embeddings:
            record.pop("collapse_duplicate_embeddings")
        if not self.high_min_success_count:
            record.pop("high_min_success_count")
        if self.high_path_discovery is HighPathDiscovery.CONTRASTIVE_SUBSEQUENCE:
            record.pop("high_path_discovery")
        if self.semantic_neighbor_scope is SemanticNeighborScope.GLOBAL:
            record.pop("semantic_neighbor_scope")
        if self.graph_outcome_mode is GraphOutcomeMode.BINARY_CONTRASTIVE:
            record.pop("graph_outcome_mode")
        if self.graph_outcome_mode is not GraphOutcomeMode.CONTINUOUS_RESIDUAL:
            record.pop("continuous_residual_weight")
        if self.max_mid_clusters is None:
            record.pop("max_mid_clusters")
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
        if "collapse_duplicate_embeddings" in record and not isinstance(
            record["collapse_duplicate_embeddings"], bool
        ):
            raise ValueError("duplicate embedding collapse switch must be boolean")
        if "event_type_stratification" in record:
            raise ValueError("event-type stratification is not part of the Trace2Tower algorithm")
        return cls(
            method=MethodName(record["method"]),
            semantic_only=record["semantic_only"],
            use_transition_edge=record["use_transition_edge"],
            use_outcome_edge=record["use_outcome_edge"],
            use_contrastive_decomposition=record["use_contrastive_decomposition"],
            failure_penalty=float(record["failure_penalty"]),
            min_mid_clusters=int(record["min_mid_clusters"]),
            max_mid_clusters=(
                int(record["max_mid_clusters"])
                if record.get("max_mid_clusters") is not None
                else None
            ),
            random_state=int(record["random_state"]),
            max_high_path_length=int(record.get("max_high_path_length", 4)),
            high_min_support_ratio=float(record.get("high_min_support_ratio", 0.02)),
            high_path_epsilon=float(record.get("high_path_epsilon", 1e-6)),
            success_threshold=float(record.get("success_threshold", 0.999)),
            collapse_duplicate_embeddings=bool(
                record.get("collapse_duplicate_embeddings", False)
            ),
            high_min_success_count=int(record.get("high_min_success_count", 0)),
            high_path_discovery=HighPathDiscovery(
                record.get(
                    "high_path_discovery",
                    HighPathDiscovery.CONTRASTIVE_SUBSEQUENCE,
                )
            ),
            semantic_neighbor_scope=SemanticNeighborScope(
                record.get("semantic_neighbor_scope", SemanticNeighborScope.GLOBAL)
            ),
            graph_outcome_mode=GraphOutcomeMode(
                record.get("graph_outcome_mode", GraphOutcomeMode.BINARY_CONTRASTIVE)
            ),
            continuous_residual_weight=float(record.get("continuous_residual_weight", 0.2)),
        )
