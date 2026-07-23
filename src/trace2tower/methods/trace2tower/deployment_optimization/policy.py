from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DeploymentPolicy:
    policy_id: str
    snapshot_id: str
    high_score_penalties: tuple[tuple[str, float], ...]
    source_report_sha256: str

    def __post_init__(self) -> None:
        if not self.snapshot_id.startswith("tower_"):
            raise ValueError("deployment policy requires a Tower snapshot ID")
        high_ids = tuple(skill_id for skill_id, _ in self.high_score_penalties)
        if len(high_ids) != len(set(high_ids)) or any(
            not skill_id.startswith("high_") or penalty <= 0
            for skill_id, penalty in self.high_score_penalties
        ):
            raise ValueError("deployment policy contains invalid High penalties")
        if len(self.source_report_sha256) != 64:
            raise ValueError("deployment policy requires a source report SHA-256")
        if self.policy_id and self.policy_id != self.expected_policy_id:
            raise ValueError("deployment policy ID does not match its contents")

    @property
    def expected_policy_id(self) -> str:
        payload = json.dumps(self.content_record(), sort_keys=True, separators=(",", ":"))
        return f"policy_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    @property
    def score_penalties(self) -> dict[str, float]:
        return dict(self.high_score_penalties)

    def content_record(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "high_score_penalties": dict(self.high_score_penalties),
            "source_report_sha256": self.source_report_sha256,
        }

    def to_record(self) -> dict:
        return {"policy_id": self.policy_id or self.expected_policy_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: dict) -> DeploymentPolicy:
        penalties = record["high_score_penalties"]
        if not isinstance(penalties, dict):
            raise ValueError("deployment policy penalties must be an object")
        return cls(
            policy_id=str(record["policy_id"]),
            snapshot_id=str(record["snapshot_id"]),
            high_score_penalties=tuple(
                sorted((str(skill_id), float(penalty)) for skill_id, penalty in penalties.items())
            ),
            source_report_sha256=str(record["source_report_sha256"]),
        )

    @classmethod
    def from_path(cls, path: Path) -> DeploymentPolicy:
        return cls.from_record(json.loads(path.read_text(encoding="utf-8")))


def build_deployment_policy(
    snapshot_id: str,
    high_score_penalties: dict[str, float],
    source_report_sha256: str,
) -> DeploymentPolicy:
    policy = DeploymentPolicy(
        policy_id="",
        snapshot_id=snapshot_id,
        high_score_penalties=tuple(sorted(high_score_penalties.items())),
        source_report_sha256=source_report_sha256,
    )
    return DeploymentPolicy.from_record(policy.to_record())
