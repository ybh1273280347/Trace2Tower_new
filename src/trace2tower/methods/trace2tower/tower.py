from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.models import HighCommunity, HighPath, MidCluster
from trace2tower.semantic_index import SkillEmbeddingIndex
from trace2tower.methods.trace2tower.skills import (
    HighSkillCard,
    LowSkill,
    MidSkillCard,
)


class TowerVersion(StrEnum):
    V0 = "v0"
    V1 = "v1"
    V2 = "v2"


@dataclass(frozen=True, slots=True)
class TowerSourceHashes:
    preprocessed_trajectories: str
    clusters: str
    high_paths: str
    rendered_cards: str
    retrieval_index: str

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError("tower source hashes must be lowercase SHA-256 values")

    @classmethod
    def from_record(cls, record: Mapping) -> TowerSourceHashes:
        return cls(**{field: str(record[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class TowerSnapshot:
    snapshot_id: str
    version: TowerVersion
    benchmark: Benchmark
    config: Trace2TowerConfig
    training_trajectory_ids: tuple[str, ...]
    source_hashes: TowerSourceHashes
    low_skills: tuple[LowSkill, ...]
    mid_clusters: tuple[MidCluster, ...]
    high_paths: tuple[HighPath, ...]
    mid_cards: tuple[MidSkillCard, ...]
    high_cards: tuple[HighSkillCard, ...]
    mid_index: SkillEmbeddingIndex
    high_index: SkillEmbeddingIndex
    mid_coverage_complete: bool
    high_coverage_complete: bool
    high_communities: tuple[HighCommunity, ...] = ()

    def __post_init__(self) -> None:
        if not self.training_trajectory_ids or len(set(self.training_trajectory_ids)) != len(
            self.training_trajectory_ids
        ):
            raise ValueError("tower requires unique training trajectory provenance")
        training_ids = set(self.training_trajectory_ids)
        clusters = {cluster.cluster_id: cluster for cluster in self.mid_clusters}
        paths = {path.path_id: path for path in self.high_paths}
        communities = {
            community.community_id: community for community in self.high_communities
        }
        mid_cards = {card.skill_id: card for card in self.mid_cards}
        high_cards = {card.skill_id: card for card in self.high_cards}
        if any(
            len(items) != len(mapping)
            for items, mapping in (
                (self.mid_clusters, clusters),
                (self.high_paths, paths),
                (self.high_communities, communities),
                (self.mid_cards, mid_cards),
                (self.high_cards, high_cards),
            )
        ):
            raise ValueError("tower structure contains duplicate stable IDs")
        if set(mid_cards) - set(clusters):
            raise ValueError("tower cards reference unknown structure")
        if self.version is TowerVersion.V2:
            if not communities or set(high_cards) - set(communities):
                raise ValueError("Tower v2 High cards require known High communities")
        elif communities or set(high_cards) - set(paths):
            raise ValueError("legacy Tower High cards require known High paths")
        if any(
            segment_id.rsplit(":segment:", 1)[0] not in training_ids
            for cluster in self.mid_clusters
            for segment_id in cluster.member_segment_ids
        ):
            raise ValueError("Mid clusters contain segments outside training provenance")
        if any(not set(path.supporting_trajectory_ids) <= training_ids for path in self.high_paths):
            raise ValueError("High paths contain support outside training provenance")
        if any(
            not set(community.member_mid_ids) <= set(clusters)
            or not set(community.member_path_ids) <= set(paths)
            or not set(community.supporting_trajectory_ids) <= training_ids
            for community in self.high_communities
        ):
            raise ValueError("High communities reference unknown Tower evidence")
        if any(
            len(path.ordered_mid_ids) < 2
            or len(path.ordered_mid_ids) > self.config.max_high_path_length
            or len(set(path.ordered_mid_ids)) < 2
            for path in self.high_paths
        ):
            raise ValueError("High path violates the configured structural contract")
        for skill_id, card in mid_cards.items():
            if card.member_segment_ids != clusters[skill_id].member_segment_ids:
                raise ValueError(f"Mid card membership differs from cluster: {skill_id}")
        for skill_id, card in high_cards.items():
            if self.version is TowerVersion.V2:
                if card.member_mid_ids != communities[skill_id].member_mid_ids:
                    raise ValueError(f"High card membership differs from community: {skill_id}")
                if card.ordered_mid_ids and not any(
                    paths[path_id].ordered_mid_ids == card.ordered_mid_ids
                    for path_id in communities[skill_id].member_path_ids
                ):
                    raise ValueError(
                        f"High card order is not supported by its community: {skill_id}"
                    )
            elif card.ordered_mid_ids != paths[skill_id].ordered_mid_ids:
                raise ValueError(f"High card order differs from path: {skill_id}")
            if not set(card.child_mid_ids) <= set(mid_cards):
                raise ValueError(f"High card has missing child Mid cards: {skill_id}")
        if set(self.mid_index.skill_ids) != set(mid_cards):
            raise ValueError("Mid index and cards differ")
        if set(self.high_index.skill_ids) != set(high_cards):
            raise ValueError("High index and cards differ")
        if not self.mid_index.text_hashes or (self.high_cards and not self.high_index.text_hashes):
            raise ValueError("formal tower indexes require card text hashes")
        expected_mid_coverage = set(mid_cards) == set(clusters)
        expected_high_coverage = set(high_cards) == (
            set(communities) if self.version is TowerVersion.V2 else set(paths)
        )
        if self.mid_coverage_complete != expected_mid_coverage:
            raise ValueError("Mid coverage flag disagrees with snapshot contents")
        if self.high_coverage_complete != expected_high_coverage:
            raise ValueError("High coverage flag disagrees with snapshot contents")
        if self.snapshot_id:
            expected_id = f"tower_{_snapshot_digest(self.content_record())[:16]}"
            if self.snapshot_id != expected_id:
                raise ValueError("tower snapshot ID does not match its contents")

    @property
    def is_complete(self) -> bool:
        return self.mid_coverage_complete and self.high_coverage_complete

    def require_complete(self) -> None:
        if not self.is_complete:
            raise ValueError("formal execution requires complete Mid and High coverage")

    def content_record(self) -> dict:
        record = {
            "version": self.version,
            "benchmark": self.benchmark,
            "config": self.config.to_record(),
            "training_trajectory_ids": self.training_trajectory_ids,
            "source_hashes": asdict(self.source_hashes),
            "low_skills": [skill.to_record() for skill in self.low_skills],
            "mid_clusters": [cluster.to_record() for cluster in self.mid_clusters],
            "high_paths": [path.to_record() for path in self.high_paths],
            "mid_cards": [card.to_record() for card in self.mid_cards],
            "high_cards": [card.to_record() for card in self.high_cards],
            "mid_index": self.mid_index.to_record(),
            "high_index": self.high_index.to_record(),
            "mid_coverage_complete": self.mid_coverage_complete,
            "high_coverage_complete": self.high_coverage_complete,
        }
        if self.version is TowerVersion.V2:
            record["high_communities"] = [
                community.to_record() for community in self.high_communities
            ]
        return record

    def to_record(self) -> dict:
        return {"snapshot_id": self.snapshot_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: Mapping) -> TowerSnapshot:
        return cls(
            snapshot_id=str(record["snapshot_id"]),
            version=TowerVersion(record["version"]),
            benchmark=Benchmark(record["benchmark"]),
            config=Trace2TowerConfig.from_record(record["config"]),
            training_trajectory_ids=tuple(record["training_trajectory_ids"]),
            source_hashes=TowerSourceHashes.from_record(record["source_hashes"]),
            low_skills=tuple(LowSkill.from_record(item) for item in record["low_skills"]),
            mid_clusters=tuple(MidCluster.from_record(item) for item in record["mid_clusters"]),
            high_paths=tuple(HighPath.from_record(item) for item in record["high_paths"]),
            mid_cards=tuple(MidSkillCard.from_record(item) for item in record["mid_cards"]),
            high_cards=tuple(HighSkillCard.from_record(item) for item in record["high_cards"]),
            mid_index=SkillEmbeddingIndex.from_record(record["mid_index"]),
            high_index=SkillEmbeddingIndex.from_record(record["high_index"]),
            mid_coverage_complete=bool(record["mid_coverage_complete"]),
            high_coverage_complete=bool(record["high_coverage_complete"]),
            high_communities=tuple(
                HighCommunity.from_record(item)
                for item in record.get("high_communities", ())
            ),
        )


def build_tower_snapshot(
    *,
    version: TowerVersion,
    benchmark: Benchmark,
    config: Trace2TowerConfig,
    training_trajectory_ids: tuple[str, ...],
    source_hashes: TowerSourceHashes,
    low_skills: tuple[LowSkill, ...],
    mid_clusters: tuple[MidCluster, ...],
    high_paths: tuple[HighPath, ...],
    mid_cards: tuple[MidSkillCard, ...],
    high_cards: tuple[HighSkillCard, ...],
    mid_index: SkillEmbeddingIndex,
    high_index: SkillEmbeddingIndex,
    high_communities: tuple[HighCommunity, ...] = (),
) -> TowerSnapshot:
    snapshot = TowerSnapshot(
        snapshot_id="",
        version=version,
        benchmark=benchmark,
        config=config,
        training_trajectory_ids=tuple(sorted(training_trajectory_ids)),
        source_hashes=source_hashes,
        low_skills=tuple(sorted(low_skills, key=lambda item: item.primitive_action)),
        mid_clusters=tuple(sorted(mid_clusters, key=lambda item: item.cluster_id)),
        high_paths=tuple(sorted(high_paths, key=lambda item: item.path_id)),
        mid_cards=tuple(sorted(mid_cards, key=lambda item: item.skill_id)),
        high_cards=tuple(sorted(high_cards, key=lambda item: item.skill_id)),
        mid_index=mid_index,
        high_index=high_index,
        mid_coverage_complete={card.skill_id for card in mid_cards}
        == {cluster.cluster_id for cluster in mid_clusters},
        high_coverage_complete={card.skill_id for card in high_cards}
        == (
            {community.community_id for community in high_communities}
            if version is TowerVersion.V2
            else {path.path_id for path in high_paths}
        ),
        high_communities=tuple(
            sorted(high_communities, key=lambda item: item.community_id)
        ),
    )
    return replace(
        snapshot,
        snapshot_id=f"tower_{_snapshot_digest(snapshot.content_record())[:16]}",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_digest(record: dict) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
