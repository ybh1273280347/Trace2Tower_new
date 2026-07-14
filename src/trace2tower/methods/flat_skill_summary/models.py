from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from trace2tower.manifests import Benchmark
from trace2tower.semantic_index import SkillEmbeddingIndex


@dataclass(frozen=True, slots=True)
class FlatSkillCard:
    skill_id: str
    source_trajectory_id: str
    name: str
    description: str
    procedure: tuple[str, ...]
    constraints: tuple[str, ...]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> FlatSkillCard:
        return cls(
            skill_id=str(record["skill_id"]),
            source_trajectory_id=str(record["source_trajectory_id"]),
            name=str(record["name"]),
            description=str(record["description"]),
            procedure=tuple(record["procedure"]),
            constraints=tuple(record["constraints"]),
        )


@dataclass(frozen=True, slots=True)
class CorpusFlatSkillCard:
    skill_id: str
    supporting_trajectory_ids: tuple[str, ...]
    name: str
    description: str
    procedure: tuple[str, ...]
    constraints: tuple[str, ...]

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping) -> CorpusFlatSkillCard:
        return cls(
            skill_id=str(record["skill_id"]),
            supporting_trajectory_ids=tuple(record["supporting_trajectory_ids"]),
            name=str(record["name"]),
            description=str(record["description"]),
            procedure=tuple(record["procedure"]),
            constraints=tuple(record["constraints"]),
        )


FlatCard = FlatSkillCard | CorpusFlatSkillCard


@dataclass(frozen=True, slots=True)
class FlatSkillLibrary:
    library_id: str
    benchmark: Benchmark
    prompt_sha256: str
    training_trajectory_ids: tuple[str, ...]
    cards: tuple[FlatSkillCard, ...]
    index: SkillEmbeddingIndex

    def __post_init__(self) -> None:
        if len(self.prompt_sha256) != 64:
            raise ValueError("Flat prompt hash must be SHA-256")
        source_ids = {card.source_trajectory_id for card in self.cards}
        card_ids = {card.skill_id for card in self.cards}
        if len(source_ids) != len(self.cards) or len(card_ids) != len(self.cards):
            raise ValueError("Flat library contains duplicate card provenance or IDs")
        if source_ids != set(self.training_trajectory_ids):
            raise ValueError("Flat cards must cover every selected successful trajectory")
        if set(self.index.skill_ids) != card_ids or not self.index.text_hashes:
            raise ValueError("Flat index must cover cards with text hashes")
        if self.library_id:
            expected = f"flatlib_{_digest(self.content_record())[:16]}"
            if self.library_id != expected:
                raise ValueError("Flat library ID does not match its contents")

    def content_record(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "prompt_sha256": self.prompt_sha256,
            "training_trajectory_ids": self.training_trajectory_ids,
            "cards": [card.to_record() for card in self.cards],
            "index": self.index.to_record(),
        }

    def to_record(self) -> dict:
        return {"library_id": self.library_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: Mapping) -> FlatSkillLibrary:
        return cls(
            library_id=str(record["library_id"]),
            benchmark=Benchmark(record["benchmark"]),
            prompt_sha256=str(record["prompt_sha256"]),
            training_trajectory_ids=tuple(record["training_trajectory_ids"]),
            cards=tuple(FlatSkillCard.from_record(item) for item in record["cards"]),
            index=SkillEmbeddingIndex.from_record(record["index"]),
        )


@dataclass(frozen=True, slots=True)
class CorpusFlatSkillLibrary:
    library_id: str
    benchmark: Benchmark
    prompt_sha256: str
    corpus_sha256: str
    training_trajectory_ids: tuple[str, ...]
    cards: tuple[CorpusFlatSkillCard, ...]
    index: SkillEmbeddingIndex

    def __post_init__(self) -> None:
        if len(self.prompt_sha256) != 64 or len(self.corpus_sha256) != 64:
            raise ValueError("corpus-induced Flat hashes must be SHA-256")
        training_ids = set(self.training_trajectory_ids)
        card_ids = {card.skill_id for card in self.cards}
        if not training_ids or len(training_ids) != len(self.training_trajectory_ids):
            raise ValueError("corpus-induced Flat requires unique training trajectories")
        if not card_ids or len(card_ids) != len(self.cards):
            raise ValueError("corpus-induced Flat requires unique skill cards")
        if any(
            not card.supporting_trajectory_ids
            or not set(card.supporting_trajectory_ids) <= training_ids
            for card in self.cards
        ):
            raise ValueError("corpus-induced Flat card has invalid evidence provenance")
        if set(self.index.skill_ids) != card_ids or not self.index.text_hashes:
            raise ValueError("corpus-induced Flat index must cover every card")
        if self.library_id:
            expected = f"flatcorpus_{_digest(self.content_record())[:16]}"
            if self.library_id != expected:
                raise ValueError("corpus-induced Flat library ID does not match its contents")

    def content_record(self) -> dict:
        return {
            "library_kind": "corpus_induced",
            "benchmark": self.benchmark,
            "prompt_sha256": self.prompt_sha256,
            "corpus_sha256": self.corpus_sha256,
            "training_trajectory_ids": self.training_trajectory_ids,
            "cards": [card.to_record() for card in self.cards],
            "index": self.index.to_record(),
        }

    def to_record(self) -> dict:
        return {"library_id": self.library_id, **self.content_record()}

    @classmethod
    def from_record(cls, record: Mapping) -> CorpusFlatSkillLibrary:
        if record.get("library_kind") != "corpus_induced":
            raise ValueError("unknown corpus-induced Flat library kind")
        return cls(
            library_id=str(record["library_id"]),
            benchmark=Benchmark(record["benchmark"]),
            prompt_sha256=str(record["prompt_sha256"]),
            corpus_sha256=str(record["corpus_sha256"]),
            training_trajectory_ids=tuple(record["training_trajectory_ids"]),
            cards=tuple(
                CorpusFlatSkillCard.from_record(item) for item in record["cards"]
            ),
            index=SkillEmbeddingIndex.from_record(record["index"]),
        )


FlatLibrary = FlatSkillLibrary | CorpusFlatSkillLibrary


def build_flat_library(
    benchmark: Benchmark,
    prompt_sha256: str,
    cards: tuple[FlatSkillCard, ...],
    index: SkillEmbeddingIndex,
) -> FlatSkillLibrary:
    ordered_cards = tuple(sorted(cards, key=lambda card: card.skill_id))
    library = FlatSkillLibrary(
        library_id="",
        benchmark=benchmark,
        prompt_sha256=prompt_sha256,
        training_trajectory_ids=tuple(
            sorted(card.source_trajectory_id for card in ordered_cards)
        ),
        cards=ordered_cards,
        index=index,
    )
    return replace(
        library,
        library_id=f"flatlib_{_digest(library.content_record())[:16]}",
    )


def build_corpus_flat_library(
    benchmark: Benchmark,
    prompt_sha256: str,
    corpus_sha256: str,
    training_trajectory_ids: tuple[str, ...],
    cards: tuple[CorpusFlatSkillCard, ...],
    index: SkillEmbeddingIndex,
) -> CorpusFlatSkillLibrary:
    library = CorpusFlatSkillLibrary(
        library_id="",
        benchmark=benchmark,
        prompt_sha256=prompt_sha256,
        corpus_sha256=corpus_sha256,
        training_trajectory_ids=tuple(sorted(training_trajectory_ids)),
        cards=tuple(sorted(cards, key=lambda card: card.skill_id)),
        index=index,
    )
    return replace(
        library,
        library_id=f"flatcorpus_{_digest(library.content_record())[:16]}",
    )


def load_flat_library(record: Mapping) -> FlatLibrary:
    if record.get("library_kind") == "corpus_induced":
        return CorpusFlatSkillLibrary.from_record(record)
    return FlatSkillLibrary.from_record(record)


def _digest(record: dict) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
