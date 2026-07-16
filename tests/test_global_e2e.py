from __future__ import annotations

import asyncio

import pytest

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import EmbeddingResult, LLMUsage
from trace2tower.manifests import Benchmark
from trace2tower.methods.global_e2e.models import (
    GlobalE2ESkillCard,
    GlobalE2ESkillLibrary,
    build_global_e2e_library,
)
from trace2tower.methods.global_e2e.provider import GlobalE2ESkillProvider
from trace2tower.semantic_index import SkillEmbeddingIndex


def library() -> GlobalE2ESkillLibrary:
    trajectory_ids = (
        "webshop:train:no_skill:webshop:1000:0",
        "webshop:train:no_skill:webshop:1001:0",
    )
    cards = (
        GlobalE2ESkillCard(
            "global_a",
            trajectory_ids,
            "Direct purchase",
            "Use for a direct product match.",
            ("Search and verify the product.", "Buy after all constraints match."),
            ("Respect the price ceiling.",),
        ),
        GlobalE2ESkillCard(
            "global_b",
            trajectory_ids,
            "Attribute verification",
            "Use when hidden properties require inspection.",
            ("Inspect the relevant detail tab.", "Return and buy after verification."),
            ("Do not infer absent properties.",),
        ),
    )
    return build_global_e2e_library(
        Benchmark.WEBSHOP,
        "a" * 64,
        "b" * 64,
        trajectory_ids,
        cards,
        SkillEmbeddingIndex(
            tuple(card.skill_id for card in cards),
            ((1.0, 0.0), (0.0, 1.0)),
            ("c" * 64, "d" * 64),
        ),
    )


def test_library_round_trip_and_content_id_reject_tampering() -> None:
    current = library()
    assert GlobalE2ESkillLibrary.from_record(current.to_record()) == current
    record = current.to_record()
    record["cards"][0]["name"] = "tampered"
    with pytest.raises(ValueError, match="library ID"):
        GlobalE2ESkillLibrary.from_record(record)


def test_provider_injects_exactly_one_end_to_end_skill() -> None:
    class FakeRuntime:
        async def embed(self, texts) -> EmbeddingResult:
            assert texts == ["goal\ninitial"]
            return EmbeddingResult(((1.0, 0.0),), LLMUsage(11, None, None), 1)

    selection = asyncio.run(
        GlobalE2ESkillProvider(FakeRuntime(), library()).select(
            "goal",
            EnvironmentState("initial", (), {}, False, 0.0, False, True),
        )
    )
    assert selection.skill_ids == ("global_a",)
    assert selection.model_input_tokens == 11
    assert "Direct purchase" in selection.context
