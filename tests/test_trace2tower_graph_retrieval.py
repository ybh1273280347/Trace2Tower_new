from __future__ import annotations

import pytest

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
)
from trace2tower.methods.trace2tower.graph_retrieval import (
    TowerGraphProfile,
    retrieve_tower_graph,
)
from trace2tower.methods.trace2tower.models import (
    AlfworldEventType,
    HighPath,
    PrimitiveAction,
    WebShopEventType,
    WebShopPageType,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.webshop_events import (
    infer_webshop_page_type,
    webshop_applicable_events,
)
from trace2tower.semantic_index import SkillEmbeddingIndex


def mid_card(skill_id: str, name: str) -> MidSkillCard:
    return MidSkillCard(
        skill_id,
        (f"trajectory:segment:{skill_id}",),
        name,
        "Use in the matching state.",
        (f"Execute {name}.",),
        (),
        (PrimitiveAction.CLICK,),
    )


def test_graph_retrieval_uses_current_event_and_respects_total_mid_budget() -> None:
    mids = {
        "mid_search": mid_card("mid_search", "Search"),
        "mid_option": mid_card("mid_option", "Choose option"),
        "mid_detail": mid_card("mid_detail", "Inspect details"),
        "mid_back": mid_card("mid_back", "Return to product"),
    }
    highs = {
        "high_search": HighSkillCard(
            "high_search",
            ("mid_search", "mid_option"),
            "Search then configure",
            "Use for product search.",
            ("Search, then configure.",),
        ),
        "high_detail": HighSkillCard(
            "high_detail",
            ("mid_detail", "mid_back"),
            "Inspect then return",
            "Use for product verification.",
            ("Inspect details, then return.",),
        ),
    }
    paths = {
        "high_search": HighPath(
            "high_search",
            ("mid_search", "mid_option"),
            0.6,
            0.2,
            0.4,
            ("trajectory",),
        ),
        "high_detail": HighPath(
            "high_detail",
            ("mid_detail", "mid_back"),
            0.8,
            0.1,
            0.7,
            ("trajectory",),
        ),
    }
    profile = TowerGraphProfile(
        "tower_test",
        Benchmark.WEBSHOP,
        {
            "mid_search": {WebShopEventType.QUERY_FORMULATION: 10},
            "mid_option": {WebShopEventType.OPTION_SELECTION: 10},
            "mid_detail": {WebShopEventType.ATTRIBUTE_INSPECTION: 10},
            "mid_back": {WebShopEventType.DETAIL_BACKTRACKING: 10},
        },
    )
    result = retrieve_tower_graph(
        (1.0, 0.0),
        (0.0, 1.0),
        SkillEmbeddingIndex(
            ("high_search", "high_detail"),
            ((1.0, 0.0), (0.8, 0.2)),
        ),
        SkillEmbeddingIndex(
            ("mid_search", "mid_option", "mid_detail", "mid_back"),
            ((1.0, 0.0), (0.2, 0.8), (0.0, 1.0), (0.1, 0.9)),
        ),
        highs,
        mids,
        paths,
        profile,
        webshop_applicable_events(WebShopPageType.ITEM),
        mid_context_budget=2,
    )
    assert result.graph_high_match.skill_id == "high_detail"
    assert result.retrieval.context_skill_ids == ("high_detail", "mid_detail")


def test_webshop_page_inference_drives_applicable_events() -> None:
    assert infer_webshop_page_type("WebShop search page.") is WebShopPageType.SEARCH
    assert infer_webshop_page_type("Search results page 1:\nitem") is WebShopPageType.RESULTS
    assert infer_webshop_page_type("Product: item") is WebShopPageType.ITEM
    assert infer_webshop_page_type("Attributes:\n['x']") is WebShopPageType.ITEM_DETAIL
    assert WebShopEventType.ATTRIBUTE_INSPECTION in webshop_applicable_events(WebShopPageType.ITEM)
    assert WebShopEventType.ATTRIBUTE_INSPECTION not in webshop_applicable_events(
        WebShopPageType.RESULTS
    )


def test_alfworld_graph_retrieval_supports_mid_only_fallback() -> None:
    mids = {
        "mid_pick": mid_card("mid_pick", "Pick up object"),
        "mid_put": mid_card("mid_put", "Place object"),
    }
    profile = TowerGraphProfile(
        "tower_alfworld",
        Benchmark.ALFWORLD,
        {
            "mid_pick": {AlfworldEventType.PICKUP_OBJECT: 10},
            "mid_put": {AlfworldEventType.PUT_OBJECT: 10},
        },
    )

    result = retrieve_tower_graph(
        (1.0, 0.0),
        (0.0, 1.0),
        SkillEmbeddingIndex((), ()),
        SkillEmbeddingIndex(
            ("mid_pick", "mid_put"),
            ((1.0, 0.0), (0.0, 1.0)),
        ),
        {},
        mids,
        {},
        profile,
        frozenset((AlfworldEventType.PUT_OBJECT,)),
        mid_context_budget=1,
    )

    assert result.graph_high_match is None
    assert result.retrieval.context_skill_ids == ("mid_put",)


def test_graph_mid_retrieval_can_return_empty_below_similarity_threshold() -> None:
    mids = {
        "mid_pick": mid_card("mid_pick", "Pick up object"),
        "mid_put": mid_card("mid_put", "Place object"),
    }
    profile = TowerGraphProfile(
        "tower_alfworld",
        Benchmark.ALFWORLD,
        {
            "mid_pick": {AlfworldEventType.PICKUP_OBJECT: 10},
            "mid_put": {AlfworldEventType.PUT_OBJECT: 10},
        },
    )

    result = retrieve_tower_graph(
        (),
        (1.0, 0.0),
        SkillEmbeddingIndex((), ()),
        SkillEmbeddingIndex(
            ("mid_pick", "mid_put"),
            ((0.0, 1.0), (0.0, 1.0)),
        ),
        {},
        mids,
        {},
        profile,
        frozenset(
            (
                AlfworldEventType.PICKUP_OBJECT,
                AlfworldEventType.PUT_OBJECT,
            )
        ),
        mid_context_budget=2,
        direct_mid_similarity_threshold=0.45,
    )

    assert result.retrieval.context_skill_ids == ()


def test_alfworld_high_path_requires_exact_goal_event_coverage() -> None:
    mids = {
        "mid_move": mid_card("mid_move", "Move"),
        "mid_clean": mid_card("mid_clean", "Clean"),
        "mid_toggle": mid_card("mid_toggle", "Toggle"),
    }
    highs = {
        "high_clean": HighSkillCard(
            "high_clean",
            ("mid_move", "mid_clean"),
            "Move and clean",
            "Use for cleaning.",
            ("Move, then clean.",),
        ),
        "high_toggle": HighSkillCard(
            "high_toggle",
            ("mid_move", "mid_toggle"),
            "Move and toggle",
            "Use for a lamp.",
            ("Move, then toggle.",),
        ),
    }
    paths = {
        skill_id: HighPath(
            skill_id,
            card.ordered_mid_ids,
            0.5,
            0.1,
            0.4,
            ("trajectory",),
        )
        for skill_id, card in highs.items()
    }
    profile = TowerGraphProfile(
        "tower_alfworld",
        Benchmark.ALFWORLD,
        {
            "mid_move": {AlfworldEventType.GOTO_LOCATION: 10},
            "mid_clean": {AlfworldEventType.CLEAN_OBJECT: 10},
            "mid_toggle": {AlfworldEventType.TOGGLE_OBJECT: 10},
        },
    )
    arguments = (
        (1.0, 0.0),
        (1.0, 0.0),
        SkillEmbeddingIndex(
            ("high_clean", "high_toggle"),
            ((0.8, 0.2), (1.0, 0.0)),
        ),
        SkillEmbeddingIndex(
            ("mid_move", "mid_clean", "mid_toggle"),
            ((1.0, 0.0), (0.8, 0.2), (0.9, 0.1)),
        ),
        highs,
        mids,
        paths,
        profile,
        frozenset(
            (
                AlfworldEventType.GOTO_LOCATION,
                AlfworldEventType.CLEAN_OBJECT,
            )
        ),
    )

    clean = retrieve_tower_graph(
        *arguments,
        mid_context_budget=2,
        required_path_events=frozenset((AlfworldEventType.CLEAN_OBJECT,)),
        exclusive_path_events=ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    )
    plain = retrieve_tower_graph(
        *arguments,
        mid_context_budget=2,
        exclusive_path_events=ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    )

    assert clean.graph_high_match.skill_id == "high_clean"
    assert plain.graph_high_match is None


def test_graph_profile_rejects_events_from_another_benchmark() -> None:
    with pytest.raises(ValueError, match="event types do not match"):
        TowerGraphProfile(
            "tower_alfworld",
            Benchmark.ALFWORLD,
            {"mid": {WebShopEventType.QUERY_FORMULATION: 1}},
        )
