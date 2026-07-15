from __future__ import annotations

from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.graph_retrieval import (
    TowerGraphProfile,
    retrieve_tower_graph,
)
from trace2tower.methods.trace2tower.models import (
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
    assert result.retrieval.context_skill_ids == (
        "high_detail",
        "mid_detail",
        "mid_back",
    )


def test_webshop_page_inference_drives_applicable_events() -> None:
    assert infer_webshop_page_type("WebShop search page.") is WebShopPageType.SEARCH
    assert (
        infer_webshop_page_type("Search results page 1:\nitem")
        is WebShopPageType.RESULTS
    )
    assert infer_webshop_page_type("Product: item") is WebShopPageType.ITEM
    assert infer_webshop_page_type("Attributes:\n['x']") is WebShopPageType.ITEM_DETAIL
    assert WebShopEventType.ATTRIBUTE_INSPECTION in webshop_applicable_events(
        WebShopPageType.ITEM
    )
    assert WebShopEventType.ATTRIBUTE_INSPECTION not in webshop_applicable_events(
        WebShopPageType.RESULTS
    )
