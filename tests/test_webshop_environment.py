from __future__ import annotations

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.benchmarks.webshop import WebShopEnvironment, WebShopPage


def environment_with_product() -> WebShopEnvironment:
    environment = WebShopEnvironment.__new__(WebShopEnvironment)
    environment.page = WebShopPage.ITEM
    environment.product = {
        "asin": "ITEM12345",
        "Title": "Example product",
        "pricing": "$19.99",
        "Description": "This content must require an explicit click.",
        "options": {"color": ["red", "blue"]},
    }
    environment.selected_options = {}
    environment.current_state = environment._item_page()
    return environment


def test_item_page_requires_description_click() -> None:
    environment = environment_with_product()

    assert "This content must require an explicit click." not in environment.current_state.observation
    assert "Description" in environment.current_state.admissible_actions
    assert "< Prev" in environment.current_state.admissible_actions


def test_item_prev_restores_result_page(monkeypatch) -> None:
    environment = environment_with_product()
    result_state = EnvironmentState("Search results page 2:", (), {}, False, 0, False, True)
    monkeypatch.setattr(environment, "_results_page", lambda: result_state)

    state = environment._click("< Prev")

    assert state is result_state
    assert environment.page is WebShopPage.RESULTS
    assert environment.product is None
    assert environment.selected_options == {}
