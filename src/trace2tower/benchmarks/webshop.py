from __future__ import annotations

import asyncio
import importlib
import json
import re
import sqlite3
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from trace2tower.benchmarks.models import ClickableKind, EnvironmentState, EpisodeStart
from trace2tower.core.manifests import Benchmark, ManifestEntry


class WebShopPage(StrEnum):
    SEARCH = "search"
    RESULTS = "results"
    ITEM = "item"
    ITEM_DETAIL = "item_detail"
    TERMINAL = "terminal"


class WebShopEnvironment:
    benchmark = Benchmark.WEBSHOP
    tool_schemas = (
        {
            "type": "function",
            "function": {
                "name": "search_action",
                "description": "Search for products using keywords.",
                "parameters": {
                    "type": "object",
                    "properties": {"keywords": {"type": "string"}},
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "click_action",
                "description": "Click one value from the available clickables.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
    )

    def __init__(self, dataset_root: Path, webshop_source_root: Path):
        self.dataset_root = dataset_root
        self.goals = json.loads((dataset_root / "goals.json").read_text(encoding="utf-8"))
        source_path = str(webshop_source_root.resolve())
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        self.get_reward = importlib.import_module("web_agent_site.engine.goal").get_reward
        self.page = WebShopPage.SEARCH
        self.goal: dict[str, Any] | None = None
        self.results: list[str] = []
        self.result_page = 1
        self.product: dict[str, Any] | None = None
        self.selected_options: dict[str, str] = {}
        self.detail_name: str | None = None
        self.current_state: EnvironmentState | None = None

    async def reset(self, entry: ManifestEntry) -> EpisodeStart:
        self.goal = self.goals[entry.dataset_index]
        self.page = WebShopPage.SEARCH
        self.results = []
        self.product = None
        self.selected_options = {}
        self.current_state = self._search_page()
        return EpisodeStart(task_goal=self.goal["instruction_text"], state=self.current_state)

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> EnvironmentState:
        if tool_name == "search_action" and isinstance(arguments.get("keywords"), str):
            if not self.current_state or not self.current_state.search_available:
                return self._invalid("Search is not available on this page.")
            self.results = await asyncio.to_thread(self._search, arguments["keywords"])
            self.result_page = 1
            self.page = WebShopPage.RESULTS
            self.current_state = self._results_page()
            return self.current_state
        if tool_name != "click_action" or not isinstance(arguments.get("value"), str):
            return self._invalid("Invalid tool call.")
        return self._click(arguments["value"])

    async def close(self) -> None:
        return None

    def _click(self, raw_value: str) -> EnvironmentState:
        if self.current_state is None:
            raise RuntimeError("WebShop environment has not been reset")
        values = {value.casefold(): value for value in self.current_state.admissible_actions}
        value = values.get(raw_value.casefold())
        if value is None:
            return self._invalid(f"Invalid click: {raw_value}")

        if value.casefold() == "back to search":
            self.page = WebShopPage.SEARCH
            self.current_state = self._search_page()
        elif self.page is WebShopPage.RESULTS and value == "next >":
            self.result_page += 1
            self.current_state = self._results_page()
        elif self.page is WebShopPage.RESULTS and value == "< prev":
            self.result_page -= 1
            self.current_state = self._results_page()
        elif self.page is WebShopPage.RESULTS:
            self.product = self._load_product(value)
            self.selected_options = {}
            self.page = WebShopPage.ITEM
            self.current_state = self._item_page()
        elif self.page is WebShopPage.ITEM and value in (
            "Description",
            "Features",
            "Reviews",
            "Attributes",
        ):
            self.detail_name = value
            self.page = WebShopPage.ITEM_DETAIL
            self.current_state = self._detail_page()
        elif self.page is WebShopPage.ITEM_DETAIL and value == "< prev":
            self.page = WebShopPage.ITEM
            self.current_state = self._item_page()
        elif self.page is WebShopPage.ITEM and value == "Buy Now":
            self.current_state = self._purchase()
        elif self.page is WebShopPage.ITEM:
            option_name = next(
                (name for name, options in self.product["options"].items() if value in options),
                None,
            )
            if option_name is None:
                return self._invalid(f"Invalid option: {value}")
            self.selected_options[option_name] = value
            self.current_state = self._item_page()
        else:
            return self._invalid(f"Invalid click on {self.page}: {value}")
        return self.current_state

    def _search(self, keywords: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", keywords.casefold())
        if not tokens:
            return []
        query = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
        with sqlite3.connect(
            f"file:{(self.dataset_root / 'search.sqlite').as_posix()}?mode=ro", uri=True
        ) as database:
            rows = database.execute(
                "SELECT asin FROM products_fts WHERE products_fts MATCH ? "
                "ORDER BY bm25(products_fts) LIMIT 100",
                (query,),
            ).fetchall()
        unique = []
        for (asin,) in rows:
            if asin not in unique:
                unique.append(asin)
            if len(unique) == 50:
                break
        return unique

    def _load_product(self, asin: str) -> dict[str, Any]:
        with sqlite3.connect(
            f"file:{(self.dataset_root / 'products.sqlite').as_posix()}?mode=ro", uri=True
        ) as database:
            row = database.execute(
                "SELECT product_json FROM products WHERE asin = ?", (asin,)
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown product: {asin}")
        return json.loads(row[0])

    def _search_page(self) -> EnvironmentState:
        return EnvironmentState(
            observation="WebShop search page.",
            admissible_actions=(),
            clickable_types={},
            search_available=True,
            reward=0,
            done=False,
            valid_action=True,
        )

    def _results_page(self) -> EnvironmentState:
        start = (self.result_page - 1) * 10
        page_asins = self.results[start : start + 10]
        products = [self._load_product(asin) for asin in page_asins]
        lines = [f"Search results page {self.result_page}:"]
        lines.extend(
            f"{product['asin']} | {product['Title']} | {product.get('pricing', '')}"
            for product in products
        )
        actions = list(page_asins)
        kinds = {asin: ClickableKind.PRODUCT_LINK for asin in page_asins}
        if start + 10 < len(self.results):
            actions.append("next >")
            kinds["next >"] = ClickableKind.BUTTON
        if self.result_page > 1:
            actions.append("< prev")
            kinds["< prev"] = ClickableKind.BUTTON
        actions.append("Back to Search")
        kinds["Back to Search"] = ClickableKind.BUTTON
        return EnvironmentState("\n".join(lines), tuple(actions), kinds, False, 0, False, True)

    def _item_page(self) -> EnvironmentState:
        product = self.product
        lines = [
            f"Product: {product['Title']}",
            f"ASIN: {product['asin']}",
            f"Price: {product.get('pricing', '')}",
            f"Description: {product.get('Description', '')[:1200]}",
            "Options:",
        ]
        actions = []
        kinds = {}
        for name, options in product.get("options", {}).items():
            lines.append(f"- {name}: {', '.join(options)}")
            for value in options:
                actions.append(value)
                kinds[value] = ClickableKind.OPTION
        for value in (
            "Description",
            "Features",
            "Reviews",
            "Attributes",
            "Buy Now",
            "Back to Search",
        ):
            actions.append(value)
            kinds[value] = ClickableKind.BUTTON
        if self.selected_options:
            lines.append(f"Selected: {self.selected_options}")
        return EnvironmentState("\n".join(lines), tuple(actions), kinds, False, 0, False, True)

    def _detail_page(self) -> EnvironmentState:
        field = {
            "Description": "Description",
            "Features": "BulletPoints",
            "Reviews": "Reviews",
            "Attributes": "Attributes",
        }[self.detail_name]
        content = self.product.get(field, "")
        observation = f"{self.detail_name}:\n{content}"
        actions = ("< prev", "Back to Search")
        kinds = {value: ClickableKind.BUTTON for value in actions}
        return EnvironmentState(observation, actions, kinds, False, 0, False, True)

    def _purchase(self) -> EnvironmentState:
        reward = float(
            self.get_reward(
                self.product,
                self.goal,
                float(self.product["_price"]),
                self.selected_options,
            )
        )
        self.page = WebShopPage.TERMINAL
        return EnvironmentState(
            observation=f"Purchase completed. Reward: {reward:.4f}",
            admissible_actions=(),
            clickable_types={},
            search_available=False,
            reward=reward,
            done=True,
            valid_action=True,
        )

    def _invalid(self, message: str) -> EnvironmentState:
        state = self.current_state
        return EnvironmentState(
            observation=message,
            admissible_actions=state.admissible_actions if state else (),
            clickable_types=state.clickable_types if state else {},
            search_available=state.search_available if state else False,
            reward=0,
            done=False,
            valid_action=False,
        )
