from __future__ import annotations

from trace2tower.benchmarks.models import EnvironmentState


class WebshopPlanRewriteAdapter:
    domain = "webshop"
    plan_rewrite_instructions = (
        "WebShop plan semantics:\n"
        "- Replace every product category, brand, attribute value, option value, quantity, "
        "and price from reference strategies with the current task requirements. Never carry "
        "a reference value into the plan unless the current task requires the same value.\n"
        "- Use references as procedural evidence for query formulation, candidate screening, "
        "detail inspection, option selection, recovery, and purchase. When no reference shares "
        "the product category, retain only this workflow evidence.\n"
        "- Form a concise search query from the current product type and its most discriminative "
        "constraints. Do not require every phrase or the price limit to appear in the query.\n"
        "- Treat visible result titles, prices, product details, attribute views, and selectable "
        "options as authoritative. Do not infer compliance from semantic similarity alone.\n"
        "- Backtrack or refine the query when a candidate cannot satisfy a required property. "
        "Buy only after the product, price, and every required option have been verified.\n"
    )

    def task_text(self, task_goal: str, state: EnvironmentState) -> str:
        return task_goal
