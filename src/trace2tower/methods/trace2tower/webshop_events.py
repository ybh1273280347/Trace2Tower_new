from __future__ import annotations

import json
import re
from collections.abc import Sequence

from trace2tower.benchmarks.models import ClickableKind
from trace2tower.methods.trace2tower.models import (
    SegmentInstance,
    StepTransition,
    WebShopEventType,
    WebShopPageType,
)
from trace2tower.methods.trace2tower.transitions import build_transitions
from trace2tower.trajectory import EpisodeTrajectory, StepRecord

DETAIL_VALUES = {"description", "features", "reviews", "attributes"}
ASIN_PATTERN = re.compile(r"^[A-Za-z0-9]{8,16}$")
RESULT_ENTITY_PATTERN = re.compile(
    r"^(?P<asin>[A-Za-z0-9]{8,16})\s*\|\s*(?P<title>.*?)\s*\|\s*(?P<price>.+)$"
)


def infer_webshop_page_type(observation: str) -> WebShopPageType:
    first_line = observation.splitlines()[0].strip().casefold() if observation else ""
    if first_line == "webshop search page.":
        return WebShopPageType.SEARCH
    if first_line.startswith("search results page"):
        return WebShopPageType.RESULTS
    if first_line.startswith("product:"):
        return WebShopPageType.ITEM
    if any(
        first_line.startswith(f"{detail}:")
        for detail in ("description", "features", "reviews", "attributes")
    ):
        return WebShopPageType.ITEM_DETAIL
    if first_line.startswith("purchase completed"):
        return WebShopPageType.TERMINAL
    return WebShopPageType.UNKNOWN


def webshop_applicable_events(page: WebShopPageType) -> frozenset[WebShopEventType]:
    return {
        WebShopPageType.SEARCH: frozenset(
            (WebShopEventType.QUERY_FORMULATION, WebShopEventType.QUERY_REFINEMENT)
        ),
        WebShopPageType.RESULTS: frozenset(
            (
                WebShopEventType.CANDIDATE_SELECTION,
                WebShopEventType.RESULT_NAVIGATION,
                WebShopEventType.SEARCH_BACKTRACKING,
            )
        ),
        WebShopPageType.ITEM: frozenset(
            (
                WebShopEventType.OPTION_SELECTION,
                WebShopEventType.ATTRIBUTE_INSPECTION,
                WebShopEventType.PURCHASE_DECISION,
                WebShopEventType.SEARCH_BACKTRACKING,
            )
        ),
        WebShopPageType.ITEM_DETAIL: frozenset(
            (
                WebShopEventType.DETAIL_BACKTRACKING,
                WebShopEventType.SEARCH_BACKTRACKING,
            )
        ),
    }.get(page, frozenset((WebShopEventType.OTHER_CLICK,)))


class WebShopEventClassifier:
    def __init__(
        self,
        page: WebShopPageType = WebShopPageType.SEARCH,
        *,
        searched: bool = False,
    ):
        self.page = page
        self.searched = searched

    def classify(self, step: StepRecord) -> WebShopEventType:
        if step.action_name == "search_action" and isinstance(
            (step.action_arguments or {}).get("keywords"), str
        ):
            event = (
                WebShopEventType.QUERY_REFINEMENT
                if self.searched
                else WebShopEventType.QUERY_FORMULATION
            )
            self.searched = True
            self.page = WebShopPageType.RESULTS
            return event

        if step.action_name != "click_action" or not isinstance(
            (step.action_arguments or {}).get("value"), str
        ):
            return WebShopEventType.OTHER_CLICK

        raw_value = step.action_arguments["value"]
        value = raw_value.strip().casefold()
        if value == "buy now":
            event = WebShopEventType.PURCHASE_DECISION
        elif value == "back to search":
            event = WebShopEventType.SEARCH_BACKTRACKING
        elif self.page is WebShopPageType.RESULTS and value in {"next >", "< prev"}:
            event = WebShopEventType.RESULT_NAVIGATION
        elif self.page is WebShopPageType.ITEM_DETAIL and value == "< prev":
            event = WebShopEventType.DETAIL_BACKTRACKING
        elif self.page is WebShopPageType.ITEM and value in DETAIL_VALUES:
            event = WebShopEventType.ATTRIBUTE_INSPECTION
        else:
            clickable_kind = next(
                (
                    kind
                    for clickable, kind in step.clickable_types.items()
                    if clickable.casefold() == value
                ),
                None,
            )
            if self.page is WebShopPageType.RESULTS and (
                clickable_kind is ClickableKind.PRODUCT_LINK
                or (
                    not step.clickable_types and value not in {"next >", "< prev", "back to search"}
                )
            ):
                event = WebShopEventType.CANDIDATE_SELECTION
            elif self.page is WebShopPageType.UNKNOWN and ASIN_PATTERN.fullmatch(raw_value.strip()):
                event = WebShopEventType.CANDIDATE_SELECTION
            elif self.page is WebShopPageType.ITEM and (
                clickable_kind is ClickableKind.OPTION
                or (
                    not step.clickable_types
                    and value not in DETAIL_VALUES | {"buy now", "back to search"}
                    and not ASIN_PATTERN.fullmatch(raw_value.strip())
                )
            ):
                event = WebShopEventType.OPTION_SELECTION
            else:
                event = WebShopEventType.OTHER_CLICK

        page_updates = {
            WebShopEventType.QUERY_FORMULATION: WebShopPageType.RESULTS,
            WebShopEventType.QUERY_REFINEMENT: WebShopPageType.RESULTS,
            WebShopEventType.RESULT_NAVIGATION: WebShopPageType.RESULTS,
            WebShopEventType.CANDIDATE_SELECTION: WebShopPageType.ITEM,
            WebShopEventType.OPTION_SELECTION: WebShopPageType.ITEM,
            WebShopEventType.ATTRIBUTE_INSPECTION: WebShopPageType.ITEM_DETAIL,
            WebShopEventType.DETAIL_BACKTRACKING: WebShopPageType.ITEM,
            WebShopEventType.SEARCH_BACKTRACKING: WebShopPageType.SEARCH,
            WebShopEventType.PURCHASE_DECISION: WebShopPageType.TERMINAL,
        }
        if event is not WebShopEventType.OTHER_CLICK:
            self.page = page_updates[event]
        return event


def classify_webshop_steps(steps: Sequence[StepRecord]) -> tuple[WebShopEventType, ...]:
    classifier = WebShopEventClassifier()
    return tuple(classifier.classify(step) for step in steps)


def segment_webshop_trajectory(
    trajectory: EpisodeTrajectory,
    transitions: Sequence[StepTransition] | None = None,
    embeddings: Sequence[Sequence[float]] | None = None,
) -> tuple[SegmentInstance, ...]:
    aligned_transitions = tuple(transitions or build_transitions(trajectory))
    if len(aligned_transitions) != len(trajectory.steps):
        raise ValueError("trajectory and transitions must align")
    if embeddings is not None and len(embeddings) != len(aligned_transitions):
        raise ValueError("trajectory and embeddings must align")
    events = classify_webshop_steps(trajectory.steps)
    if not events:
        return ()

    boundaries = []
    start = 0
    for index in range(1, len(events) + 1):
        if index == len(events) or events[index] is not events[start]:
            boundaries.append((start, index - 1, events[start]))
            start = index

    segments = []
    for start, end, event in boundaries:
        if embeddings is None:
            embedding = ()
        else:
            segment_vectors = embeddings[start : end + 1]
            embedding = tuple(
                sum(vector[index] for vector in segment_vectors) / len(segment_vectors)
                for index in range(len(segment_vectors[0]))
            )
        segments.append(
            SegmentInstance(
                segment_id=f"{trajectory.trajectory_id}:segment:{start}-{end}",
                trajectory_id=trajectory.trajectory_id,
                start_step=start,
                end_step=end,
                transition_ids=tuple(
                    transition.transition_id for transition in aligned_transitions[start : end + 1]
                ),
                embedding=embedding,
                trajectory_score=trajectory.primary_score,
                event_type=event,
                raw_actions=tuple(
                    transition.raw_action for transition in aligned_transitions[start : end + 1]
                ),
                observation_before=aligned_transitions[start].observation_before,
                observation_after=aligned_transitions[end].observation_after,
            )
        )
    return tuple(segments)


def webshop_segment_signature(
    segment: SegmentInstance,
    *,
    goal: str = "",
    previous_event: WebShopEventType | None = None,
    next_event: WebShopEventType | None = None,
) -> str:
    if segment.event_type is None:
        raise ValueError("WebShop segment signature requires an event type")

    before_context = _compact_page_context(segment.observation_before)
    after_context = _compact_page_context(segment.observation_after)
    action_templates = []
    for raw_action in segment.raw_actions:
        try:
            action = json.loads(raw_action)
        except json.JSONDecodeError:
            action_templates.append("UNKNOWN_ACTION")
            continue
        name = action.get("name")
        arguments = action.get("arguments") or {}
        if name == "search_action":
            keywords = arguments.get("keywords")
            query = _normalize_text(keywords) if isinstance(keywords, str) else ""
            action_templates.append(f"SEARCH(query={query})")
            continue
        if name != "click_action":
            action_templates.append(str(name or "UNKNOWN_ACTION").upper())
            continue

        value = arguments.get("value")
        normalized_value = value.strip().casefold() if isinstance(value, str) else ""
        if segment.event_type is WebShopEventType.ATTRIBUTE_INSPECTION:
            detail = normalized_value if normalized_value in DETAIL_VALUES else "detail"
            action_templates.append(f"OPEN_DETAIL({detail}, product={before_context})")
        elif segment.event_type is WebShopEventType.RESULT_NAVIGATION:
            direction = "next" if normalized_value == "next >" else "previous"
            action_templates.append(f"PAGE({direction})")
        elif segment.event_type is WebShopEventType.CANDIDATE_SELECTION:
            action_templates.append(
                f"OPEN_PRODUCT(id={normalized_value}, product={after_context})"
            )
        elif segment.event_type is WebShopEventType.OPTION_SELECTION:
            action_templates.append(
                f"SELECT_OPTION(value={normalized_value}, product={before_context})"
            )
        elif segment.event_type is WebShopEventType.DETAIL_BACKTRACKING:
            action_templates.append("BACK_TO_PRODUCT")
        elif segment.event_type is WebShopEventType.SEARCH_BACKTRACKING:
            action_templates.append("BACK_TO_SEARCH")
        elif segment.event_type is WebShopEventType.PURCHASE_DECISION:
            action_templates.append("BUY_NOW")
        else:
            action_templates.append("CLICK")

    return "\n".join(
        (
            f"Goal: {_normalize_text(goal)}",
            f"Previous event: {previous_event.value if previous_event else 'START'}",
            f"Event: {segment.event_type.value}",
            f"Next event: {next_event.value if next_event else 'END'}",
            f"Length: {segment.end_step - segment.start_step + 1}",
            f"Actions: {' -> '.join(action_templates)}",
            f"Page before: {before_context}",
            f"Page after: {after_context}",
        )
    )


def webshop_entity_signature(
    segment: SegmentInstance,
    *,
    goal: str = "",
    previous_event: WebShopEventType | None = None,
    next_event: WebShopEventType | None = None,
) -> str:
    """用商品实体和约束状态表达片段，事件只作为实体关系保留。"""
    if segment.event_type is None:
        raise ValueError("WebShop entity signature requires an event type")
    relation = {
        WebShopEventType.QUERY_FORMULATION: "formulate search for target product",
        WebShopEventType.QUERY_REFINEMENT: "refine search for target product",
        WebShopEventType.RESULT_NAVIGATION: "navigate candidate product set",
        WebShopEventType.CANDIDATE_SELECTION: "open candidate product",
        WebShopEventType.OPTION_SELECTION: "configure candidate product variant",
        WebShopEventType.ATTRIBUTE_INSPECTION: "verify candidate product attributes",
        WebShopEventType.DETAIL_BACKTRACKING: "return to candidate product",
        WebShopEventType.SEARCH_BACKTRACKING: "return to target product search",
        WebShopEventType.PURCHASE_DECISION: "purchase verified product",
        WebShopEventType.OTHER_CLICK: "interact with product state",
    }[segment.event_type]
    return "\n".join(
        (
            f"Target product and constraints: {_normalize_text(goal)}",
            f"Observed entity before: {_product_entity_context(segment.observation_before)}",
            f"Observed entity after: {_product_entity_context(segment.observation_after)}",
            f"Entity relation: {relation}",
            f"Previous relation: {previous_event.value if previous_event else 'START'}",
            f"Next relation: {next_event.value if next_event else 'END'}",
            f"Actions: {' -> '.join(segment.raw_actions)}",
        )
    )


def _product_entity_context(observation: str) -> str:
    lines = tuple(line.strip() for line in observation.splitlines() if line.strip())
    if not lines:
        return "EMPTY"
    if lines[0].casefold() == "webshop search page.":
        return "SEARCH_SPACE"
    products = []
    title = ""
    price = ""
    options = []
    selected = ""
    for line in lines:
        lowered = line.casefold()
        if lowered.startswith("product:"):
            title = _normalize_text(line.split(":", 1)[1])
        elif lowered.startswith("price:"):
            price = _normalize_text(line.split(":", 1)[1])
        elif lowered.startswith("selected:"):
            selected = _normalize_text(line.split(":", 1)[1])
        elif line.startswith("- ") and ":" in line:
            name, values = line[2:].split(":", 1)
            options.append(f"{_normalize_text(name)}={_normalize_text(values)[:160]}")
        else:
            match = RESULT_ENTITY_PATTERN.match(line)
            if match:
                products.append(
                    f"{_normalize_text(match.group('title'))}"
                    f" price={_normalize_text(match.group('price'))[:40]}"
                )
    if title:
        context = [f"PRODUCT title={title[:240]}"]
        if price:
            context.append(f"price={price[:40]}")
        if options:
            context.append("options=" + "; ".join(options[:6]))
        if selected:
            context.append(f"selected={selected[:160]}")
        return " | ".join(context)
    if products:
        return "RESULT_SET " + " || ".join(products[:5])
    return infer_webshop_page_type(observation).value


def _compact_page_context(observation: str) -> str:
    lines = tuple(line.strip() for line in observation.splitlines() if line.strip())
    first_line = lines[0] if lines else ""
    return f"{infer_webshop_page_type(observation).value}:{_normalize_text(first_line)[:240]}"


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().strip().split())
