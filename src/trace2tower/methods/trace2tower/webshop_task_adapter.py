from __future__ import annotations

import re
from collections.abc import Mapping

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.methods.trace2tower.models import WebShopEventType
from trace2tower.methods.trace2tower.skills import HighSkillCard
from trace2tower.methods.trace2tower.task_conditioning import (
    TaskCompatibility,
    TaskCondition,
    TaskProperty,
)


_IGNORED_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "i",
        "in",
        "is",
        "me",
        "of",
        "on",
        "the",
        "to",
        "want",
        "with",
    }
)
_WORKFLOW_EVENTS = (
    WebShopEventType.QUERY_FORMULATION.value,
    WebShopEventType.CANDIDATE_SELECTION.value,
    WebShopEventType.PURCHASE_DECISION.value,
)


def _constraint_terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            term
            for term in re.findall(r"[a-z0-9]+", text.casefold())
            if term not in _IGNORED_TERMS
        )
    )


class WebshopTaskAdapter:
    domain = "webshop"

    def extract_query(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> TaskCondition:
        return TaskCondition(
            task_text=task_goal,
            retrieval_text=task_goal,
            required_events=_WORKFLOW_EVENTS,
            properties=(TaskProperty("constraints", _constraint_terms(task_goal)),),
        )

    def profile_condition(self, record: Mapping) -> TaskCondition:
        task_text = str(record["task_text"])
        return TaskCondition(
            task_text=task_text,
            retrieval_text=task_text,
            required_events=tuple(
                str(value) for value in record.get("required_events", _WORKFLOW_EVENTS)
            ),
            properties=(TaskProperty("constraints", _constraint_terms(task_text)),),
        )

    def compatibility(
        self,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> TaskCompatibility:
        if not set(query.required_events).intersection(candidate.required_events):
            return TaskCompatibility.INCOMPATIBLE
        query_terms = set(query.values("constraints"))
        candidate_terms = set(candidate.values("constraints"))
        if query_terms == candidate_terms:
            return TaskCompatibility.EXACT
        if query_terms.intersection(candidate_terms):
            return TaskCompatibility.PARTIAL
        return TaskCompatibility.WORKFLOW

    def bind(
        self,
        source_card: HighSkillCard,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> HighSkillCard:
        return HighSkillCard(
            skill_id=source_card.skill_id,
            ordered_mid_ids=source_card.ordered_mid_ids,
            name=source_card.name,
            description=f"Current concrete shopping task: {query.task_text}",
            procedure=(
                f"Bind every required product constraint from: {query.task_text}",
                *source_card.procedure,
            ),
            constraints=source_card.constraints,
            member_mid_ids=source_card.member_mid_ids,
        )
