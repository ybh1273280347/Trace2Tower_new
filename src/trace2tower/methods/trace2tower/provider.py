from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.action_parser import parse_alfworld_action
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    alfworld_applicable_events,
    alfworld_goal_events,
)
from trace2tower.methods.trace2tower.graph_retrieval import (
    TowerGraphProfile,
    retrieve_task_high,
    retrieve_tower_graph,
)
from trace2tower.methods.trace2tower.models import PrimitiveAction
from trace2tower.methods.trace2tower.refinement import SkillStatus
from trace2tower.methods.trace2tower.retrieval import format_tower_context, retrieve_tower
from trace2tower.methods.trace2tower.tower import TowerSnapshot
from trace2tower.semantic_index import SkillEmbeddingIndex
from trace2tower.methods.trace2tower.webshop_events import (
    infer_webshop_page_type,
    webshop_applicable_events,
)


class Trace2TowerSkillProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        *,
        include_high: bool = True,
        direct_mid_top_k: int = 3,
        high_similarity_threshold: float = -1.0,
        include_high_child_context: bool = True,
        direct_mid_candidate_top_k: int | None = None,
        direct_mid_similarity_threshold: float = 0.45,
        direct_mid_relative_margin: float = 0.08,
        direct_mid_dedup_similarity_threshold: float = 0.95,
        direct_mid_mmr_lambda: float = 0.75,
        downweighted_skill_ids: frozenset[str] = frozenset(),
        status_tie_epsilon: float = 0.0,
        graph_profile: TowerGraphProfile | None = None,
        mid_context_budget: int | None = None,
        min_event_compatibility: float = 0.1,
        low_top_k: int = 3,
    ):
        snapshot.require_complete()
        if not -1 <= high_similarity_threshold <= 1:
            raise ValueError("High similarity threshold must be in [-1, 1]")
        if not isinstance(include_high, bool):
            raise ValueError("High inclusion switch must be boolean")
        if not 0 <= low_top_k <= 3:
            raise ValueError("Low Top-K must be in [0, 3]")
        if not 1 <= direct_mid_top_k <= 12:
            raise ValueError("direct Mid cap must be in [1, 12]")
        if not isinstance(include_high_child_context, bool):
            raise ValueError("High child context switch must be boolean")
        self.runtime = runtime
        self.snapshot = snapshot
        self.include_high = include_high
        self.direct_mid_top_k = direct_mid_top_k
        self.high_similarity_threshold = high_similarity_threshold
        self.include_high_child_context = include_high_child_context
        self.direct_mid_candidate_top_k = direct_mid_candidate_top_k
        self.direct_mid_similarity_threshold = direct_mid_similarity_threshold
        self.direct_mid_relative_margin = direct_mid_relative_margin
        self.direct_mid_dedup_similarity_threshold = direct_mid_dedup_similarity_threshold
        self.direct_mid_mmr_lambda = direct_mid_mmr_lambda
        self.downweighted_skill_ids = downweighted_skill_ids
        self.status_tie_epsilon = status_tie_epsilon
        self.graph_profile = graph_profile
        self.mid_context_budget = mid_context_budget
        self.min_event_compatibility = min_event_compatibility
        self.low_top_k = low_top_k
        if graph_profile is not None:
            if graph_profile.tower_snapshot_id != snapshot.snapshot_id:
                raise ValueError("graph profile does not bind to this Tower snapshot")
            if graph_profile.benchmark is not snapshot.benchmark:
                raise ValueError("graph profile benchmark does not match the Tower")
            if mid_context_budget is None:
                raise ValueError("graph retrieval requires a Mid context budget")
        elif mid_context_budget is not None:
            raise ValueError("Mid context budget is only valid for graph retrieval")
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}
        self.mid_cards = {card.skill_id: card for card in snapshot.mid_cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        lifecycle_report_path: Path | None = None,
        graph_profile_path: Path | None = None,
        **kwargs,
    ) -> Trace2TowerSkillProvider:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot = TowerSnapshot.from_record(payload)
        downweighted_skill_ids = frozenset()
        graph_profile = (
            TowerGraphProfile.from_record(
                json.loads(graph_profile_path.read_text(encoding="utf-8"))
            )
            if graph_profile_path is not None
            else None
        )
        if lifecycle_report_path is not None:
            lifecycle = json.loads(lifecycle_report_path.read_text(encoding="utf-8"))
            if (
                lifecycle.get("tower_snapshot_id") != snapshot.snapshot_id
                or lifecycle.get("ranking_status") != "complete"
            ):
                raise ValueError("lifecycle report does not bind to this Tower snapshot")
            updates = lifecycle.get("downweight", ())
            if any(
                update.get("new_status") != SkillStatus.DOWNWEIGHTED.value for update in updates
            ):
                raise ValueError("lifecycle report contains an invalid status update")
            downweighted_skill_ids = frozenset(str(update["skill_id"]) for update in updates)
        return cls(
            runtime,
            snapshot,
            downweighted_skill_ids=downweighted_skill_ids,
            graph_profile=graph_profile,
            **kwargs,
        )

    async def select(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        task_selection = await self.select_task(task_goal, state)
        state_selection = await self.select_state(task_goal, state)
        return SkillSelection(
            (*task_selection.skill_ids, *state_selection.skill_ids),
            "\n\n".join(
                context
                for context in (task_selection.context, state_selection.context)
                if context
            ),
            (
                None
                if task_selection.model_input_tokens is None
                or state_selection.model_input_tokens is None
                else task_selection.model_input_tokens
                + state_selection.model_input_tokens
            ),
            0,
            (*task_selection.context_skill_ids, *state_selection.context_skill_ids),
        )

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        if not self.include_high:
            return SkillSelection((), "")

        query_result = await self.runtime.embed([task_goal])
        if self.graph_profile:
            required_path_events = (
                alfworld_goal_events(task_goal)
                if self.snapshot.benchmark is Benchmark.ALFWORLD
                else frozenset()
            )
            exclusive_path_events = (
                ALFWORLD_EXCLUSIVE_PATH_EVENTS
                if self.snapshot.benchmark is Benchmark.ALFWORLD
                else frozenset()
            )
            retrieval = retrieve_task_high(
                query_result.vectors[0],
                self.snapshot.high_index,
                self.high_cards,
                self.graph_profile,
                required_path_events,
                exclusive_path_events,
                high_similarity_threshold=self.high_similarity_threshold,
                min_event_compatibility=self.min_event_compatibility,
                downweighted_skill_ids=self.downweighted_skill_ids,
                status_tie_epsilon=self.status_tie_epsilon,
            )
            high_card = retrieval.high_card
        else:
            matches = self.snapshot.high_index.search(
                query_result.vectors[0],
                1,
                score_penalties={
                    skill_id: self.status_tie_epsilon
                    for skill_id in self.downweighted_skill_ids
                    if skill_id in self.high_cards
                },
            )
            match = matches[0] if matches else None
            high_card = (
                self.high_cards[match.skill_id]
                if match
                and match.cosine_similarity >= self.high_similarity_threshold
                else None
            )
        return SkillSelection(
            (high_card.skill_id,) if high_card else (),
            format_tower_context(high_card, ()),
            query_result.usage.input_tokens,
            0,
        )

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        if self.mid_context_budget == 0 and self.low_top_k == 0:
            return SkillSelection((), "")

        observation = state.observation
        query_result = await self.runtime.embed([f"{task_goal}\n{observation}"])
        mid_index = self.snapshot.mid_index
        mid_cards = self.mid_cards
        if self.graph_profile:
            if self.snapshot.benchmark is Benchmark.ALFWORLD:
                allowed_events = alfworld_applicable_events(
                    state.admissible_actions,
                )
            else:
                allowed_events = webshop_applicable_events(
                    infer_webshop_page_type(observation)
                )
            retrieval = retrieve_tower_graph(
                (),
                query_result.vectors[0],
                SkillEmbeddingIndex((), ()),
                mid_index,
                {},
                mid_cards,
                {},
                self.graph_profile,
                allowed_events,
                mid_context_budget=self.mid_context_budget,
                direct_mid_similarity_threshold=self.direct_mid_similarity_threshold,
                direct_mid_dedup_similarity_threshold=(self.direct_mid_dedup_similarity_threshold),
                direct_mid_mmr_lambda=self.direct_mid_mmr_lambda,
                downweighted_skill_ids=self.downweighted_skill_ids,
                status_tie_epsilon=self.status_tie_epsilon,
                min_event_compatibility=self.min_event_compatibility,
            ).retrieval
        else:
            retrieval = retrieve_tower(
                (),
                query_result.vectors[0],
                self.snapshot.high_index,
                mid_index,
                self.high_cards,
                mid_cards,
                high_top_k=0,
                direct_mid_top_k=self.direct_mid_top_k,
                include_high_child_context=self.include_high_child_context,
                direct_mid_candidate_top_k=self.direct_mid_candidate_top_k,
                direct_mid_similarity_threshold=self.direct_mid_similarity_threshold,
                direct_mid_relative_margin=self.direct_mid_relative_margin,
                direct_mid_dedup_similarity_threshold=(self.direct_mid_dedup_similarity_threshold),
                direct_mid_mmr_lambda=self.direct_mid_mmr_lambda,
                downweighted_skill_ids=self.downweighted_skill_ids,
                status_tie_epsilon=self.status_tie_epsilon,
            )
        available_actions = (
            {
                primitive
                for action in state.admissible_actions
                if (primitive := parse_alfworld_action("take_action", {"action": action}))
                is not PrimitiveAction.INVALID
            }
            if self.snapshot.benchmark is Benchmark.ALFWORLD
            else {
                action
                for action, available in (
                    (PrimitiveAction.SEARCH, state.search_available),
                    (PrimitiveAction.CLICK, bool(state.admissible_actions)),
                )
                if available
            }
        )
        preferred_actions = tuple(
            dict.fromkeys(
                action
                for card in retrieval.mid_cards
                for action in card.grounding_actions
                if action in available_actions
            )
        )
        low_by_action = {skill.primitive_action: skill for skill in self.snapshot.low_skills}
        low_skills = tuple(
            low_by_action[action]
            for action in preferred_actions
            if action in low_by_action
        )[: self.low_top_k]
        low_ids = tuple(f"low:{skill.primitive_action.value}" for skill in low_skills)
        context = format_tower_context(None, retrieval.mid_cards, low_skills)
        return SkillSelection(
            skill_ids=(*retrieval.skill_ids, *low_ids),
            context=context,
            model_input_tokens=query_result.usage.input_tokens,
            model_output_tokens=0,
            context_skill_ids=(*retrieval.context_skill_ids, *low_ids),
        )
