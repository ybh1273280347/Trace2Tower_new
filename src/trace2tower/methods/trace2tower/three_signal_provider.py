from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    alfworld_applicable_events,
    alfworld_goal_events,
)
from trace2tower.methods.trace2tower.provider import Trace2TowerSkillProvider
from trace2tower.methods.trace2tower.retrieval import format_tower_context
from trace2tower.methods.trace2tower.three_signal_retrieval import (
    MidTransitionSignalProfile,
    retrieve_mid_three_signal,
)


class ThreeSignalTrace2TowerSkillProvider:
    def __init__(
        self,
        base_provider: Trace2TowerSkillProvider,
        signal_profile: MidTransitionSignalProfile,
        *,
        mid_top_k: int,
        score_threshold: float,
    ):
        if base_provider.snapshot.benchmark is not Benchmark.ALFWORLD:
            raise ValueError("three-signal validation currently supports ALFWorld only")
        if base_provider.graph_profile is None:
            raise ValueError("three-signal provider requires an event graph profile")
        if not 1 <= mid_top_k <= len(base_provider.mid_cards):
            raise ValueError("three-signal Mid Top-K must fit the Mid library")
        self.base_provider = base_provider
        self.signal_profile = signal_profile
        self.mid_top_k = mid_top_k
        self.score_threshold = score_threshold

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        *,
        graph_profile_path: Path,
        signal_profile_path: Path,
        mid_top_k: int,
        score_threshold: float,
        high_similarity_threshold: float = -1.0,
        min_event_compatibility: float = 0.1,
    ) -> ThreeSignalTrace2TowerSkillProvider:
        base_provider = Trace2TowerSkillProvider.from_path(
            runtime,
            snapshot_path,
            graph_profile_path=graph_profile_path,
            mid_context_budget=0,
            low_top_k=0,
            high_similarity_threshold=high_similarity_threshold,
            min_event_compatibility=min_event_compatibility,
        )
        payload = json.loads(signal_profile_path.read_text(encoding="utf-8"))
        signal_profile = MidTransitionSignalProfile.from_record(
            payload.get("signal_profile", payload)
        )
        return cls(
            base_provider,
            signal_profile,
            mid_top_k=mid_top_k,
            score_threshold=score_threshold,
        )

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return await self.base_provider.select_task(task_goal, state)

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        query_result = await self.base_provider.runtime.embed(
            [f"{task_goal}\n{state.observation}"]
        )
        event_profile = self.base_provider.graph_profile
        allowed_events = alfworld_applicable_events(state.admissible_actions)
        required_events = alfworld_goal_events(task_goal)
        candidate_mid_ids = frozenset(
            mid_id
            for mid_id in self.base_provider.mid_cards
            if event_profile.compatibility(mid_id, allowed_events)
            >= self.base_provider.min_event_compatibility
            and frozenset(
                event
                for event in ALFWORLD_EXCLUSIVE_PATH_EVENTS
                if event_profile.compatibility(mid_id, frozenset((event,)))
                >= self.base_provider.min_event_compatibility
            )
            <= required_events
        )
        matches = retrieve_mid_three_signal(
            query_result.vectors[0],
            self.base_provider.snapshot.mid_index,
            candidate_mid_ids,
            self.signal_profile,
            top_k=self.mid_top_k,
            score_threshold=self.score_threshold,
        )
        cards = tuple(
            self.base_provider.mid_cards[match.skill_id] for match in matches
        )
        skill_ids = tuple(card.skill_id for card in cards)
        return SkillSelection(
            skill_ids,
            format_tower_context(None, cards),
            query_result.usage.input_tokens,
            0,
            skill_ids,
        )
