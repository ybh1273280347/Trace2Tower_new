from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.retrieval import format_tower_context
from trace2tower.methods.trace2tower.task_conditioning import (
    DomainTaskAdapter,
    TaskCompatibility,
    TaskCondition,
    TaskConditionProfile,
    retrieve_task_conditioned_high,
)
from trace2tower.methods.trace2tower.tower import TowerSnapshot


class TaskConditionedHighProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        conditions: dict[str, TaskCondition],
        adapter: DomainTaskAdapter,
        *,
        minimum_compatibility: TaskCompatibility,
        similarity_threshold: float = -1.0,
    ):
        snapshot.require_complete()
        if set(conditions) != set(snapshot.high_index.skill_ids):
            raise ValueError("task-condition profile does not bind to the Tower")
        self.runtime = runtime
        self.snapshot = snapshot
        self.conditions = conditions
        self.adapter = adapter
        self.minimum_compatibility = minimum_compatibility
        self.similarity_threshold = similarity_threshold
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        profile_path: Path,
        adapter: DomainTaskAdapter,
        **kwargs,
    ) -> TaskConditionedHighProvider:
        snapshot = TowerSnapshot.from_record(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        profile = TaskConditionProfile.from_record(
            json.loads(profile_path.read_text(encoding="utf-8"))
        )
        if profile.domain != adapter.domain:
            raise ValueError("task-condition profile domain does not match the adapter")
        conditions = profile.by_skill_id
        return cls(runtime, snapshot, conditions, adapter, **kwargs)

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        query_condition = self.adapter.extract_query(task_goal, state)
        query = await self.runtime.embed([query_condition.retrieval_text])
        selected = retrieve_task_conditioned_high(
            query.vectors[0],
            self.snapshot.high_index,
            query_condition,
            self.conditions,
            self.adapter,
            minimum_compatibility=self.minimum_compatibility,
            similarity_threshold=self.similarity_threshold,
        )
        if selected is None:
            card = None
        else:
            source_card = self.high_cards[selected.semantic_match.skill_id]
            card = self.adapter.bind(
                source_card,
                query_condition,
                self.conditions[source_card.skill_id],
            )
        return SkillSelection(
            skill_ids=(card.skill_id,) if card else (),
            context=format_tower_context(card, ()),
            model_input_tokens=query.usage.input_tokens,
            model_output_tokens=0,
            context_skill_ids=(card.skill_id,) if card else (),
        )

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return SkillSelection((), "", 0, 0, ())
