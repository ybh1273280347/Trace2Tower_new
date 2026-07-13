from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.retrieval import retrieve_tower
from trace2tower.methods.trace2tower.tower import TowerSnapshot


class Trace2TowerSkillProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        *,
        high_similarity_threshold: float = -1.0,
        include_high_child_context: bool = True,
        direct_mid_candidate_top_k: int | None = None,
        direct_mid_similarity_threshold: float = 0.45,
        direct_mid_relative_margin: float = 0.08,
        direct_mid_dedup_similarity_threshold: float = 0.95,
        direct_mid_mmr_lambda: float = 0.75,
    ):
        snapshot.require_complete()
        if not -1 <= high_similarity_threshold <= 1:
            raise ValueError("High similarity threshold must be in [-1, 1]")
        if not isinstance(include_high_child_context, bool):
            raise ValueError("High child context switch must be boolean")
        self.runtime = runtime
        self.snapshot = snapshot
        self.high_similarity_threshold = high_similarity_threshold
        self.include_high_child_context = include_high_child_context
        self.direct_mid_candidate_top_k = direct_mid_candidate_top_k
        self.direct_mid_similarity_threshold = direct_mid_similarity_threshold
        self.direct_mid_relative_margin = direct_mid_relative_margin
        self.direct_mid_dedup_similarity_threshold = direct_mid_dedup_similarity_threshold
        self.direct_mid_mmr_lambda = direct_mid_mmr_lambda
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}
        self.mid_cards = {card.skill_id: card for card in snapshot.mid_cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        **kwargs,
    ) -> Trace2TowerSkillProvider:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return cls(
            runtime,
            TowerSnapshot.from_record(payload),
            **kwargs,
        )

    async def select(self, task_goal: str, initial_observation: str) -> SkillSelection:
        query_result = await self.runtime.embed(
            [task_goal, f"{task_goal}\n{initial_observation}"]
        )
        retrieval = retrieve_tower(
            query_result.vectors[0],
            query_result.vectors[1],
            self.snapshot.high_index,
            self.snapshot.mid_index,
            self.high_cards,
            self.mid_cards,
            high_top_k=self.snapshot.config.high_top_k,
            direct_mid_top_k=self.snapshot.config.direct_mid_top_k,
            high_similarity_threshold=self.high_similarity_threshold,
            include_high_child_context=self.include_high_child_context,
            direct_mid_candidate_top_k=self.direct_mid_candidate_top_k,
            direct_mid_similarity_threshold=self.direct_mid_similarity_threshold,
            direct_mid_relative_margin=self.direct_mid_relative_margin,
            direct_mid_dedup_similarity_threshold=self.direct_mid_dedup_similarity_threshold,
            direct_mid_mmr_lambda=self.direct_mid_mmr_lambda,
        )
        return SkillSelection(
            skill_ids=retrieval.skill_ids,
            context=retrieval.context,
            model_input_tokens=query_result.usage.input_tokens,
            model_output_tokens=0,
        )
