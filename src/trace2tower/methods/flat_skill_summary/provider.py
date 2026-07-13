from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.flat_skill_summary.models import FlatSkillLibrary
from trace2tower.methods.flat_skill_summary.retrieval import (
    retrieve_flat_skills,
    retrieve_flat_skills_legacy,
)


class FlatSkillProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        library: FlatSkillLibrary,
        *,
        candidate_top_k: int = 100,
        similarity_threshold: float = 0.45,
        relative_margin: float = 0.08,
        dedup_similarity_threshold: float = 0.95,
        mmr_lambda: float = 0.75,
        max_skills: int = 8,
        retrieval_strategy: str = "diverse",
    ):
        if retrieval_strategy not in {"legacy", "diverse"}:
            raise ValueError("unknown Flat retrieval strategy")
        self.runtime = runtime
        self.library = library
        self.candidate_top_k = candidate_top_k
        self.similarity_threshold = similarity_threshold
        self.relative_margin = relative_margin
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.mmr_lambda = mmr_lambda
        self.max_skills = max_skills
        self.retrieval_strategy = retrieval_strategy
        self.cards = {card.skill_id: card for card in library.cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        library_path: Path,
        **kwargs,
    ) -> FlatSkillProvider:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
        return cls(runtime, FlatSkillLibrary.from_record(payload), **kwargs)

    async def select(self, task_goal: str, initial_observation: str) -> SkillSelection:
        embedding = await self.runtime.embed([f"{task_goal}\n{initial_observation}"])
        retrieval = (
            retrieve_flat_skills_legacy(
                embedding.vectors[0], self.library.index, self.cards, self.max_skills
            )
            if self.retrieval_strategy == "legacy"
            else retrieve_flat_skills(
                embedding.vectors[0],
                self.library.index,
                self.cards,
                candidate_top_k=self.candidate_top_k,
                similarity_threshold=self.similarity_threshold,
                relative_margin=self.relative_margin,
                dedup_similarity_threshold=self.dedup_similarity_threshold,
                mmr_lambda=self.mmr_lambda,
                max_skills=self.max_skills,
            )
        )
        return SkillSelection(
            retrieval.skill_ids,
            retrieval.context,
            embedding.usage.input_tokens,
            0,
        )
