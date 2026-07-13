from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.flat_skill_summary.models import FlatSkillLibrary
from trace2tower.methods.flat_skill_summary.retrieval import retrieve_flat_skills


class FlatSkillProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        library: FlatSkillLibrary,
        *,
        top_k: int = 3,
    ):
        if top_k != 3:
            raise ValueError("Flat Skill Summary uses fixed Top-3 retrieval")
        self.runtime = runtime
        self.library = library
        self.top_k = top_k
        self.cards = {card.skill_id: card for card in library.cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        library_path: Path,
        *,
        top_k: int = 3,
    ) -> FlatSkillProvider:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
        return cls(runtime, FlatSkillLibrary.from_record(payload), top_k=top_k)

    async def select(self, task_goal: str, initial_observation: str) -> SkillSelection:
        embedding = await self.runtime.embed([f"{task_goal}\n{initial_observation}"])
        retrieval = retrieve_flat_skills(
            embedding.vectors[0],
            self.library.index,
            self.cards,
            top_k=self.top_k,
        )
        return SkillSelection(
            retrieval.skill_ids,
            retrieval.context,
            embedding.usage.input_tokens,
            0,
        )
