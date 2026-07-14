from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import AlfworldTaskFamily
from trace2tower.methods.global_e2e.models import GlobalE2ESkillLibrary
from trace2tower.methods.global_e2e.retrieval import retrieve_global_e2e_skill


class GlobalE2ESkillProvider:
    def __init__(self, runtime: CommonLLMRuntime, library: GlobalE2ESkillLibrary):
        self.runtime = runtime
        self.library = library
        self.cards = {card.skill_id: card for card in library.cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        library_path: Path,
    ) -> GlobalE2ESkillProvider:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
        return cls(runtime, GlobalE2ESkillLibrary.from_record(payload))

    async def select(
        self,
        task_goal: str,
        initial_observation: str,
        task_family: AlfworldTaskFamily | None = None,
    ) -> SkillSelection:
        embedding = await self.runtime.embed([f"{task_goal}\n{initial_observation}"])
        retrieval = retrieve_global_e2e_skill(
            embedding.vectors[0],
            self.library.index,
            self.cards,
        )
        return SkillSelection(
            (retrieval.card.skill_id,),
            retrieval.context,
            embedding.usage.input_tokens,
            0,
        )
