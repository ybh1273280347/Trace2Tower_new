from __future__ import annotations

from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.manifests import AlfworldTaskFamily


class ManualSkillProvider:
    def __init__(self, skill_id: str, context: str):
        if not skill_id or not context.strip():
            raise ValueError("manual skill ID and context must be non-empty")
        self.skill_id = skill_id
        self.context = context.strip()

    @classmethod
    def from_path(cls, skill_id: str, path: Path) -> ManualSkillProvider:
        return cls(skill_id, path.read_text(encoding="utf-8"))

    async def select(
        self,
        task_goal: str,
        initial_observation: str,
        task_family: AlfworldTaskFamily | None = None,
    ) -> SkillSelection:
        return SkillSelection((self.skill_id,), self.context)
