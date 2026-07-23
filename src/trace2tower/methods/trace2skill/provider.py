from __future__ import annotations

import json
from pathlib import Path

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.agent import SkillSelection
from trace2tower.methods.trace2skill.models import Trace2SkillArtifact


class Trace2SkillProvider:
    """Preload one consolidated Trace2Skill directory without test-time retrieval."""

    def __init__(self, artifact: Trace2SkillArtifact):
        self.artifact = artifact

    @classmethod
    def from_path(cls, path: Path) -> Trace2SkillProvider:
        return cls(Trace2SkillArtifact.from_record(json.loads(path.read_text(encoding="utf-8"))))

    async def select(self, task_goal: str, state: EnvironmentState) -> SkillSelection:
        return SkillSelection((self.artifact.artifact_id,), self.artifact.skill_markdown)
