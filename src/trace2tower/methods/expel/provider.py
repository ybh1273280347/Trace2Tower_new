from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.agent import SkillSelection
from trace2tower.components.llm_runtime import CommonLLMRuntime
from trace2tower.methods.expel.models import EXPEL_COMMIT, ExpeLEpisode, ExpeLExecutionLibrary
from trace2tower.methods.expel.retrieval import task_scope


class ExpeLProvider:
    """复现 ExpeL 的全局规则与相似成功轨迹联合注入合同。"""

    def __init__(
        self,
        runtime: CommonLLMRuntime,
        library: ExpeLExecutionLibrary,
        *,
        episode_top_k: int,
    ):
        if episode_top_k <= 0:
            raise ValueError("ExpeL episode retrieval count must be positive")
        if library.expel_commit != EXPEL_COMMIT:
            raise ValueError("ExpeL library was not built from the frozen native commit")
        self.runtime = runtime
        self.library = library
        self.episode_top_k = episode_top_k
        self.episodes = {episode.episode_id: episode for episode in library.episodes}
        self.rule_ids = tuple(
            f"expel_rule_{hashlib.sha256(rule.encode()).hexdigest()[:12]}"
            for rule in library.rules
        )
        scopes = {episode.task_scope for episode in library.episodes}
        self.scope_indices = {
            scope: library.episode_index.subset(
                {
                    episode.episode_id
                    for episode in library.episodes
                    if episode.task_scope == scope
                }
            )
            for scope in scopes
        }

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        path: Path,
        *,
        episode_top_k: int,
    ) -> ExpeLProvider:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(runtime, ExpeLExecutionLibrary.from_record(payload), episode_top_k=episode_top_k)

    async def select(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        embedding = await self.runtime.embed([task_goal])
        index = self.scope_indices.get(task_scope(self.library.benchmark, task_goal))
        matches = index.search(embedding.vectors[0], self.episode_top_k) if index else ()
        episodes = tuple(self.episodes[match.skill_id] for match in matches)
        context = self._context(episodes)
        return SkillSelection(
            (*self.rule_ids, *(episode.episode_id for episode in episodes)),
            context,
            embedding.usage.input_tokens,
            embedding.usage.output_tokens,
        )

    def _context(self, episodes: tuple[ExpeLEpisode, ...]) -> str:
        rules = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(self.library.rules, 1)
        )
        sections = [
            "# ExpeL Global Insights\n\n"
            "Use these accumulated cross-task rules as references. Current observations and "
            "available actions remain authoritative.\n\n"
            f"{rules}"
        ]
        if episodes:
            memories = "\n\n".join(
                f"## Similar Successful Experience {index}\n\n{episode.trajectory}"
                for index, episode in enumerate(episodes, 1)
            )
            sections.append(
                "# ExpeL Retrieved Successful Experiences\n\n"
                "Use these task-similar successful trajectories as demonstrations, adapting "
                "object names and currently available actions to the present task.\n\n"
                f"{memories}"
            )
        return "\n\n".join(sections)
