from __future__ import annotations

import json
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.skillx.models import SkillXExecutionLibrary
from trace2tower.methods.skillx.retrieval import (
    SkillXRetrieval,
    format_retrieval,
    plan_steps,
    retrieve_plan,
    retrieve_skills,
)


class SkillXProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        library: SkillXExecutionLibrary,
        *,
        allowed_tools: set[str],
        similarity_threshold: float,
        plan_top_k: int,
        skills_per_step: int,
        max_skills: int,
    ):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("SkillX similarity threshold must be between zero and one")
        if min(plan_top_k, skills_per_step, max_skills) <= 0:
            raise ValueError("SkillX retrieval counts must be positive")
        unknown_tools = {
            tool for skill in library.skills for tool in skill.tools
        } - allowed_tools
        if unknown_tools:
            raise ValueError(f"SkillX library references unavailable tools: {unknown_tools}")
        self.runtime = runtime
        self.library = library
        self.similarity_threshold = similarity_threshold
        self.plan_top_k = plan_top_k
        self.skills_per_step = skills_per_step
        self.max_skills = max_skills
        self.plans = {plan.plan_id: plan for plan in library.plans}
        self.skills = {skill.skill_id: skill for skill in library.skills}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        library_path: Path,
        **kwargs,
    ) -> SkillXProvider:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
        return cls(runtime, SkillXExecutionLibrary.from_record(payload), **kwargs)

    async def retrieve(
        self,
        task_goal: str,
    ) -> tuple[SkillXRetrieval, int | None]:
        plans = self.plans
        skills = self.skills
        plan_index = self.library.plan_index
        skill_index = self.library.skill_index
        task_embedding = await self.runtime.embed([task_goal])
        plan, plan_match = retrieve_plan(
            task_embedding.vectors[0],
            plan_index,
            plans,
            top_k=self.plan_top_k,
            threshold=self.similarity_threshold,
        )
        if plan:
            steps = plan_steps(plan.plan)
            step_embeddings = await self.runtime.embed(list(steps))
            query_vectors = step_embeddings.vectors
            input_tokens = _sum_tokens(
                task_embedding.usage.input_tokens,
                step_embeddings.usage.input_tokens,
            )
        else:
            query_vectors = task_embedding.vectors
            input_tokens = task_embedding.usage.input_tokens
        skills, skill_matches = retrieve_skills(
            query_vectors,
            skill_index,
            skills,
            top_k=self.skills_per_step if plan else self.max_skills * 2,
            max_skills=self.max_skills,
            threshold=self.similarity_threshold,
        )
        retrieval = SkillXRetrieval(
            plan,
            plan_match,
            skills,
            skill_matches,
            format_retrieval(plan, skills),
        )
        return retrieval, input_tokens

    async def select(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        retrieval, input_tokens = await self.retrieve(task_goal)
        return SkillSelection(
            retrieval.skill_ids,
            retrieval.context,
            input_tokens,
            0,
        )


def _sum_tokens(*counts: int | None) -> int | None:
    return None if any(count is None for count in counts) else sum(counts)
