from __future__ import annotations

import json
from pathlib import Path

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.agent import SkillSelection
from trace2tower.components.llm_runtime import CommonLLMRuntime
from trace2tower.methods.skillx.models import SkillXExecutionLibrary
from trace2tower.methods.skillx.native_inference import (
    SKILLX_COMMIT,
    NativeSkillCandidate,
    NativeSkillXInference,
    format_native_context,
)
from trace2tower.methods.skillx.retrieval import (
    SkillXRetrieval,
    plan_steps,
    retrieve_plans,
    retrieve_skills,
)


class SkillXProvider:
    """Adapter for the frozen SkillX main inference path."""

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
        llm_max_output_tokens: int,
        rewrite_plan: bool,
    ):
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("SkillX similarity threshold must be between zero and one")
        if min(plan_top_k, skills_per_step, max_skills) <= 0:
            raise ValueError("SkillX retrieval counts must be positive")
        if not isinstance(rewrite_plan, bool):
            raise ValueError("SkillX rewrite switch must be boolean")
        if library.skillx_commit != SKILLX_COMMIT:
            raise ValueError("SkillX library was not built by the frozen native commit")
        unknown_tools = {tool for skill in library.skills for tool in skill.tools} - allowed_tools
        if unknown_tools:
            raise ValueError(f"SkillX library references unavailable tools: {unknown_tools}")
        self.runtime = runtime
        self.library = library
        self.similarity_threshold = similarity_threshold
        self.plan_top_k = plan_top_k
        self.skills_per_step = skills_per_step
        self.max_skills = max_skills
        self.rewrite_plan = rewrite_plan
        self.plans = {plan.plan_id: plan for plan in library.plans}
        self.skills = {skill.skill_id: skill for skill in library.skills}
        self.inference = NativeSkillXInference(
            runtime,
            max_output_tokens=llm_max_output_tokens,
        )

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
    ) -> tuple[SkillXRetrieval, int | None, int | None]:
        task_embedding = await self.runtime.embed([task_goal])
        plans, plan_matches = retrieve_plans(
            task_embedding.vectors[0],
            self.library.plan_index,
            self.plans,
            top_k=self.plan_top_k,
            threshold=self.similarity_threshold,
        )
        source_plan = plans[0] if plans else None
        source_plan_match = plan_matches[0] if plan_matches else None
        injected_plan = source_plan.plan if source_plan else None
        rewrite_input_tokens: int | None = 0
        rewrite_output_tokens: int | None = 0
        if source_plan and self.rewrite_plan:
            rewrite = await self.inference.rewrite_plan(
                task=task_goal,
                reference_task=source_plan.task,
                reference_plan=source_plan.plan,
            )
            injected_plan = rewrite.plan
            rewrite_input_tokens = rewrite.input_tokens
            rewrite_output_tokens = rewrite.output_tokens

        if injected_plan:
            steps = plan_steps(injected_plan)
            skill_embedding = await self.runtime.embed(steps)
            skill_top_k = self.skills_per_step
        else:
            skill_embedding = await self.runtime.embed([task_goal])
            skill_top_k = self.max_skills * 2
        raw_skills, raw_matches = retrieve_skills(
            skill_embedding.vectors,
            self.library.skill_index,
            self.skills,
            top_k=skill_top_k,
            threshold=self.similarity_threshold,
        )
        candidates = tuple(_candidate(skill) for skill in raw_skills)
        selected = await self.inference.select_skills(
            task=task_goal,
            plan=injected_plan or task_goal,
            skills=candidates,
            max_skills=self.max_skills,
        )
        selected_ids = {candidate.skill_id for candidate in selected.skills}
        selected_skills = tuple(skill for skill in raw_skills if skill.skill_id in selected_ids)
        selected_matches = tuple(match for match in raw_matches if match.skill_id in selected_ids)
        retrieval = SkillXRetrieval(
            source_plan,
            source_plan_match,
            injected_plan,
            selected_skills,
            selected_matches,
            format_native_context(injected_plan, selected.skills),
        )
        return (
            retrieval,
            _sum_tokens(
                task_embedding.usage.input_tokens,
                rewrite_input_tokens,
                skill_embedding.usage.input_tokens,
                selected.input_tokens,
            ),
            _sum_tokens(rewrite_output_tokens, selected.output_tokens),
        )

    async def select(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        retrieval, input_tokens, output_tokens = await self.retrieve(task_goal)
        context_plan_ids = (
            (retrieval.source_plan.plan_id,)
            if retrieval.source_plan and retrieval.injected_plan
            else ()
        )
        context_skill_ids = context_plan_ids + tuple(skill.skill_id for skill in retrieval.skills)
        return SkillSelection(
            retrieval.skill_ids,
            retrieval.context,
            input_tokens,
            output_tokens,
            context_skill_ids,
        )


def _candidate(skill) -> NativeSkillCandidate:
    return NativeSkillCandidate(
        skill_id=skill.skill_id,
        name=skill.name,
        document=skill.document,
        content=skill.content,
    )


def _sum_tokens(*counts: int | None) -> int | None:
    return None if any(count is None for count in counts) else sum(counts)
