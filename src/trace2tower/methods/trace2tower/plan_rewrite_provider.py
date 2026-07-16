from __future__ import annotations

import json
import re
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.methods.trace2tower.retrieval import format_tower_context
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.task_conditioning import DomainTaskAdapter
from trace2tower.methods.trace2tower.tower import TowerSnapshot


_REWRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_task_plan",
        "description": "Submit the rewritten end-to-end plan for the current task.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "procedure": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "description", "procedure", "constraints"],
            "additionalProperties": False,
        },
    },
}


class PlanRewriteTrace2TowerProvider:
    """Retrieve graph plans, rewrite one task plan, then retrieve its Mid functions."""

    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        adapter: DomainTaskAdapter,
        *,
        reference_high_top_k: int,
        high_similarity_threshold: float,
        skills_per_step: int,
        max_mid_skills: int,
        mid_similarity_threshold: float,
        expose_reference_mid_evidence: bool,
        rewrite_model_role: ModelRole,
        rewrite_max_output_tokens: int,
    ):
        snapshot.require_complete()
        if not 1 <= reference_high_top_k <= len(snapshot.high_cards):
            raise ValueError("reference High Top-K must fit the High library")
        if not 1 <= skills_per_step <= len(snapshot.mid_cards):
            raise ValueError("skills per step must fit the Mid library")
        if not 0 <= max_mid_skills <= len(snapshot.mid_cards):
            raise ValueError("maximum Mid skills must fit the Mid library")
        if not -1 <= high_similarity_threshold <= 1:
            raise ValueError("High similarity threshold must be in [-1, 1]")
        if not -1 <= mid_similarity_threshold <= 1:
            raise ValueError("Mid similarity threshold must be in [-1, 1]")
        if rewrite_max_output_tokens <= 0:
            raise ValueError("rewrite output limit must be positive")
        if not isinstance(expose_reference_mid_evidence, bool):
            raise ValueError("reference Mid evidence switch must be boolean")
        if rewrite_model_role is ModelRole.EMBEDDING:
            raise ValueError("plan rewrite requires a chat model role")
        self.runtime = runtime
        self.snapshot = snapshot
        self.adapter = adapter
        self.reference_high_top_k = reference_high_top_k
        self.high_similarity_threshold = high_similarity_threshold
        self.skills_per_step = skills_per_step
        self.max_mid_skills = max_mid_skills
        self.mid_similarity_threshold = mid_similarity_threshold
        self.expose_reference_mid_evidence = expose_reference_mid_evidence
        self.rewrite_model_role = rewrite_model_role
        self.rewrite_max_output_tokens = rewrite_max_output_tokens
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}
        self.mid_cards = {card.skill_id: card for card in snapshot.mid_cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        adapter: DomainTaskAdapter,
        **kwargs,
    ) -> PlanRewriteTrace2TowerProvider:
        snapshot = TowerSnapshot.from_record(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        return cls(runtime, snapshot, adapter, **kwargs)

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        condition = self.adapter.extract_query(task_goal, state)
        task_embedding = await self.runtime.embed([condition.retrieval_text])
        high_matches = tuple(
            match
            for match in self.snapshot.high_index.search(
                task_embedding.vectors[0], self.reference_high_top_k
            )
            if match.cosine_similarity >= self.high_similarity_threshold
        )
        if not high_matches:
            return SkillSelection(
                (), "", task_embedding.usage.input_tokens, 0, ()
            )

        references = tuple(self.high_cards[match.skill_id] for match in high_matches)
        rewritten, rewrite_input_tokens, rewrite_output_tokens = await self._rewrite(
            condition.task_text,
            state,
            references,
        )
        source_high_ids = tuple(card.skill_id for card in references)
        if self.max_mid_skills == 0:
            return SkillSelection(
                skill_ids=source_high_ids,
                context=format_tower_context(rewritten, ()),
                model_input_tokens=_sum_tokens(
                    task_embedding.usage.input_tokens,
                    rewrite_input_tokens,
                ),
                model_output_tokens=rewrite_output_tokens,
                context_skill_ids=source_high_ids,
            )
        step_embeddings = await self.runtime.embed(list(rewritten.procedure))
        mid_candidates = self._retrieve_mid_candidates(step_embeddings.vectors)
        (
            selected_mid_cards,
            selection_input_tokens,
            selection_output_tokens,
        ) = await self._select_mid(condition.task_text, rewritten, mid_candidates)
        mid_ids = tuple(card.skill_id for card in selected_mid_cards)
        skill_ids = source_high_ids + mid_ids
        return SkillSelection(
            skill_ids=skill_ids,
            context=format_tower_context(rewritten, selected_mid_cards),
            model_input_tokens=_sum_tokens(
                task_embedding.usage.input_tokens,
                rewrite_input_tokens,
                step_embeddings.usage.input_tokens,
                selection_input_tokens,
            ),
            model_output_tokens=_sum_tokens(
                rewrite_output_tokens,
                selection_output_tokens,
            ),
            context_skill_ids=skill_ids,
        )

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return SkillSelection((), "", 0, 0, ())

    async def _rewrite(
        self,
        task: str,
        state: EnvironmentState,
        references: tuple[HighSkillCard, ...],
    ) -> tuple[HighSkillCard, int | None, int | None]:
        result = await self.runtime.chat(
            self.rewrite_model_role,
            [
                {
                    "role": "system",
                    "content": (
                        "Rewrite retrieved reference strategies into one compact, executable, "
                        "end-to-end plan for the current task. Reuse only relevant evidence, "
                        "reorder it when dependencies require, and bind it to concrete entities "
                        "visible in the task or initial state. Cover discovery, prerequisites, "
                        "the requested operation, delivery or completion, recovery checks, and "
                        "the stopping condition. Reference strategies are evidence, while the "
                        "current task and environment state are authoritative. Do not invent "
                        "entities, actions, tools, or task requirements. Use only actions that "
                        "the environment can support. Follow the domain plan semantics below and "
                        "return the plan with the required tool.\n"
                        f"{self.adapter.plan_rewrite_instructions}"
                    ),
                },
                {
                    "role": "user",
                    "content": _rewrite_input(
                        task,
                        state,
                        references,
                        self.mid_cards if self.expose_reference_mid_evidence else {},
                    ),
                },
            ],
            tools=[_REWRITE_TOOL],
            tool_choice="required",
            temperature=0.0,
            max_output_tokens=self.rewrite_max_output_tokens,
            prompt_cache_key=f"trace2tower:plan-rewrite:{self.adapter.domain}:v3",
        )
        calls = tuple(call for call in result.tool_calls if call.name == "submit_task_plan")
        if len(calls) != 1:
            raise ValueError("plan rewrite must return exactly one task plan")
        payload = json.loads(calls[0].arguments)
        expected_fields = {"name", "description", "procedure", "constraints"}
        if set(payload) != expected_fields:
            raise ValueError("plan rewrite returned unexpected fields")
        procedure = _normalize_procedure(payload["procedure"])
        constraints = tuple(str(item).strip() for item in payload["constraints"])
        if len(procedure) < 2 or any(not step for step in procedure):
            raise ValueError("plan rewrite returned an invalid procedure")
        if any(not item for item in constraints):
            raise ValueError("plan rewrite returned an invalid constraint")
        return (
            HighSkillCard(
                skill_id="runtime_rewritten_high",
                ordered_mid_ids=(),
                name=str(payload["name"]).strip(),
                description=str(payload["description"]).strip(),
                procedure=procedure,
                constraints=constraints,
            ),
            result.usage.input_tokens,
            result.usage.output_tokens,
        )

    def _retrieve_mid_candidates(self, query_vectors) -> tuple[MidSkillCard, ...]:
        selected = []
        seen = set()
        for vector in query_vectors:
            for match in self.snapshot.mid_index.search(vector, self.skills_per_step):
                if (
                    match.cosine_similarity < self.mid_similarity_threshold
                    or match.skill_id in seen
                ):
                    continue
                seen.add(match.skill_id)
                selected.append(self.mid_cards[match.skill_id])
        return tuple(selected)

    async def _select_mid(
        self,
        task: str,
        plan: HighSkillCard,
        candidates: tuple[MidSkillCard, ...],
    ) -> tuple[tuple[MidSkillCard, ...], int | None, int | None]:
        if not candidates:
            return (), 0, 0
        selection_limit = min(self.max_mid_skills, len(candidates))
        tool = {
            "type": "function",
            "function": {
                "name": "select_supporting_skills",
                "description": "Select only the Mid skills useful for the current plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [card.skill_id for card in candidates],
                            },
                            "minItems": 0,
                            "maxItems": selection_limit,
                            "uniqueItems": True,
                        }
                    },
                    "required": ["skill_ids"],
                    "additionalProperties": False,
                },
            },
        }
        result = await self.runtime.chat(
            self.rewrite_model_role,
            [
                {
                    "role": "system",
                    "content": (
                        "The task-specific High plan is already complete and has priority. Select "
                        "the smallest set of Mid skills that adds operational information absent "
                        "from that plan. A selected Mid must contribute at least one new concrete "
                        "action pattern, observable precondition, recovery rule, or completion "
                        "check. Repeating the plan's search, verification, selection, transport, "
                        "or completion steps at the same level of specificity is not useful. "
                        "Reject skills tied to a wrong object, operation, destination, state "
                        "change, or option. Return at most the requested number of IDs with the "
                        "required tool; an empty list is the normal answer when the High plan is "
                        "already sufficient."
                    ),
                },
                {
                    "role": "user",
                    "content": _mid_selection_input(task, plan, candidates),
                },
            ],
            tools=[tool],
            tool_choice="required",
            temperature=0.0,
            max_output_tokens=400,
            prompt_cache_key=f"trace2tower:mid-self-filter:{self.adapter.domain}:v4",
        )
        calls = tuple(
            call for call in result.tool_calls if call.name == "select_supporting_skills"
        )
        if len(calls) != 1:
            raise ValueError("Mid self-filter must return exactly one selection")
        payload = json.loads(calls[0].arguments)
        if set(payload) != {"skill_ids"}:
            raise ValueError("Mid self-filter returned unexpected fields")
        selected_ids = tuple(dict.fromkeys(str(item) for item in payload["skill_ids"]))
        candidates_by_id = {card.skill_id: card for card in candidates}
        if (
            len(selected_ids) > self.max_mid_skills
            or not set(selected_ids) <= set(candidates_by_id)
        ):
            raise ValueError("Mid self-filter returned an invalid selection")
        return (
            tuple(candidates_by_id[skill_id] for skill_id in selected_ids),
            result.usage.input_tokens,
            result.usage.output_tokens,
        )


class PlanRewriteWithDynamicMidProvider:
    def __init__(self, task_provider, state_provider):
        self.task_provider = task_provider
        self.state_provider = state_provider

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return await self.task_provider.select_task(task_goal, state)

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return await self.state_provider.select_state(task_goal, state)


def _rewrite_input(
    task: str,
    state: EnvironmentState,
    references: tuple[HighSkillCard, ...],
    mid_cards: dict[str, MidSkillCard],
) -> str:
    reference_sections = []
    for index, card in enumerate(references, 1):
        lines = [
            f"### Reference {index}: {card.name}",
            f"Use when: {card.description}",
            "Procedure:",
            *(f"{step_index}. {step}" for step_index, step in enumerate(card.procedure, 1)),
            "Constraints:",
            *(f"- {item}" for item in card.constraints),
        ]
        child_cards = tuple(
            mid_cards[mid_id]
            for mid_id in card.child_mid_ids
            if mid_id in mid_cards
        )
        if child_cards:
            lines.append("Ordered graph-function evidence:")
            for child_index, child in enumerate(child_cards, 1):
                lines.extend(
                    (
                        f"{child_index}. {child.name}: {child.description}",
                        *(f"   - {step}" for step in child.procedure),
                    )
                )
        reference_sections.append("\n".join(lines))
    actions = "\n".join(f"- {action}" for action in state.admissible_actions)
    return (
        f"# Current Task\n{task}\n\n"
        f"# Initial Observation\n{state.observation}\n\n"
        f"# Initially Available Actions\n{actions or '- None listed'}\n\n"
        f"# Retrieved Reference Strategies\n\n" + "\n\n".join(reference_sections)
    )


def _mid_selection_input(
    task: str,
    plan: HighSkillCard,
    candidates: tuple[MidSkillCard, ...],
) -> str:
    plan_steps = "\n".join(
        f"{index}. {step}" for index, step in enumerate(plan.procedure, 1)
    )
    candidate_text = "\n\n".join(
        f"[{card.skill_id}] {card.name}\n{card.description}"
        for card in candidates
    )
    return (
        f"# Current Task\n{task}\n\n"
        f"# Rewritten Plan\n{plan_steps}\n\n"
        f"# Candidate Mid Skills\n{candidate_text}"
    )


def _sum_tokens(*counts: int | None) -> int | None:
    return None if any(count is None for count in counts) else sum(counts)


def _normalize_procedure(raw_steps) -> tuple[str, ...]:
    steps = tuple(str(step).strip() for step in raw_steps if str(step).strip())
    if len(steps) != 1:
        return steps
    lines = tuple(
        re.sub(r"^(?:#\s*)?(?:step\s*)?\d+[.):\-]\s*", "", line).strip()
        for line in steps[0].splitlines()
        if line.strip()
    )
    return lines if len(lines) >= 2 else steps
