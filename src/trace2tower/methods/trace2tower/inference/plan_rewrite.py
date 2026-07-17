from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.agent import SkillSelection
from trace2tower.components.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.methods.trace2tower.adapters.alfworld.plan_rewrite import (
    AlfworldPlanRewriteAdapter,
)
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.inference.formatting import format_tower_context

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
                "procedure": {"type": "array", "items": {"type": "string"}, "minItems": 2},
                "constraints": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "description", "procedure", "constraints"],
            "additionalProperties": False,
        },
    },
}


class PlanRewriteTrace2TowerProvider:
    """The frozen pre-cleanup ALFWorld plan-rewrite deployment contract."""

    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        *,
        reference_high_top_k: int,
        high_similarity_threshold: float,
        skills_per_step: int,
        max_mid_skills: int,
        mid_similarity_threshold: float,
        expose_reference_mid_evidence: bool,
        rewrite_model_role: ModelRole,
        rewrite_max_output_tokens: int,
        high_score_penalties: Mapping[str, float] | None = None,
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
        if rewrite_model_role is ModelRole.EMBEDDING:
            raise ValueError("plan rewrite requires a chat model role")
        penalties = dict(high_score_penalties or {})
        if set(penalties) - {card.skill_id for card in snapshot.high_cards} or any(
            penalty < 0 for penalty in penalties.values()
        ):
            raise ValueError("plan rewrite High score penalties are invalid")
        self.runtime = runtime
        self.snapshot = snapshot
        self.adapter = AlfworldPlanRewriteAdapter()
        self.reference_high_top_k = reference_high_top_k
        self.high_similarity_threshold = high_similarity_threshold
        self.skills_per_step = skills_per_step
        self.max_mid_skills = max_mid_skills
        self.mid_similarity_threshold = mid_similarity_threshold
        self.expose_reference_mid_evidence = expose_reference_mid_evidence
        self.rewrite_model_role = rewrite_model_role
        self.rewrite_max_output_tokens = rewrite_max_output_tokens
        self.high_score_penalties = penalties
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}
        self.mid_cards = {card.skill_id: card for card in snapshot.mid_cards}

    @classmethod
    def from_path(cls, runtime: CommonLLMRuntime, snapshot_path: Path, **kwargs):
        snapshot = TowerSnapshot.from_record(json.loads(snapshot_path.read_text(encoding="utf-8")))
        return cls(runtime, snapshot, **kwargs)

    async def select_task(self, task_goal: str, state: EnvironmentState) -> SkillSelection:
        task = self.adapter.task_text(task_goal, state)
        embedding = await self.runtime.embed([task])
        references = tuple(
            self.high_cards[match.skill_id]
            for match in self.snapshot.high_index.search(
                embedding.vectors[0],
                self.reference_high_top_k,
                score_penalties=self.high_score_penalties,
            )
            if match.cosine_similarity >= self.high_similarity_threshold
        )
        if not references:
            return SkillSelection((), "", embedding.usage.input_tokens, 0, ())
        rewritten, rewrite_input, rewrite_output = await self._rewrite(task, state, references)
        high_ids = tuple(card.skill_id for card in references)
        if self.max_mid_skills == 0:
            return SkillSelection(
                high_ids,
                format_tower_context(rewritten, ()),
                _sum_tokens(embedding.usage.input_tokens, rewrite_input),
                rewrite_output,
                high_ids,
            )
        step_embedding = await self.runtime.embed(list(rewritten.procedure))
        candidates = self._retrieve_mid_candidates(step_embedding.vectors)
        selected, select_input, select_output = await self._select_mid(task, rewritten, candidates)
        mid_ids = tuple(card.skill_id for card in selected)
        return SkillSelection(
            high_ids + mid_ids,
            format_tower_context(rewritten, selected),
            _sum_tokens(
                embedding.usage.input_tokens,
                rewrite_input,
                step_embedding.usage.input_tokens,
                select_input,
            ),
            _sum_tokens(rewrite_output, select_output),
            high_ids + mid_ids,
        )

    async def select_state(self, task_goal: str, state: EnvironmentState) -> SkillSelection:
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
            prompt_cache_key="trace2tower:plan-rewrite:alfworld:v3",
        )
        calls = tuple(call for call in result.tool_calls if call.name == "submit_task_plan")
        if len(calls) != 1:
            raise ValueError("plan rewrite must return exactly one task plan")
        payload = json.loads(calls[0].arguments)
        if set(payload) != {"name", "description", "procedure", "constraints"}:
            raise ValueError("plan rewrite returned unexpected fields")
        procedure = _normalize_procedure(payload["procedure"])
        constraints = tuple(str(item).strip() for item in payload["constraints"])
        if (
            len(procedure) < 2
            or any(not step for step in procedure)
            or any(not item for item in constraints)
        ):
            raise ValueError("plan rewrite returned an invalid plan")
        return (
            HighSkillCard(
                "runtime_rewritten_high",
                (),
                str(payload["name"]).strip(),
                str(payload["description"]).strip(),
                procedure,
                constraints,
            ),
            result.usage.input_tokens,
            result.usage.output_tokens,
        )

    def _retrieve_mid_candidates(self, vectors) -> tuple[MidSkillCard, ...]:
        selected = []
        seen = set()
        for vector in vectors:
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
        limit = min(self.max_mid_skills, len(candidates))
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
                            "maxItems": limit,
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
                {"role": "user", "content": _mid_selection_input(task, plan, candidates)},
            ],
            tools=[tool],
            tool_choice="required",
            temperature=0.0,
            max_output_tokens=400,
            prompt_cache_key="trace2tower:mid-self-filter:alfworld:v4",
        )
        calls = tuple(call for call in result.tool_calls if call.name == "select_supporting_skills")
        if len(calls) != 1:
            raise ValueError("Mid self-filter must return exactly one selection")
        payload = json.loads(calls[0].arguments)
        selected_ids = tuple(dict.fromkeys(str(item) for item in payload.get("skill_ids", ())))
        by_id = {card.skill_id: card for card in candidates}
        if (
            set(payload) != {"skill_ids"}
            or len(selected_ids) > self.max_mid_skills
            or not set(selected_ids) <= set(by_id)
        ):
            raise ValueError("Mid self-filter returned an invalid selection")
        return (
            tuple(by_id[item] for item in selected_ids),
            result.usage.input_tokens,
            result.usage.output_tokens,
        )


def _rewrite_input(task, state, references, mid_cards):
    sections = []
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
            mid_cards[mid_id] for mid_id in card.child_mid_ids if mid_id in mid_cards
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
        sections.append("\n".join(lines))
    actions = "\n".join(f"- {action}" for action in state.admissible_actions)
    return "\n\n".join(
        (
            f"# Current Task\n{task}",
            f"# Initial Observation\n{state.observation}",
            f"# Initially Available Actions\n{actions or '- None listed'}",
            "# Retrieved Reference Strategies\n\n" + "\n\n".join(sections),
        )
    )


def _mid_selection_input(task, plan, candidates):
    plan_steps = "\n".join(f"{index}. {step}" for index, step in enumerate(plan.procedure, 1))
    candidate_text = "\n\n".join(
        f"[{card.skill_id}] {card.name}\n{card.description}" for card in candidates
    )
    return "\n\n".join(
        (
            f"# Current Task\n{task}",
            f"# Rewritten Plan\n{plan_steps}",
            f"# Candidate Mid Skills\n{candidate_text}",
        )
    )


def _normalize_procedure(raw_steps):
    steps = tuple(str(step).strip() for step in raw_steps if str(step).strip())
    if len(steps) != 1:
        return steps
    lines = tuple(
        re.sub(r"^(?:#\s*)?(?:step\s*)?\d+[.):\-]\s*", "", line).strip()
        for line in steps[0].splitlines()
        if line.strip()
    )
    return lines if len(lines) >= 2 else steps


def _sum_tokens(*counts):
    return None if any(count is None for count in counts) else sum(counts)
