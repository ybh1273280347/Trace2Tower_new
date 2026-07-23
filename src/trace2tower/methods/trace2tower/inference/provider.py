from __future__ import annotations

import json
from pathlib import Path

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.agent import SkillSelection
from trace2tower.components.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.methods.skillx.native_inference import (
    NativeSkillCandidate,
    NativeSkillXInference,
    format_native_context,
)
from trace2tower.methods.skillx.retrieval import plan_steps
from trace2tower.methods.trace2tower.artifacts.tower import TowerSnapshot
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard, MidSkillCard


class HighToMidSkillProvider:
    """Apply the frozen SkillX inference path to Tower High and Mid cards."""

    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        *,
        reference_high_top_k: int,
        skills_per_step: int,
        max_mid_skills: int,
        mid_similarity_threshold: float,
        rewrite_model_role: ModelRole,
        rewrite_max_output_tokens: int,
        rewrite_plan: bool,
    ):
        snapshot.require_complete()
        if not 1 <= reference_high_top_k <= len(snapshot.high_cards):
            raise ValueError("reference High Top-K must fit the High library")
        if not 1 <= skills_per_step <= len(snapshot.mid_cards):
            raise ValueError("skills per step must fit the Mid library")
        if not 0 <= max_mid_skills <= len(snapshot.mid_cards):
            raise ValueError("maximum Mid skills must fit the Mid library")
        if not -1 <= mid_similarity_threshold <= 1:
            raise ValueError("Mid similarity threshold must be in [-1, 1]")
        if rewrite_model_role is not ModelRole.RENDERER:
            raise ValueError("SkillX-compatible postprocessing must use the renderer model")
        if not isinstance(rewrite_plan, bool):
            raise ValueError("High rewrite switch must be boolean")
        self.runtime = runtime
        self.snapshot = snapshot
        self.reference_high_top_k = reference_high_top_k
        self.skills_per_step = skills_per_step
        self.max_mid_skills = max_mid_skills
        self.mid_similarity_threshold = mid_similarity_threshold
        self.rewrite_plan = rewrite_plan
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}
        self.mid_cards = {card.skill_id: card for card in snapshot.mid_cards}
        self.inference = NativeSkillXInference(
            runtime,
            max_output_tokens=rewrite_max_output_tokens,
        )

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        **kwargs,
    ) -> HighToMidSkillProvider:
        snapshot = TowerSnapshot.from_record(json.loads(snapshot_path.read_text(encoding="utf-8")))
        return cls(runtime, snapshot, **kwargs)

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        task_embedding = await self.runtime.embed([task_goal])
        high_matches = self.snapshot.high_index.search(
            task_embedding.vectors[0], self.reference_high_top_k
        )
        if not high_matches:
            raise ValueError("complete Tower snapshot must provide a High skill")
        source_high = self.high_cards[high_matches[0].skill_id]
        source_plan = _high_reference_plan(source_high)
        rewrite_input_tokens: int | None = 0
        rewrite_output_tokens: int | None = 0
        injected_high_plan = source_plan
        if self.rewrite_plan:
            rewrite = await self.inference.rewrite_plan(
                task=task_goal,
                reference_task=_high_reference_task(source_high),
                reference_plan=source_plan,
            )
            injected_high_plan = rewrite.plan or source_plan
            rewrite_input_tokens = rewrite.input_tokens
            rewrite_output_tokens = rewrite.output_tokens

        selected_mid_cards: tuple[MidSkillCard, ...] = ()
        mid_embedding_tokens: int | None = 0
        selection_input_tokens: int | None = 0
        selection_output_tokens: int | None = 0
        if self.max_mid_skills:
            queries = plan_steps(injected_high_plan)
            mid_embedding = await self.runtime.embed(queries)
            mid_embedding_tokens = mid_embedding.usage.input_tokens
            raw_mid_cards = self._retrieve_mid_candidates(mid_embedding.vectors)
            candidates = tuple(_mid_candidate(card) for card in raw_mid_cards)
            selected = await self.inference.select_skills(
                task=task_goal,
                plan=injected_high_plan,
                skills=candidates,
                max_skills=self.max_mid_skills,
            )
            selected_ids = {candidate.skill_id for candidate in selected.skills}
            selected_mid_cards = tuple(
                card for card in raw_mid_cards if card.skill_id in selected_ids
            )
            selection_input_tokens = selected.input_tokens
            selection_output_tokens = selected.output_tokens

        source_high_ids = (source_high.skill_id,)
        mid_ids = tuple(card.skill_id for card in selected_mid_cards)
        skill_ids = source_high_ids + mid_ids
        return SkillSelection(
            skill_ids=skill_ids,
            context=format_native_context(
                injected_high_plan,
                tuple(_mid_candidate(card) for card in selected_mid_cards),
            ),
            model_input_tokens=_sum_tokens(
                task_embedding.usage.input_tokens,
                rewrite_input_tokens,
                mid_embedding_tokens,
                selection_input_tokens,
            ),
            model_output_tokens=_sum_tokens(
                rewrite_output_tokens,
                selection_output_tokens,
            ),
            context_skill_ids=source_high_ids + mid_ids,
        )

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return SkillSelection((), "", 0, 0, ())

    def _retrieve_mid_candidates(self, query_vectors) -> tuple[MidSkillCard, ...]:
        selected = []
        seen_names = set()
        for vector in query_vectors:
            for match in self.snapshot.mid_index.search(vector, self.skills_per_step):
                card = self.mid_cards[match.skill_id]
                if (
                    match.cosine_similarity < self.mid_similarity_threshold
                    or card.name in seen_names
                ):
                    continue
                seen_names.add(card.name)
                selected.append(card)
        return tuple(selected)


def _high_reference_task(card: HighSkillCard) -> str:
    return card.retrieval_condition or card.description or card.name


def _high_reference_plan(card: HighSkillCard) -> str:
    lines = [f"# step {index}: {step}" for index, step in enumerate(card.procedure, 1)]
    lines.extend(f"# constraint: {constraint}" for constraint in card.constraints)
    return "\n".join(lines)


def _mid_candidate(card: MidSkillCard) -> NativeSkillCandidate:
    content = "\n".join(
        (
            *(f"{index}. {step}" for index, step in enumerate(card.procedure, 1)),
            *(f"Constraint: {constraint}" for constraint in card.constraints),
        )
    )
    return NativeSkillCandidate(
        skill_id=card.skill_id,
        name=card.name,
        document=card.description,
        content=content,
    )


def _sum_tokens(*counts: int | None) -> int | None:
    return None if any(count is None for count in counts) else sum(counts)
