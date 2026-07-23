from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex, SkillMatch
from trace2tower.methods.skillx.models import SkillXCard, SkillXPlan


@dataclass(frozen=True, slots=True)
class SkillXRetrieval:
    source_plan: SkillXPlan | None
    source_plan_match: SkillMatch | None
    injected_plan: str | None
    skills: tuple[SkillXCard, ...]
    skill_matches: tuple[SkillMatch, ...]
    context: str

    @property
    def skill_ids(self) -> tuple[str, ...]:
        plan_ids = (self.source_plan.plan_id,) if self.source_plan else ()
        return plan_ids + tuple(skill.skill_id for skill in self.skills)


def retrieve_plans(
    query_vector: Sequence[float],
    index: SkillEmbeddingIndex,
    plans: Mapping[str, SkillXPlan],
    *,
    top_k: int,
    threshold: float,
) -> tuple[tuple[SkillXPlan, ...], tuple[SkillMatch, ...]]:
    if set(index.skill_ids) != set(plans):
        raise ValueError("SkillX plan index and plan library differ")
    matches = tuple(
        match for match in index.search(query_vector, top_k) if match.cosine_similarity >= threshold
    )
    return tuple(plans[match.skill_id] for match in matches), matches


def retrieve_skills(
    query_vectors: Sequence[Sequence[float]],
    index: SkillEmbeddingIndex,
    skills: Mapping[str, SkillXCard],
    *,
    top_k: int,
    threshold: float,
) -> tuple[tuple[SkillXCard, ...], tuple[SkillMatch, ...]]:
    if set(index.skill_ids) != set(skills):
        raise ValueError("SkillX skill index and skill library differ")
    selected = []
    matches = []
    seen_names = set()
    for query_vector in query_vectors:
        for match in index.search(query_vector, top_k):
            skill = skills[match.skill_id]
            if match.cosine_similarity < threshold or skill.name in seen_names:
                continue
            seen_names.add(skill.name)
            selected.append(skill)
            matches.append(match)
    return tuple(selected), tuple(matches)


def plan_steps(plan: str) -> tuple[str, ...]:
    steps = tuple(
        line
        for raw_line in plan.splitlines()
        if (line := raw_line.strip())
        and (line.startswith("#") or line.startswith("step") or len(line) > 10)
        and len(line) >= 5
    )
    return steps or (plan,)
