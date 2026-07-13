from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trace2tower.methods.skillx.models import SkillXCard, SkillXPlan
from trace2tower.semantic_index import SkillEmbeddingIndex, SkillMatch


@dataclass(frozen=True, slots=True)
class SkillXRetrieval:
    plan: SkillXPlan | None
    plan_match: SkillMatch | None
    skills: tuple[SkillXCard, ...]
    skill_matches: tuple[SkillMatch, ...]
    context: str

    @property
    def skill_ids(self) -> tuple[str, ...]:
        plan_ids = (self.plan.plan_id,) if self.plan else ()
        return plan_ids + tuple(skill.skill_id for skill in self.skills)


def retrieve_plan(
    query_vector: Sequence[float],
    index: SkillEmbeddingIndex,
    plans: Mapping[str, SkillXPlan],
    *,
    top_k: int,
    threshold: float,
) -> tuple[SkillXPlan | None, SkillMatch | None]:
    if set(index.skill_ids) != set(plans):
        raise ValueError("SkillX plan index and plan library differ")
    matches = tuple(
        match
        for match in index.search(query_vector, top_k)
        if match.cosine_similarity >= threshold
    )
    if not matches:
        return None, None
    return plans[matches[0].skill_id], matches[0]


def retrieve_skills(
    query_vectors: Sequence[Sequence[float]],
    index: SkillEmbeddingIndex,
    skills: Mapping[str, SkillXCard],
    *,
    top_k: int,
    max_skills: int,
    threshold: float,
) -> tuple[tuple[SkillXCard, ...], tuple[SkillMatch, ...]]:
    if set(index.skill_ids) != set(skills):
        raise ValueError("SkillX skill index and skill library differ")
    selected = []
    matches = []
    seen = set()
    for query_vector in query_vectors:
        for match in index.search(query_vector, top_k):
            if match.cosine_similarity < threshold or match.skill_id in seen:
                continue
            seen.add(match.skill_id)
            selected.append(skills[match.skill_id])
            matches.append(match)
            if len(selected) == max_skills:
                return tuple(selected), tuple(matches)
    return tuple(selected), tuple(matches)


def plan_steps(plan: str) -> tuple[str, ...]:
    steps = tuple(
        line
        for raw_line in plan.splitlines()
        if (line := raw_line.strip())
        and (line.startswith("#") or line.startswith("step") or len(line) > 10)
    )
    return steps or (plan,)


def format_retrieval(
    plan: SkillXPlan | None,
    skills: Sequence[SkillXCard],
) -> str:
    sections = []
    if plan:
        sections.append(
            "## Reference Plan\n"
            f"{plan.plan}\n"
            "Adapt this plan to the current task and observation."
        )
    for skill in skills:
        sections.append(
            f"## Skill: {skill.name}\n"
            f"Description:\n{skill.document}\n\n"
            f"Reference implementation:\n{skill.content}\n\n"
            "Treat this as guidance and call only the provided benchmark tools."
        )
    return "\n\n".join(sections)
