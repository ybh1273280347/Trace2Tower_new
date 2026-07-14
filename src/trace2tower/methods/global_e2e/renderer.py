from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from trace2tower.llm_runtime import ChatResult, CommonLLMRuntime, ModelRole
from trace2tower.methods.global_e2e.end_to_end_prompt import GLOBAL_E2E_SKILL_PROMPT
from trace2tower.methods.global_e2e.models import GlobalE2ESkillCard
from trace2tower.trajectory import EpisodeTrajectory


GLOBAL_E2E_TOOL = {
    "type": "function",
    "function": {
        "name": "induce_global_e2e_skills",
        "description": "Return standalone end-to-end skills induced from the corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "description": {"type": "string", "minLength": 1},
                            "procedure": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "minItems": 1,
                            },
                            "constraints": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "minItems": 1,
                            },
                            "supporting_trajectory_ids": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "minItems": 3,
                                "maxItems": 12,
                            },
                        },
                        "required": [
                            "name",
                            "description",
                            "procedure",
                            "constraints",
                            "supporting_trajectory_ids",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["skills"],
            "additionalProperties": False,
        },
    },
}


def format_trajectory_corpus(trajectories: Sequence[EpisodeTrajectory]) -> str:
    return json.dumps(
        {
            "trajectory_count": len(trajectories),
            "trajectories": [trajectory.to_record() for trajectory in trajectories],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


async def induce_global_e2e_skills(
    runtime: CommonLLMRuntime,
    trajectories: Sequence[EpisodeTrajectory],
) -> tuple[tuple[GlobalE2ESkillCard, ...], str, ChatResult]:
    corpus = format_trajectory_corpus(trajectories)
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {"role": "system", "content": GLOBAL_E2E_SKILL_PROMPT},
            {"role": "user", "content": corpus},
        ],
        tools=[GLOBAL_E2E_TOOL],
        tool_choice="required",
        temperature=0,
        max_output_tokens=12000,
        prompt_cache_key="global-e2e:webshop:v1",
    )
    if (
        len(result.tool_calls) != 1
        or result.tool_calls[0].name != "induce_global_e2e_skills"
    ):
        raise ValueError("Global E2E renderer must return exactly one collection")
    try:
        payload = json.loads(result.tool_calls[0].arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("Global E2E renderer returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"skills"}:
        raise ValueError("Global E2E renderer returned fields outside the schema")
    raw_skills = payload["skills"]
    if not isinstance(raw_skills, list) or not 3 <= len(raw_skills) <= 6:
        raise ValueError("Global E2E renderer must return 3 to 6 skills")

    known_ids = {trajectory.trajectory_id for trajectory in trajectories}
    cards = []
    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict) or set(raw_skill) != {
            "name",
            "description",
            "procedure",
            "constraints",
            "supporting_trajectory_ids",
        }:
            raise ValueError("Global E2E skill has invalid fields")
        name = _text(raw_skill, "name")
        description = _text(raw_skill, "description")
        procedure = _text_list(raw_skill, "procedure")
        constraints = _text_list(raw_skill, "constraints")
        supporting_ids = _text_list(raw_skill, "supporting_trajectory_ids")
        if not 3 <= len(supporting_ids) <= 12 or not set(supporting_ids) <= known_ids:
            raise ValueError("Global E2E skill references invalid evidence IDs")
        card_payload = json.dumps(
            [name, description, procedure, constraints],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cards.append(
            GlobalE2ESkillCard(
                skill_id=(
                    "global_e2e_"
                    f"{hashlib.sha256(card_payload.encode()).hexdigest()[:12]}"
                ),
                supporting_trajectory_ids=supporting_ids,
                name=name,
                description=description,
                procedure=procedure,
                constraints=constraints,
            )
        )
    if len({card.skill_id for card in cards}) != len(cards):
        raise ValueError("Global E2E renderer returned duplicate skills")
    return tuple(cards), corpus, result


def global_e2e_card_text(card: GlobalE2ESkillCard) -> str:
    return "\n".join((card.name, card.description, *card.procedure, *card.constraints))


def _text(payload: Mapping, field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Global E2E field {field} must be non-empty text")
    return value.strip()


def _text_list(payload: Mapping, field: str) -> tuple[str, ...]:
    value = payload[field]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Global E2E field {field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Global E2E field {field} contains invalid text")
    return tuple(dict.fromkeys(item.strip() for item in value))
