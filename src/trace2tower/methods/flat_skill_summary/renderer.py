from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from trace2tower.llm_runtime import ChatResult, CommonLLMRuntime, ModelRole
from trace2tower.methods.flat_skill_summary.models import FlatSkillCard
from trace2tower.methods.flat_skill_summary.prompt import FLAT_SKILL_PROMPT
from trace2tower.trajectory import EpisodeTrajectory

FLAT_RENDER_TOOL = {
    "type": "function",
    "function": {
        "name": "render_flat_skill",
        "description": "Render one reusable flat skill from one successful trajectory.",
        "parameters": {
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
            },
            "required": ["name", "description", "procedure", "constraints"],
            "additionalProperties": False,
        },
    },
}


async def render_flat_skill(
    runtime: CommonLLMRuntime,
    trajectory: EpisodeTrajectory,
) -> tuple[FlatSkillCard, ChatResult]:
    if not trajectory.steps:
        raise ValueError("Flat renderer requires a non-empty trajectory")
    if any(
        current.observation != previous.next_observation
        for previous, current in zip(
            trajectory.steps,
            trajectory.steps[1:],
            strict=False,
        )
    ):
        raise ValueError("Flat renderer requires a contiguous observation chain")
    evidence = {
        "task_goal": trajectory.task_goal,
        "reward": trajectory.primary_score,
        "initial_observation": trajectory.steps[0].observation,
        "steps": [
            {
                "action_name": step.action_name,
                "action_arguments": step.action_arguments,
                "resulting_observation": step.next_observation,
                "valid_action": step.valid_action,
                "done": step.done,
            }
            for step in trajectory.steps
        ],
    }
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {"role": "system", "content": FLAT_SKILL_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        tools=[FLAT_RENDER_TOOL],
        tool_choice="required",
        temperature=0,
        max_output_tokens=1000,
        prompt_cache_key=f"flat-skill:{trajectory.benchmark.value}:compact-v1",
    )
    if len(result.tool_calls) != 1 or result.tool_calls[0].name != "render_flat_skill":
        raise ValueError("Flat renderer must call render_flat_skill exactly once")
    try:
        payload = json.loads(result.tool_calls[0].arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("Flat renderer returned invalid JSON") from exc
    allowed_fields = {"name", "description", "procedure", "constraints"}
    if not isinstance(payload, dict) or set(payload) != allowed_fields:
        raise ValueError("Flat renderer returned fields outside the fixed schema")
    return (
        FlatSkillCard(
            skill_id=f"flat_{hashlib.sha256(trajectory.trajectory_id.encode()).hexdigest()[:12]}",
            source_trajectory_id=trajectory.trajectory_id,
            name=_text(payload, "name"),
            description=_text(payload, "description"),
            procedure=_text_list(payload, "procedure"),
            constraints=_text_list(payload, "constraints"),
        ),
        result,
    )


def flat_card_text(card: FlatSkillCard) -> str:
    return "\n".join((card.name, card.description, *card.procedure, *card.constraints))


def _text(payload: Mapping, field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Flat renderer field {field} must be non-empty text")
    return value.strip()


def _text_list(payload: Mapping, field: str) -> tuple[str, ...]:
    value = payload[field]
    if not isinstance(value, list) or not value:
        raise ValueError(f"Flat renderer field {field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Flat renderer field {field} contains invalid text")
    return tuple(item.strip() for item in value)
