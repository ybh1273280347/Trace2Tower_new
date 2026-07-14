from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from trace2tower.llm_runtime import ChatResult, CommonLLMRuntime, ModelRole
from trace2tower.methods.flat_skill_summary.clustered_prompt import (
    CLUSTERED_FLAT_SKILL_PROMPT,
)
from trace2tower.methods.flat_skill_summary.models import CorpusFlatSkillCard
from trace2tower.trajectory import EpisodeTrajectory

CLUSTERED_FLAT_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "render_clustered_end_to_end_skill",
        "description": "Render one fixed task cluster as one standalone skill.",
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


async def render_clustered_flat_skill(
    runtime: CommonLLMRuntime,
    cluster_id: str,
    task_profiles: Sequence[Mapping],
    trajectories: Sequence[EpisodeTrajectory],
) -> tuple[CorpusFlatSkillCard, ChatResult]:
    evidence = {
        "cluster_id": cluster_id,
        "task_profiles": list(task_profiles),
        "trajectories": [trajectory.to_record() for trajectory in trajectories],
    }
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {"role": "system", "content": CLUSTERED_FLAT_SKILL_PROMPT},
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
        tools=[CLUSTERED_FLAT_SKILL_TOOL],
        tool_choice="required",
        temperature=0,
        max_output_tokens=5000,
        prompt_cache_key="flat-clustered:webshop:end-to-end:v1",
    )
    if (
        len(result.tool_calls) != 1
        or result.tool_calls[0].name != "render_clustered_end_to_end_skill"
    ):
        raise ValueError("clustered Flat renderer must return exactly one skill")
    try:
        payload = json.loads(result.tool_calls[0].arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("clustered Flat renderer returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "name",
        "description",
        "procedure",
        "constraints",
    }:
        raise ValueError("clustered Flat renderer returned fields outside the schema")

    name = _text(payload, "name")
    description = _text(payload, "description")
    procedure = _text_list(payload, "procedure")
    constraints = _text_list(payload, "constraints")
    card_payload = json.dumps(
        [cluster_id, name, description, procedure, constraints],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        CorpusFlatSkillCard(
            skill_id=f"flat_cluster_{hashlib.sha256(card_payload.encode()).hexdigest()[:12]}",
            supporting_trajectory_ids=tuple(
                sorted(trajectory.trajectory_id for trajectory in trajectories)
            ),
            name=name,
            description=description,
            procedure=procedure,
            constraints=constraints,
        ),
        result,
    )


def _text(payload: Mapping, field: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"clustered Flat field {field} must be non-empty text")
    return value.strip()


def _text_list(payload: Mapping, field: str) -> tuple[str, ...]:
    value = payload[field]
    if not isinstance(value, list) or not value:
        raise ValueError(f"clustered Flat field {field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"clustered Flat field {field} contains invalid text")
    return tuple(dict.fromkeys(item.strip() for item in value))
