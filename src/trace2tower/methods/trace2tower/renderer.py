# ruff: noqa: E501
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from trace2tower.llm_runtime import ChatResult, CommonLLMRuntime, ModelRole
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.models import HighPath, PrimitiveAction
from trace2tower.methods.trace2tower.skills import (
    LOW_SKILLS,
    HighSkillCard,
    MidRenderInput,
    MidSkillCard,
    legal_grounding_actions,
)

MID_RENDERER_INSTRUCTIONS = """You render one fixed Trace2Tower MidCluster into a concise reusable skill card. The cluster structure and evidence are authoritative. Your task is textual compression, not discovery, clustering, correction, scoring, planning across clusters, or evaluation.

Ownership contract:
- The builder owns cluster_id, skill_id, member_segment_ids, support_count, trajectory IDs, scores, event types, primitive counts, and every structural relationship. Never return or propose any of these fields.
- You own only name, description, procedure, constraints, and grounding_actions. Return them through the single supplied function.
- Treat every member as evidence about one recurring behavior, but do not claim that every detail appears in every member. Prefer the stable intersection of the evidence. Specific objects, receptacles, attributes, brands, colors, sizes, prices, and destinations are examples unless they are genuinely invariant.
- A positive trajectory score supports the observed behavior. A lower score is still evidence about what happened, but must not be turned into a claimed success or recommendation without corroboration.

Evidence interpretation:
- goal states the episode objective and is context, not an instruction to copy verbatim.
- raw_actions are the actual environment actions. Preserve their operational meaning while generalizing incidental entity names.
- primitive_actions are deterministic Low-level labels assigned by code. They are the only possible grounding actions.
- observation_before and observation_after show local preconditions and effects. Do not infer hidden state, unavailable actions, or causal effects that are not visible.
- trajectory_score is the official final episode score. It is not a per-segment reward.
- event_type, when present, is a deterministic WebShop event label and may guide the behavioral name.
- support_count and primitive_action_distribution describe the complete cluster even when individual evidence items differ.

Card requirements:
- name: a short action-oriented behavior label, preferably plain English and not an identifier. Do not include benchmark names, cluster numbers, support counts, or scores.
- description: one or two sentences stating the observable applicability conditions. Describe when to use the behavior, not why Trace2Tower discovered it.
- procedure: an ordered list of executable, environment-level steps. Each item must be an imperative instruction. Preserve necessary ordering from the evidence. Do not mention Mid, Low, High, clusters, segments, trajectories, embeddings, renderers, or evidence.
- constraints: a short list of operational preconditions, checks, or failure guards supported by observations and actions. Do not invent universal rules. Do not repeat the procedure as constraints.
- grounding_actions: choose only from the enum exposed by the function schema. Include an action only when it is needed by the rendered procedure. An empty list is valid if the cluster contains no legal official primitive.

Generalization rules:
- Generalize names such as mug 1, fridge 1, ASINs, query strings, and option values into functional roles only when the role is evident.
- Keep domain distinctions that change execution. Opening a closed receptacle, selecting a product option, inspecting attributes, navigating back, heating, cooling, cleaning, slicing, and purchasing are not interchangeable.
- Do not merge separate goals into a synthetic compound goal. If members show variants of one behavior, express the common behavior and place variant-specific conditions in constraints.
- Do not add recovery steps unless failed or corrective evidence actually shows them.
- Do not guarantee success, optimality, availability, price compliance, or product fit. Require verification when the evidence verifies something before acting.
- Do not expose raw IDs or quote long observations. The card will be injected into another agent prompt, so every token should help execution.

ALFWorld grounding semantics:
- GOTO changes location; PICK acquires an object; PUT places a held object; OPEN and CLOSE change receptacle state; TOGGLE uses an appliance or switch; HEAT, CLEAN, COOL, and SLICE transform an object; INVENTORY checks held items; EXAMINE and LOOK inspect state.
- Preserve prerequisites visible in evidence, such as navigating before manipulating, opening a closed container before taking from it, holding an object before putting it, and using an appropriate appliance or tool.
- Do not invent an object, receptacle, tool, or action string. Use generic roles when cluster members differ.

WebShop grounding semantics:
- SEARCH formulates or refines a keyword query; CLICK covers result navigation, candidate selection, option selection, attribute inspection, backtracking, and purchase controls.
- Distinguish these click purposes in procedure text even though they share one primitive label.
- Verify requested attributes and selectable variants before purchase when the evidence does so. Do not claim a product satisfies an attribute that was not observed.
- Treat a full score and a partial score differently. Partial reward can reveal a useful substep but does not prove the entire purchase strategy was correct.

Output discipline:
- Use the supplied function exactly once.
- Return no prose outside the function call.
- Use non-empty strings and non-empty procedure and constraints lists.
- Keep the card compact enough for retrieval-time prompt injection.
- Do not return extra keys, IDs, membership, scores, support, explanations, confidence, citations, or alternative cards.
- Before calling the function, silently check that every grounding action is allowed by the schema and occurs in the supplied primitive_action_distribution.
"""


HIGH_RENDERER_INSTRUCTIONS = """You render one fixed Trace2Tower HighPath into concise strategy guidance. The builder has already mined and scored the path. Your task is to explain when and how to execute the existing ordered child Mid skills; you do not discover, edit, filter, reorder, merge, split, or score the path.

Ownership contract:
- The builder owns path_id, skill_id, ordered_mid_ids, child membership, positive_support, negative_support, contrastive_path_score, and supporting_trajectory_ids. Never return or propose these fields.
- You own only name, description, and procedure. Return them through the single supplied function.
- ordered_mid_ids is immutable. Every child must be represented exactly once and in the supplied order. Do not add a child, omit a child, reverse two children, or replace a child with a newly invented step sequence.
- Child Mid cards are the executable source of truth. Support statistics describe training evidence strength but do not authorize stronger factual claims.

Input interpretation:
- path_id is only a stable builder identifier and must not appear in prose.
- ordered_mid_ids specifies execution order and must not be copied into output.
- child_mid_cards contain each behavior's applicability, procedure, constraints, and Low grounding. Use their operational content without mentioning hierarchy.
- positive_support is the fraction of full-success training trajectories containing this path; negative_support is the corresponding fraction among other trajectories.
- contrastive_path_score ranks mined candidates. It is not a probability, confidence score, reward, or guarantee.
- supporting_trajectory_ids establish provenance only and must never appear in the card.

Card requirements:
- name: a short strategy label describing the end-to-end behavioral sequence. Do not include IDs, levels, benchmarks, support, scores, or words such as cluster and path.
- description: one or two sentences specifying observable applicability conditions for the full sequence. It must not claim that unrelated child goals form one task merely because they co-occurred.
- procedure: concise execution guidance that preserves child order. It may compress redundant wording between adjacent child cards, but it must retain each child's essential checks, transformations, and actions.

Composition rules:
- Preserve dependencies across children only when supported by their cards. A later child may consume state established by an earlier child, but do not invent that dependency.
- If adjacent child cards are variants or partially redundant, describe their ordered roles without silently deleting either one.
- If children appear weakly related, use neutral sequential language rather than inventing a causal story or shared objective.
- Carry forward safety checks and applicability constraints that materially affect execution.
- Do not introduce Low actions absent from child cards, new object types, new product attributes, new destinations, new tools, new prices, or hidden environment state.
- Do not guarantee success or optimality. High positive support and zero observed negative support are finite training evidence, not proof.
- Do not mention training data, trajectories, rewards, support ratios, contrastive scores, or rendering.

ALFWorld composition:
- Maintain environment preconditions such as navigation, open state, object possession, appliance use, and destination availability.
- Do not combine separate household objectives into a fabricated objective. When the children are merely sequential in observed behavior, describe the sequence literally and conservatively.
- Avoid copying numbered entity names from child examples unless a distinction is essential.

WebShop composition:
- Preserve the usual distinctions among query formulation, result screening, product inspection, option selection, verification, backtracking, and purchase.
- A purchase step must remain after relevant verification when that order is present.
- A partial-reward pattern must not be presented as a proven complete strategy.
- Do not turn a specific observed product into a general recommendation.

Output discipline:
- Use the supplied function exactly once and return no prose outside it.
- Return only name, description, and procedure, with non-empty strings and a non-empty procedure list.
- Keep the result compact for retrieval-time injection.
- Before calling the function, silently verify that procedure coverage follows ordered_mid_ids exactly and that no builder-owned field is returned.
"""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _text_property(description: str) -> dict:
    return {"type": "string", "description": description, "minLength": 1}


def _string_array(description: str, *, min_items: int = 1) -> dict:
    return {
        "type": "array",
        "description": description,
        "items": {"type": "string", "minLength": 1},
        "minItems": min_items,
    }


async def render_mid_card(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    render_input: MidRenderInput,
) -> tuple[MidSkillCard, ChatResult]:
    legal_actions = legal_grounding_actions(benchmark, render_input)
    official_actions = [skill.primitive_action.value for skill in LOW_SKILLS[benchmark]]
    tool = _tool(
        "render_mid_skill",
        "Render a reusable executable skill from the supplied fixed evidence.",
        {
            "name": _text_property("Short behavior name."),
            "description": _text_property("When the behavior applies."),
            "procedure": _string_array("Ordered executable steps."),
            "constraints": _string_array("Evidence-grounded cautions or preconditions."),
            "grounding_actions": {
                "type": "array",
                "items": {"type": "string", "enum": official_actions},
                "uniqueItems": True,
            },
        },
        ["name", "description", "procedure", "constraints", "grounding_actions"],
    )
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {
                "role": "system",
                "content": MID_RENDERER_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": json.dumps(
                    render_input.to_record(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        tools=[tool],
        tool_choice="required",
        temperature=0,
        max_output_tokens=1200,
        prompt_cache_key=f"trace2tower:mid:{benchmark.value}:v1",
    )
    payload = _tool_payload(
        result,
        "render_mid_skill",
        {"name", "description", "procedure", "constraints", "grounding_actions"},
    )
    grounding_actions = tuple(PrimitiveAction(value) for value in payload["grounding_actions"])
    if not set(grounding_actions) <= set(legal_actions):
        raise ValueError("renderer returned an action outside the cluster Low grounding")
    return (
        MidSkillCard(
            skill_id=render_input.cluster_id,
            member_segment_ids=render_input.member_segment_ids,
            name=_required_text(payload, "name"),
            description=_required_text(payload, "description"),
            procedure=_required_text_tuple(payload, "procedure"),
            constraints=_required_text_tuple(payload, "constraints"),
            grounding_actions=grounding_actions,
        ),
        result,
    )


async def render_high_card(
    runtime: CommonLLMRuntime,
    path: HighPath,
    child_mid_cards: Mapping[str, MidSkillCard],
) -> tuple[HighSkillCard, ChatResult]:
    missing = set(path.ordered_mid_ids) - set(child_mid_cards)
    if missing:
        raise ValueError(f"High path is missing child Mid cards: {sorted(missing)}")
    render_input = {
        "path_id": path.path_id,
        "ordered_mid_ids": path.ordered_mid_ids,
        "child_mid_cards": [
            child_mid_cards[mid_id].to_record() for mid_id in path.ordered_mid_ids
        ],
        "positive_support": path.positive_support,
        "negative_support": path.negative_support,
        "contrastive_path_score": path.contrastive_score,
        "supporting_trajectory_ids": path.supporting_trajectory_ids,
    }
    tool = _tool(
        "render_high_skill",
        "Render applicability and execution guidance for the fixed Mid-skill path.",
        {
            "name": _text_property("Short strategy name."),
            "description": _text_property("When the strategy applies."),
            "procedure": _string_array("Execution guidance preserving the supplied order."),
        },
        ["name", "description", "procedure"],
    )
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {
                "role": "system",
                "content": HIGH_RENDERER_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": json.dumps(
                    render_input,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ],
        tools=[tool],
        tool_choice="required",
        temperature=0,
        max_output_tokens=1000,
        prompt_cache_key="trace2tower:high:v1",
    )
    payload = _tool_payload(
        result, "render_high_skill", {"name", "description", "procedure"}
    )
    return (
        HighSkillCard(
            skill_id=path.path_id,
            ordered_mid_ids=path.ordered_mid_ids,
            name=_required_text(payload, "name"),
            description=_required_text(payload, "description"),
            procedure=_required_text_tuple(payload, "procedure"),
        ),
        result,
    )


def _tool_payload(
    result: ChatResult, expected_name: str, allowed_fields: set[str]
) -> dict[str, Any]:
    if len(result.tool_calls) != 1 or result.tool_calls[0].name != expected_name:
        raise ValueError(f"renderer must call {expected_name} exactly once")
    try:
        payload = json.loads(result.tool_calls[0].arguments)
    except json.JSONDecodeError as exc:
        raise ValueError("renderer returned invalid JSON arguments") from exc
    if not isinstance(payload, dict):
        raise ValueError("renderer arguments must be an object")
    unexpected_fields = set(payload) - allowed_fields
    if unexpected_fields:
        raise ValueError(f"renderer returned unexpected fields: {sorted(unexpected_fields)}")
    return payload


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"renderer field {field} must be non-empty text")
    return value.strip()


def _required_text_tuple(payload: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"renderer field {field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"renderer field {field} contains invalid text")
    return tuple(item.strip() for item in value)
