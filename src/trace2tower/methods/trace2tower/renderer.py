# ruff: noqa: E501
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from statistics import fmean
from typing import Any

from trace2tower.llm_runtime import ChatResult, CommonLLMRuntime, ModelRole
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.models import HighPath, PrimitiveAction
from trace2tower.methods.trace2tower.skills import (
    HighSkillCard,
    MidRenderInput,
    MidSkillCard,
    SegmentEvidence,
    legal_grounding_actions,
)


class RendererStyle(StrEnum):
    TRACE2TOWER = "trace2tower"
    TRACE2TOWER_DECISION_STATE = "trace2tower_decision_state"
    SKILLX = "skillx"

MID_RENDERER_INSTRUCTIONS = """You render one fixed Trace2Tower MidCluster into a concise reusable execution skill. A Mid skill is one stable behavior phase discovered by contrastive spectral decomposition. It is not an atomic action and not an end-to-end task strategy.

Ownership contract:
- The builder owns cluster_id, skill_id, member_segment_ids, support_count, trajectory IDs, scores, event types, primitive counts, and every structural relationship. Never return or propose any of these fields.
- You own only name, description, procedure, constraints, and grounding_actions. Return them through the single supplied function.
- Treat every member as evidence about one recurring behavior, but do not claim that every detail appears in every member. Prefer the stable operational intersection and generalize incidental values.
- A positive trajectory score supports the observed behavior. A lower score is still evidence about what happened, but must not be turned into a claimed success or recommendation without corroboration.

Evidence interpretation:
- goal states the episode objective and is context, not an instruction to copy verbatim.
- raw_actions are the actual environment actions. Preserve their operational meaning while generalizing incidental entity names.
- primitive_actions are deterministic Low-level labels assigned by code. They are the only possible grounding actions.
- observation_before and observation_after show local preconditions and effects. Do not infer hidden state, unavailable actions, or causal effects that are not visible.
- trajectory_score is the official final episode score. It is not a per-segment reward.
- event_type, when present, is a deterministic domain event label and may guide the behavioral name.
- support_count and primitive_action_distribution describe the complete cluster even when individual evidence items differ.
- successful_trajectory_contexts and unsuccessful_trajectory_contexts expose the complete event, Mid, and action sequence around representative local segments. Treat their contrast as first-class evidence: retain stable prerequisites, downstream effects, and completion checks from successful contexts; identify repeated actions, premature operations, missing prerequisites, and incomplete follow-through that are more characteristic of unsuccessful contexts.

Card requirements:
- name: a short action-oriented behavior label, preferably plain English and not an identifier. Do not include benchmark names, cluster numbers, support counts, or scores.
- description: one or two sentences stating the observable applicability conditions. Describe when to use the behavior, not why Trace2Tower discovered it.
- procedure: an ordered list of executable, environment-level steps. Each item must be an imperative instruction. Preserve necessary ordering from the local evidence and its position in complete successful trajectories. Do not mention Mid, Low, High, clusters, segments, trajectories, embeddings, renderers, or evidence.
- constraints: a short list of operational preconditions, checks, or failure guards supported by observations and actions. Do not invent universal rules. Do not repeat the procedure as constraints.
- grounding_actions: choose only from the enum exposed by the function schema. Include an action only when it is needed by the rendered procedure. An empty list is valid if the cluster contains no legal official primitive.

Generalization rules:
- Preserve goal-defining entity categories, attributes, quantities, and constraints when they are stable across positive members. Generalize a value into a functional role only when it actually varies across the positive evidence or is absent from the goals.
- Keep distinctions that change execution, prerequisites, observable state, or completion.
- Do not merge separate goals into a synthetic compound goal. If members show variants of one behavior, express the common behavior and place variant-specific conditions in constraints.
- Do not omit a stable prerequisite, state check, or completion check merely because it occurs immediately outside the local segment. Keep the card focused on the Mid behavior, but make it self-contained enough that another agent can apply it without rediscovering the required state.
- Remove incidental exploration, repeated actions, and failed steps from the recommended procedure. Convert recurring unsuccessful behavior into concise guards. A guard should state the observable condition and corrective action, not merely warn that failure is possible.
- Do not guarantee success, optimality, or availability. Require verification when successful evidence verifies something before acting.
- Do not expose raw IDs or quote long observations. The card will be injected into another agent prompt, so every token should help execution.

Output discipline:
- Use the supplied function exactly once.
- Return no prose outside the function call.
- Use non-empty strings and non-empty procedure and constraints lists.
- Keep the card compact enough for retrieval-time prompt injection.
- Before finalizing, silently check reusability, correct ordering, missing critical prerequisites, and whether the instructions are directly actionable by another agent.
- Do not return extra keys, IDs, membership, scores, support, explanations, confidence, citations, or alternative cards.
- Before calling the function, silently check that every grounding action is allowed by the schema and occurs in the supplied primitive_action_distribution.
"""


MID_BENCHMARK_INSTRUCTIONS = {
    Benchmark.ALFWORLD: """
ALFWorld execution semantics:
- GOTO changes location; PICK acquires an object; PUT places a held object; OPEN and CLOSE change receptacle state; TOGGLE uses an appliance or switch; HEAT, CLEAN, COOL, and SLICE transform an object; INVENTORY checks held items; EXAMINE and LOOK inspect state.
- Preserve visible prerequisites such as navigating before manipulating, opening a closed container before taking from it, holding an object before placing or transforming it, and using the required appliance or tool.
- Do not invent an object, receptacle, tool, or action string. Use generic functional roles when examples use different numbered entities.
""",
    Benchmark.WEBSHOP: """
WebShop execution semantics:
- SEARCH formulates or refines a keyword query; CLICK covers result navigation, candidate selection, option selection, attribute inspection, backtracking, and purchase controls.
- Distinguish these click purposes in procedure text even though they share one primitive label.
- Verify requested attributes and selectable variants before purchase when successful evidence does so. Do not claim a product satisfies an unobserved attribute.
- Partial reward can support a useful substep but does not establish a complete purchase strategy.
""",
}


DECISION_STATE_RENDERER_INSTRUCTIONS = """
WebShop decision-state rendering profile:
- Render a decision card, not a generic search/click tutorial. The card must
  bind the task's product category, identity, attributes, options, and price
  constraints to observable page evidence.
- Name the candidate discriminator: what visible title, category, price,
  option, or detail evidence makes a candidate acceptable or unacceptable.
- Treat evidence as supported, contradicted, or uncertain. Contradiction means
  reject and recover; uncertainty means perform one targeted verification when
  the evidence supports it; supported means advance or stop when the purchase
  gate is complete.
- Convert repeated failure behavior into a concrete recovery guard: change the
  query, return to results, reject the current candidate, correct an option, or
  stop. Do not emit a warning without the corrective action.
- Preserve exact product attributes and option values when they are stable in
  the evidence. Do not replace a concrete product behavior with generic words
  such as "target item" or "check details".
- A reusable card may cover one decision role (query, candidate, option,
  verification, recovery, or stop), but it must state its applicability and
  completion condition so it can be routed by the current state.
"""


WEBSHOP_HIGH_DECISION_TREE_INSTRUCTIONS = """
WebShop High strategy form:
- WebShop execution is a conditional policy, not a single linear event list.
  The procedure may contain explicit `IF condition -> action` and
  `ELSE/OTHERWISE -> recovery` items. Use this form whenever successful and
  unsuccessful trajectories diverge.
- Start with one objective-binding and query step, then branch on the observed
  candidate set. If no plausible candidate is present, reformulate or return
  to search; do not pretend that the original query succeeded.
- For each candidate, branch on category/identity, price, attributes, and
  required options: contradicted means reject and recover; uncertain means one
  targeted verification; supported means continue toward purchase.
- Include the positive stopping branch explicitly: when all required slots are
  supported and exact options are selected, purchase immediately. Do not force
  every task through detail tabs or a fixed number of candidates.
- Keep the strategy end to end, but express alternatives as guarded branches
  rather than flattening them into a generic `search -> inspect -> buy` list.
"""


MID_CONTRASTIVE_BOUNDARY_INSTRUCTIONS = """
This target was created by one fixed structural Split or coordinated Split/Merge transaction. Sibling profiles are supplied only to express the target's observable boundary; they are not members of the target and must not be merged into it.

- Make the target description and constraints distinguish it from siblings using differences supported by target evidence, such as query composition, observable page state, verification behavior, option handling, recovery condition, or action ordering.
- Do not manufacture a distinction from product names or incidental categories when the operational behavior is the same.
- Do not mention siblings, clusters, comparisons, scores, or evidence in the card.
- If the stable behavior is shared, keep shared steps concise and use the applicability condition or constraints for the defensible boundary.
"""


SKILLX_MID_ADAPTER_INSTRUCTIONS = """

Diagnostic adapter contract:
- The input is one fixed Trace2Tower Mid cluster. Its membership and mixed outcome evidence are immutable; do not discover, split, merge, filter, or relabel the cluster.
- Treat the cluster evidence as the trajectory and extract exactly one reusable skill for the recurring subtask shared by its segments.
- Preserve SkillX's generalization rules: use a generic action-oriented name, parameterize incidental values, keep the skill self-contained, exclude exploration and incorrect actions from the recommended procedure, and retain observed failure modes only as guards.
- Return the result through the supplied function instead of `<skill>` tags. Map SkillX `document` to description and constraints, `content` to an imperative procedure, and `tools` to grounding_actions.
- Return no IDs, scores, evidence metadata, or prose outside the function call.
"""


HIGH_RENDERER_INSTRUCTIONS = """You render one fixed Trace2Tower High structure into a compact task-level strategy. A High skill represents a successful composition of Mid-level behaviors and must guide an entire task from its initial state to its completion condition. It is injected once when the task starts, so it must never assume that an intermediate prerequisite has already been completed.

Ownership contract:
- The builder owns path_id, skill_id, ordered_mid_ids, child membership, support values, scores, and provenance. Never return or propose these fields.
- You own only name, description, procedure, and constraints. Return them through the supplied function.
- The mined Mid path is a structural anchor, not the output boundary. Use complete successful trajectories to restore required behavior before and after the mined path so the rendered strategy is end to end.
- Child Mid cards are local behavior descriptions. They may inform individual phases but must not determine the task-level scope.

Contrastive evidence contract:
- successful_examples contain complete trajectories that demonstrate the strategy or its structural anchor.
- unsuccessful_examples are selected because they attempt the same task instance or objective but may omit the path, enter a later phase early, bind the wrong entity, repeat actions, or stop before completion.
- Compare complete successful and unsuccessful sequences. Do not require a failure to contain the successful Mid path.
- Extract the earliest stable divergence that separates successful execution from failure. Convert it into an explicit prerequisite, ordering rule, progress check, completion check, or recovery guard.
- Absence is evidence: when failures consistently skip a behavior that successful trajectories perform, include the missing behavior in the strategy.
- Never copy a failed action sequence into the recommended procedure.

Card requirements:
- name: a short task-strategy label. Do not include IDs, benchmark names, support, scores, or hierarchy terminology.
- description: state the complete objective pattern for which the strategy applies. Do not phrase it as an intermediate-state condition such as already holding, already reaching, or already selecting something.
- procedure: an ordered end-to-end checklist beginning with objective binding or initial discovery and ending with the observable completion action or signal. Preserve stable Mid composition order while restoring prerequisites and suffixes from successful complete trajectories.
- constraints: concise failure guards and recovery rules justified by success/failure differences. State both the observable condition and the corrective action.

Composition rules:
- Bind the objective's required entities, quantities, attributes, transformations, and destination before acting when successful evidence does so.
- Treat task_condition as authoritative applicability evidence. Preserve its goal-defining entity category, attributes, quantities, and constraints; do not erase them into generic words such as target, product, object, attribute, or option. Parameterize only values that vary across successful examples, and never memorize opaque environment identifiers.
- Preserve dependencies across phases. Do not enter a later phase merely because its tool, location, control, or entity is visible.
- Generalize incidental values but keep distinctions that change prerequisites, order, verification, or completion.
- Remove exploration and repeated detours from the main procedure. Put necessary search and recovery discipline in constraints.
- Include progress tracking for repeated subgoals when successful trajectories require it.
- Do not stop at an intermediate state. End with the task's demonstrated completion action and require confirmation when evidence provides one.
- Do not invent operations, entities, attributes, tools, destinations, or hidden state absent from the supplied evidence.
- Do not mention training data, trajectories, failures, rewards, support, scores, paths, clusters, or rendering in the card.

Output discipline:
- Use the supplied function exactly once and return no prose outside it.
- Return only name, description, procedure, and constraints, all non-empty.
- Keep the card compact enough for one-time task prompt injection.
- Before calling the function, silently verify that the card starts from the initial task state, covers the full demonstrated chain, includes at least one contrast-derived guard, and ends at completion.
"""


HIGH_BENCHMARK_INSTRUCTIONS = {
    Benchmark.ALFWORLD: """
ALFWorld task-strategy semantics:
- Derive the ordered task chain from the evidence, typically locating the required object, acquiring it, applying every required state change, and placing or examining it at the destination.
- Keep the goal object bound across every phase. Do not treat an incidental object at an appliance or destination as the target.
- Treat available environment actions as authoritative state evidence. When a required action is unavailable, repair its missing prerequisite rather than repeating it or skipping ahead.
- Preserve possession, receptacle state, appliance/tool, quantity, and final-placement prerequisites demonstrated by successful trajectories.
- Do not waste the step budget revisiting checked locations or closing receptacles unless closure is required by the task or a later demonstrated action.
""",
    Benchmark.WEBSHOP: """
WebShop task-strategy semantics:
- Derive the complete chain from query formulation through candidate screening, product inspection, required option selection, final verification, and purchase.
- Keep every requested attribute bound across search and verification. Do not substitute title similarity for observed compliance.
- Treat visible page controls and selectable options as authoritative state evidence. Recover by revising the query, returning to the relevant page, or correcting options as demonstrated.
- Purchase only after the required product attributes and selected variants have been verified.
""",
}


SKILLX_HIGH_ADAPTER_INSTRUCTIONS = """

Diagnostic adapter contract:
- The input is one fixed Trace2Tower High path with immutable ordered Mid children and supporting examples. Do not discover, reorder, merge, split, filter, or score the path.
- Apply SkillX plan-writing rules to the fixed path: describe natural reusable sub-goals, remove incidental exploration or failed actions, preserve necessary order, list only demonstrated environment operations, and keep the plan concise.
- Return the plan through the supplied function instead of `<plan>` tags. Use name for a short plan label, description for applicability, and procedure for the ordered plan steps.
- Return no IDs, scores, evidence metadata, or prose outside the function call.
"""


def _mid_renderer_instructions(style: RendererStyle, benchmark: Benchmark) -> str:
    if style is RendererStyle.TRACE2TOWER:
        return MID_RENDERER_INSTRUCTIONS + MID_BENCHMARK_INSTRUCTIONS[benchmark]
    if style is RendererStyle.TRACE2TOWER_DECISION_STATE:
        return (
            MID_RENDERER_INSTRUCTIONS
            + MID_BENCHMARK_INSTRUCTIONS[benchmark]
            + (DECISION_STATE_RENDERER_INSTRUCTIONS if benchmark is Benchmark.WEBSHOP else "")
        )

    from third_party.SkillX.prompts.skill_prompts import FUNCTIONAL_SKILL_PROMPT

    return (
        FUNCTIONAL_SKILL_PROMPT
        + SKILLX_MID_ADAPTER_INSTRUCTIONS
        + MID_BENCHMARK_INSTRUCTIONS[benchmark]
    )


def _high_renderer_instructions(style: RendererStyle, benchmark: Benchmark) -> str:
    if style is RendererStyle.TRACE2TOWER:
        return HIGH_RENDERER_INSTRUCTIONS + HIGH_BENCHMARK_INSTRUCTIONS[benchmark]
    if style is RendererStyle.TRACE2TOWER_DECISION_STATE:
        return (
            HIGH_RENDERER_INSTRUCTIONS
            + HIGH_BENCHMARK_INSTRUCTIONS[benchmark]
            + (
                DECISION_STATE_RENDERER_INSTRUCTIONS
                + WEBSHOP_HIGH_DECISION_TREE_INSTRUCTIONS
                if benchmark is Benchmark.WEBSHOP
                else ""
            )
        )

    from third_party.SkillX.prompts.plan_prompts import PLAN_EXTRACTION_PROMPTS

    return (
        PLAN_EXTRACTION_PROMPTS["default"]
        + SKILLX_HIGH_ADAPTER_INSTRUCTIONS
        + HIGH_BENCHMARK_INSTRUCTIONS[benchmark]
    )


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
    sibling_inputs: Sequence[MidRenderInput] = (),
    *,
    trajectory_contexts: Mapping[str, Mapping[str, object]] = {},
    renderer_style: RendererStyle = RendererStyle.TRACE2TOWER,
) -> tuple[MidSkillCard, ChatResult]:
    legal_actions = legal_grounding_actions(benchmark, render_input)
    legal_action_values = sorted(action.value for action in legal_actions)
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
                "items": {"type": "string", "enum": legal_action_values},
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
                "content": (
                    _mid_renderer_instructions(renderer_style, benchmark)
                    + (MID_CONTRASTIVE_BOUNDARY_INSTRUCTIONS if sibling_inputs else "")
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    (
                        {
                            "target": _mid_render_profile(
                                render_input,
                                trajectory_contexts,
                            ),
                            "sibling_profiles": [
                                _mid_render_profile(item, trajectory_contexts)
                                for item in sibling_inputs
                            ],
                        }
                        if sibling_inputs
                        else _mid_render_profile(render_input, trajectory_contexts)
                    ),
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
        prompt_cache_key=(
            f"trace2tower:mid:{benchmark.value}:{renderer_style}:contrastive-v4"
            if sibling_inputs
            else f"trace2tower:mid:{benchmark.value}:{renderer_style}:v3"
        ),
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


def _mid_render_profile(
    render_input: MidRenderInput,
    trajectory_contexts: Mapping[str, Mapping[str, object]],
) -> dict:
    scores = [item.trajectory_score for item in render_input.segment_evidence]
    event_counts = Counter(
        item.event_type or "UNLABELED" for item in render_input.segment_evidence
    )
    examples = _representative_mid_examples(render_input, event_counts, limit=8)
    context_trajectory_ids = tuple(
        dict.fromkeys(item.trajectory_id for item in examples)
    )
    return {
        "cluster_id": render_input.cluster_id,
        "support_count": render_input.support_count,
        "mean_trajectory_score": fmean(scores),
        "full_score_ratio": sum(score >= 0.999 for score in scores) / len(scores),
        "primitive_action_distribution": render_input.primitive_action_distribution,
        "event_type_distribution": dict(sorted(event_counts.items())),
        "representative_examples": [
            {
                "goal": item.goal,
                "raw_actions": item.raw_actions,
                "observation_before": item.observation_before[:1200],
                "observation_after": item.observation_after[:1200],
                "trajectory_score": item.trajectory_score,
            }
            for item in examples
        ],
        "successful_trajectory_contexts": [
            trajectory_contexts[trajectory_id]
            for trajectory_id in context_trajectory_ids
            if trajectory_id in trajectory_contexts
            and float(trajectory_contexts[trajectory_id]["trajectory_score"]) >= 0.999
        ],
        "unsuccessful_trajectory_contexts": [
            trajectory_contexts[trajectory_id]
            for trajectory_id in context_trajectory_ids
            if trajectory_id in trajectory_contexts
            and float(trajectory_contexts[trajectory_id]["trajectory_score"]) < 0.999
        ],
    }


def _representative_mid_examples(
    render_input: MidRenderInput,
    event_counts: Counter[str],
    *,
    limit: int,
) -> list[SegmentEvidence]:
    if not render_input.segment_evidence or limit <= 0:
        return []

    total = len(render_input.segment_evidence)
    quotas = {
        event: min(count, int(count * limit / total))
        for event, count in event_counts.items()
    }
    remaining = min(limit, total) - sum(quotas.values())
    quota_order = sorted(
        event_counts,
        key=lambda event: (
            -(event_counts[event] * limit / total - quotas[event]),
            -event_counts[event],
            event,
        ),
    )
    for event in quota_order[:remaining]:
        quotas[event] += 1

    grouped = defaultdict(list)
    for item in render_input.segment_evidence:
        grouped[item.event_type or "UNLABELED"].append(item)

    selected = []
    for event in sorted(quotas, key=lambda value: (-quotas[value], value)):
        successful = sorted(
            (item for item in grouped[event] if item.trajectory_score >= 0.999),
            key=lambda item: (item.trajectory_id, item.segment_id),
        )
        unsuccessful = sorted(
            (item for item in grouped[event] if item.trajectory_score < 0.999),
            key=lambda item: (item.trajectory_id, item.segment_id),
        )
        candidates = []
        for index in range(max(len(successful), len(unsuccessful))):
            if index < len(successful):
                candidates.append(successful[index])
            if index < len(unsuccessful):
                candidates.append(unsuccessful[index])

        seen_trajectories = set()
        distinct = []
        repeated = []
        for item in candidates:
            if item.trajectory_id in seen_trajectories:
                repeated.append(item)
            else:
                distinct.append(item)
                seen_trajectories.add(item.trajectory_id)
        selected.extend((distinct + repeated)[: quotas[event]])
    return selected


async def render_high_card(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    path: HighPath,
    child_mid_cards: Mapping[str, MidSkillCard],
    supporting_examples: Sequence[Mapping[str, object]] = (),
    *,
    unsuccessful_examples: Sequence[Mapping[str, object]] = (),
    renderer_style: RendererStyle = RendererStyle.TRACE2TOWER,
) -> tuple[HighSkillCard, ChatResult]:
    missing = set(path.ordered_mid_ids) - set(child_mid_cards)
    if missing:
        raise ValueError(f"High path is missing child Mid cards: {sorted(missing)}")
    render_input = {
        "path_id": path.path_id,
        "task_condition": path.task_condition,
        "ordered_mid_ids": path.ordered_mid_ids,
        "child_mid_cards": [
            {
                "skill_id": child_mid_cards[mid_id].skill_id,
                "name": child_mid_cards[mid_id].name,
                "description": child_mid_cards[mid_id].description,
                "procedure": child_mid_cards[mid_id].procedure,
                "constraints": child_mid_cards[mid_id].constraints,
                "grounding_actions": child_mid_cards[mid_id].grounding_actions,
            }
            for mid_id in path.ordered_mid_ids
        ],
        "positive_support": path.positive_support,
        "negative_support": path.negative_support,
        "contrastive_path_score": path.contrastive_score,
        "supporting_trajectory_count": len(path.supporting_trajectory_ids),
        "successful_examples": list(supporting_examples),
        "unsuccessful_examples": list(unsuccessful_examples),
    }
    tool = _tool(
        "render_high_skill",
        "Render one end-to-end task strategy from the fixed Mid composition and contrastive trajectories.",
        {
            "name": _text_property("Short strategy name."),
            "description": _text_property("Complete objective pattern for this strategy."),
            "procedure": _string_array("End-to-end execution checklist."),
            "constraints": _string_array("Contrast-derived failure guards and recovery rules."),
        },
        ["name", "description", "procedure", "constraints"],
    )
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {
                "role": "system",
                "content": _high_renderer_instructions(renderer_style, benchmark),
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
        prompt_cache_key=f"trace2tower:high:{benchmark.value}:{renderer_style}:task-v7",
    )
    payload = _tool_payload(
        result,
        "render_high_skill",
        {"name", "description", "procedure", "constraints"},
    )
    return (
        HighSkillCard(
            skill_id=path.path_id,
            ordered_mid_ids=path.ordered_mid_ids,
            name=_required_text(payload, "name"),
            description=_required_text(payload, "description"),
            procedure=_required_text_tuple(payload, "procedure"),
            constraints=_required_text_tuple(payload, "constraints"),
            retrieval_condition=path.task_condition,
        ),
        result,
    )


async def render_task_conditioned_high_card(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    community_id: str,
    task_condition: Mapping[str, object],
    path: HighPath,
    parent_high_card: HighSkillCard,
    child_mid_cards: Mapping[str, MidSkillCard],
    member_mid_ids: Sequence[str],
    successful_examples: Sequence[Mapping[str, object]],
    unsuccessful_examples: Sequence[Mapping[str, object]],
) -> tuple[HighSkillCard, ChatResult]:
    missing = set(path.ordered_mid_ids) - set(child_mid_cards)
    if missing:
        raise ValueError(f"task-conditioned High is missing Mid cards: {sorted(missing)}")
    render_input = {
        "task_condition": task_condition,
        "parent_community_strategy": {
            "name": parent_high_card.name,
            "description": parent_high_card.description,
            "procedure": parent_high_card.procedure,
            "constraints": parent_high_card.constraints,
        },
        "ordered_mid_ids": path.ordered_mid_ids,
        "child_mid_cards": [
            {
                "skill_id": child_mid_cards[mid_id].skill_id,
                "name": child_mid_cards[mid_id].name,
                "description": child_mid_cards[mid_id].description,
                "procedure": child_mid_cards[mid_id].procedure,
                "constraints": child_mid_cards[mid_id].constraints,
                "grounding_actions": child_mid_cards[mid_id].grounding_actions,
            }
            for mid_id in path.ordered_mid_ids
        ],
        "positive_support": path.positive_support,
        "negative_support": path.negative_support,
        "contrastive_path_score": path.contrastive_score,
        "successful_complete_trajectories": list(successful_examples),
        "unsuccessful_complete_trajectories": list(unsuccessful_examples),
    }
    tool = _tool(
        "render_high_skill",
        "Render one task-conditioned end-to-end strategy from a fixed graph path.",
        {
            "name": _text_property("Short strategy name."),
            "description": _text_property("Complete concrete objective for this strategy."),
            "procedure": _string_array("End-to-end execution checklist."),
            "constraints": _string_array("Contrast-derived failure guards and recovery rules."),
        },
        ["name", "description", "procedure", "constraints"],
    )
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {
                "role": "system",
                "content": (
                    _high_renderer_instructions(RendererStyle.TRACE2TOWER, benchmark)
                    + """

Task-conditioned graph contract:
- The task condition is authoritative for target type, quantity, required transformation, device class, and final destination.
- The ordered Mid path is fixed graph evidence. Preserve its causal relations, but restore initial search, acquisition, repeated-object progress, and final completion when the mined path is only a structural backbone.
- Compare all supplied successful and unsuccessful complete trajectories for this task condition. Prefer relations repeated across successful variants over any one shortest trajectory.
- Preserve source receptacle categories that repeat across successful variants as an ordered search prior. Phrase them as likely candidates with fallback to other unvisited locations, never as a guaranteed fixed location.
- Instance numbers and one-off source locations are scene-specific and must not be copied.
- Name exact environment action forms only as conditional examples. Never prescribe a source receptacle merely because one successful trajectory used it.
- Produce one compact execution card suitable for one-time task-start injection.
"""
                ),
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
        max_output_tokens=1200,
        prompt_cache_key=f"trace2tower:high:{benchmark.value}:task-community-v3",
    )
    payload = _tool_payload(
        result,
        "render_high_skill",
        {"name", "description", "procedure", "constraints"},
    )
    return (
        HighSkillCard(
            community_id,
            path.ordered_mid_ids,
            _required_text(payload, "name"),
            _required_text(payload, "description"),
            _required_text_tuple(payload, "procedure"),
            _required_text_tuple(payload, "constraints"),
            tuple(member_mid_ids),
        ),
        result,
    )


async def render_high_community_card(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    community_id: str,
    mid_cards: Sequence[MidSkillCard],
    successful_contexts: Sequence[Mapping[str, object]],
    unsuccessful_contexts: Sequence[Mapping[str, object]],
    transition_summary: Mapping[str, object],
) -> tuple[HighSkillCard, ChatResult]:
    member_mid_ids = tuple(card.skill_id for card in mid_cards)
    render_input = {
        "community_mid_cards": [
            {
                "name": card.name,
                "description": card.description,
                "procedure": card.procedure,
                "constraints": card.constraints,
                "grounding_actions": card.grounding_actions,
            }
            for card in mid_cards
        ],
        "successful_complete_trajectories": list(successful_contexts),
        "unsuccessful_complete_trajectories": list(unsuccessful_contexts),
        "contrastive_transition_summary": transition_summary,
    }
    tool = _tool(
        "render_high_skill",
        "Render one end-to-end strategy shared by the supplied Mid transition community.",
        {
            "name": _text_property("Short strategy name."),
            "description": _text_property("Complete objective patterns covered."),
            "procedure": _string_array("End-to-end execution checklist."),
            "constraints": _string_array("Contrast-derived failure guards and recovery rules."),
        },
        ["name", "description", "procedure", "constraints"],
    )
    result = await runtime.chat(
        ModelRole.RENDERER,
        [
            {
                "role": "system",
                "content": (
                    _high_renderer_instructions(RendererStyle.TRACE2TOWER, benchmark)
                    + """

Community induction contract:
- The input is the complete Mid transition community, not one local path.
- Induce one strategy containing only relations stable across successful task variants.
- Use failed trajectories to remove brittle path-specific operations and add guards for target substitution, premature phase entry, repeated transformations, wasted search, and missing completion.
- Express conditional objective variants inside one shared strategy; do not serialize mutually exclusive transformations as one mandatory sequence.
- Prefer the shortest action allowed by current environment state. Repair a prerequisite only when the intended action is unavailable; do not add container or appliance operations merely because some trajectory used them.
"""
                ),
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
        max_output_tokens=1400,
        prompt_cache_key=f"trace2tower:high:{benchmark.value}:community-v1",
    )
    payload = _tool_payload(
        result,
        "render_high_skill",
        {"name", "description", "procedure", "constraints"},
    )
    return (
        HighSkillCard(
            community_id,
            (),
            _required_text(payload, "name"),
            _required_text(payload, "description"),
            _required_text_tuple(payload, "procedure"),
            _required_text_tuple(payload, "constraints"),
            member_mid_ids,
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
