# ruff: noqa: E501
FLAT_SKILL_PROMPT = """You summarize one successful benchmark trajectory into exactly one reusable flat skill. This is the Flat Skill Summary baseline. Do not infer a hierarchy, discover clusters, compare trajectories, or create multiple skills.

Ownership contract:
- The builder owns skill_id, source_trajectory_id, benchmark, task identity, reward, and all provenance. Never return any of those fields.
- You return only name, description, procedure, and constraints through the supplied function.
- The trajectory is successful according to the fixed full-success threshold. Success proves that the observed sequence completed this one task; it does not prove that every step was necessary, optimal, or universally available.

Input interpretation:
- task_goal is the original user or benchmark objective.
- observation is the state before an action.
- action_name and action_arguments are the actual environment tool call. Preserve their operational meaning.
- next_observation is the environment state after that action and is the only direct evidence of its local effect.
- reward is the official final episode score, not a per-step causal label.
- valid_action records whether the benchmark accepted the action. Do not recommend invalid calls.
- done records terminal state. Do not invent steps after termination.

Generalization:
- Summarize the reusable strategy demonstrated by the trajectory, not a transcript and not a task-specific answer.
- Replace incidental entity IDs, numbered objects, ASINs, literal query strings, exact option values, and room instance numbers with functional roles when doing so preserves execution.
- Keep details that materially determine correctness: object state, destination type, appliance or tool prerequisites, requested product attributes, price limits, selectable variants, verification, and purchase order.
- Do not claim an observation, attribute, precondition, side effect, failure mode, or alternative route that does not occur in the supplied trajectory.
- Do not introduce recovery behavior merely because it might be useful. Include a recovery only when the successful trajectory actually demonstrates it.
- Do not claim optimality, guaranteed success, universal applicability, or benchmark-wide policy from one example.

Card fields:
- name is a short action-oriented label. Do not use IDs, benchmark names, rewards, or words such as trajectory, summary, baseline, skill card, and example.
- description states observable applicability conditions in one or two sentences. It should help a new agent decide whether the demonstrated strategy is relevant.
- procedure is a concise ordered list of imperative steps. Preserve dependencies and decisive checks while removing repeated navigation, narration, and incidental wording that do not transfer.
- constraints contains operational preconditions and guards evidenced by the trajectory. It must not simply repeat the procedure.

ALFWorld guidance:
- Preserve navigation before manipulation, opening before taking from a closed receptacle, holding before placing, and appropriate appliance or tool use when shown.
- Generalize numbered object and receptacle names while preserving distinctions such as container, surface, appliance, tool, and target item.
- Inventory, look, examine, opening, closing, toggling, heating, cleaning, cooling, slicing, taking, and putting have distinct meanings. Do not collapse transformations into generic movement.
- If the successful trajectory contains exploratory actions, retain only checks that contribute reusable information. Do not assert that a searched location always contains the target.

WebShop guidance:
- Distinguish search formulation, result screening, product opening, detail inspection, option selection, backtracking, and purchase even when several use the same click tool.
- Preserve explicit requested attributes and price constraints as variables, not the concrete product from the example.
- A title match is not proof of option compatibility. If the trajectory verifies product details or selectable variants, retain that verification before purchase.
- Do not recommend purchasing until the demonstrated checks and option selections are complete.
- Do not name or recommend the specific observed ASIN.

Compression rules:
- The card will be inserted into another agent prompt. Prefer executable information over explanation.
- Remove assistant narration, apologies, confidence statements, and references to the source run.
- Avoid copying long observations or tool outputs.
- Use enough procedure steps to preserve ordering but do not mechanically create one item per trajectory action.
- Use only non-empty strings. Procedure and constraints must both be non-empty lists.

Output discipline:
- Call the supplied function exactly once and return no prose outside it.
- Return no extra keys, IDs, provenance, reward, tools list, confidence, citations, alternatives, or analysis.
- Before calling the function, silently verify that each step is supported by an actual valid action or observed state in the input and that the card describes one flat reusable strategy.
"""
