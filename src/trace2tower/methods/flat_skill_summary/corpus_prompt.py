# ruff: noqa: E501
CORPUS_FLAT_SKILL_PROMPT = """You induce one flat collection of reusable skills from the complete corpus of successful benchmark trajectories supplied in the user message. This is global skill induction, not one-card-per-trajectory summarization and not hierarchical clustering.

Evidence contract:
- The input corpus is authoritative and contains every selected successful training trajectory in full structured form.
- Use evidence across trajectories to separate recurring decision patterns from product names, literal option values, ASINs, exact prices, and other episode-specific accidents.
- A successful trajectory proves completion of its own task, not that every action was necessary or that its concrete product category should become a skill category.
- Never invent an environment capability, observation field, tool, recovery path, or constraint that is absent from the corpus.
- supporting_trajectory_ids must contain 2 to 12 representative input IDs that directly support the skill. Use only exact IDs present in the corpus.

Flat collection contract:
- Produce 6 to 16 mutually distinct skills with no High/Mid/Low hierarchy, parent-child links, plan graph, or execution dependencies between cards.
- Each card must be independently retrievable from a new task goal and initial observation.
- Prefer transferable decision units such as query formulation, candidate screening, attribute verification, price checking, option selection, recovery from a mismatch, and purchase readiness when supported by the corpus.
- Do not create cards named after concrete product categories such as furniture, apparel, lighting, food, or cosmetics merely because those categories occur in the evidence.
- Merge semantically redundant behaviors. Split behaviors only when their applicability conditions and executable procedures genuinely differ.
- Do not write generic tool manuals. Explain when and how to make a benchmark decision, including correctness guards that the successful corpus demonstrates.

Card fields:
- name: short action-oriented label that states the reusable decision behavior.
- description: observable conditions under which the card applies. It must help retrieval discriminate this skill from the others.
- procedure: ordered imperative steps executable with the benchmark's available actions. Preserve necessary checks and ordering.
- constraints: concise guards that prevent premature or incorrect actions. Do not repeat the procedure verbatim.
- supporting_trajectory_ids: representative evidence IDs supporting the generalized behavior.

WebShop priorities:
- Preserve task-specified product attributes and the price ceiling as variables.
- Distinguish evidence visible in search results from evidence requiring product-page inspection.
- A plausible title is not proof that all requested attributes or selectable variants match.
- Purchase only after price, required attributes, and required options have been verified.
- Treat search, opening a candidate, inspecting details, selecting options, backtracking, and buying as different decisions even when several use the same click tool.

Output discipline:
- Call the supplied function exactly once and return no prose outside it.
- Return only the skills array and exactly the required fields.
- Before calling the function, silently remove category-specific cards, duplicate cards, unsupported advice, and tool-schema restatements.
"""
