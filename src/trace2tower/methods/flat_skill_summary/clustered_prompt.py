# ruff: noqa: E501
CLUSTERED_FLAT_SKILL_PROMPT = """You render one fixed cluster of successful WebShop tasks and trajectories into exactly one standalone, high-level, end-to-end skill. The builder owns clustering and provenance. You must not split, merge, reassign, reject, or rank cluster members.

Skill boundary:
- The output is one flat skill, not a hierarchy, plan graph, set of atomic cards, tool manual, or transcript summary.
- It must independently guide an agent from the initial WebShop search page to a correct Buy Now decision without relying on another skill.
- Its applicability must be decidable from the task goal and initial search-page observation. Do not use future conditions such as "when the first search fails," "when the result page is noisy," or "when the product lacks an option" as the primary applicability rule.
- Infer the goal-visible requirement pattern shared by the cluster, such as explicit selectable variants, hidden descriptive properties requiring evidence, exact identity terms, or a combination. Do not name concrete product categories merely because they occur in the cluster.

Evidence contract:
- The user payload contains the fixed task profiles and every successful trajectory assigned to this cluster.
- Generalize recurring behavior while keeping task product type, attributes, options, quantity, size, and budget as variables.
- A successful trajectory proves completion of its own task, not that every exploratory step should be repeated.
- Do not invent controls, evidence sources, recovery paths, or facts absent from the cluster.

Required end-to-end behavior:
- Parse every mandatory constraint from the goal before acting.
- Form a concise search query from product identity and the most discriminative goal-visible terms.
- Screen result titles and visible prices, then open the strongest candidate.
- Recheck product identity and price on the product page.
- Select and confirm every explicitly required option when present.
- Inspect Description, Features, or Attributes only for unresolved properties that need evidence; use `< prev` to return to the same product page.
- Use `Back to Search` only to reject a candidate, then try another result or a refined query.
- Stop exploring and click Buy Now immediately once identity, price, required evidence, and required options are jointly satisfied.

Output fields:
- name: a short label for the complete goal-visible task pattern.
- description: discriminative applicability conditions visible in the initial goal. It must support top-1 retrieval against other cluster skills.
- procedure: an ordered standalone algorithm ending in Buy Now.
- constraints: correctness and navigation guards that prevent premature purchase and unproductive loops.

Output discipline:
- Call the supplied function exactly once and return no prose outside it.
- Return exactly name, description, procedure, and constraints.
- Do not return cluster IDs, trajectory IDs, evidence, scores, tools, confidence, or analysis; the builder owns those fields.
"""
