# ruff: noqa: E501
END_TO_END_FLAT_SKILL_PROMPT = """You induce a small flat collection of high-level, end-to-end skills from the complete corpus of successful WebShop trajectories supplied in the user message.

Core contract:
- Produce 3 to 6 independently usable skills. Each skill must begin from the WebShop search page and end with a correct Buy Now decision.
- A skill must never require another skill, hidden plan, hierarchy, parent card, or separately retrieved atomic instruction.
- Every card must contain a complete executable loop: parse the goal, formulate a search, screen candidates, open a product, verify price and required evidence, select required options, recover from a mismatch when necessary, and purchase only when ready.
- The cards may differ only by a meaningful end-to-end task pattern supported by the corpus, such as tasks resolved by visible product evidence, tasks requiring selectable variants, or tasks requiring Description/Features/Attributes inspection.
- Do not create separate skills for search, price checking, option selection, pagination, backtracking, or purchase. Those are stages inside every relevant end-to-end card.

Evidence and generalization:
- The input contains every selected successful trajectory in full structured form. Use recurring evidence across trajectories and remove product-category accidents.
- Do not create cards named after furniture, apparel, lighting, food, cosmetics, or other concrete product categories.
- Keep the product type, requested attributes, option values, and budget as variables from the current task.
- supporting_trajectory_ids must contain 3 to 12 representative exact IDs from the input that support the whole strategy.
- Do not invent tools, controls, product facts, or recovery behavior absent from the corpus.

Execution requirements for every card:
- Extract every mandatory constraint from the task before acting: product type, descriptive attributes, exact options or variants, quantity/size, and price ceiling.
- Search with the product type and the most discriminative requested terms; do not overload the query with prose.
- Screen visible result titles and prices, but treat them only as preliminary evidence.
- Open a promising candidate and verify the displayed price again.
- Verify every unresolved constraint using the product title, option list, Description, Features, or Attributes as appropriate. Inspect only views needed for the current goal; do not browse every tab mechanically.
- When entering a Description, Features, Attributes, or Reviews subpage, use the visible `< prev` control to return to the same product page. `Back to Search` rejects the current candidate and returns to searching; it is not the normal return from a detail subpage.
- If a required option is present, click the exact value and confirm its Selected state. If required evidence or an option is absent, reject the candidate and continue with another result or a refined query.
- Click Buy Now only after the conjunction of product identity, price, required attributes, and required options is satisfied.
- Include a termination bias: once every task constraint is positively verified, stop exploring and buy instead of reopening tabs or restarting the search.

Card quality:
- name is a concise label for the full task pattern, not an individual action.
- description gives discriminative applicability conditions so top-1 retrieval can choose one card from a task goal and initial observation.
- procedure is an ordered end-to-end algorithm. It must explicitly reach Buy Now.
- constraints are correctness guards and navigation semantics that prevent premature purchase and unproductive loops.
- Avoid generic tool manuals, transcript summaries, benchmark commentary, and claims of guaranteed success.

Output discipline:
- Call the supplied function exactly once and return no prose outside it.
- Return only the skills array with exactly the required fields.
- Before calling the function, silently verify that every card is standalone, end-to-end, category-independent, and materially distinct from the others.
"""
