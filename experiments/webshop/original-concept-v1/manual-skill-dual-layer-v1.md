## WebShop Dual-Layer Execution Card

This is one end-to-end policy with compact decision units. Apply the global
invariants throughout the episode; use the decision units only to decide what
evidence is sufficient, when to reject a candidate, and when to stop.

### Global objective and invariants

- Bind every action to the task's requested product identity and hard
  constraints. Preserve category, brand or model, required attributes, exact
  selectable variants, pack count, and maximum price.
- Maintain a candidate ledger: for each opened item, mark every requirement as
  `supported`, `contradicted`, or `unknown`. Unknown is not a match.
- Never transfer evidence from one product to another. A similar title,
  adjacent category, bundle, or marketing synonym does not satisfy a hard
  constraint unless the current item's page shows it.
- Prefer the shortest successful path: search, discriminate candidates, verify
  only unresolved requirements, select exact variants, and buy as soon as the
  purchase gate is satisfied.
- Treat an invalid action as a state change. Read the new observation, update
  the ledger, and choose a currently available action; never repeat the same
  invalid action.

### Decision unit: query and candidate discrimination

- Extract the category and the rarest identifying constraints before the first
  search. Use one compact query containing identity plus 2-4 high-signal
  constraints; do not put the price ceiling in the query.
- From the result list, compare visible titles and prices before opening an
  item. Reject a wrong category, explicit contradiction, or over-budget item
  immediately. Prefer the untried candidate matching the most hard constraints.
- Reformulate only when no plausible candidate is visible. Return to the search
  page first, change the retrieval angle rather than making a cosmetic rewrite,
  and do not repeat an identical query or reopen a rejected item.

### Decision unit: evidence and variant binding

- On a product page, use title, description, displayed price, option list, and
  selected state as evidence for this product only.
- Select every requested color, size, finish, flavor, or pack option exactly.
  After each selection, verify that the selected state still matches the task.
- Open at most one relevant detail view when a required non-variant property is
  still unknown. Do not inspect tabs or reviews merely for reassurance; return
  to the product page after obtaining the missing evidence.

### Decision unit: reject, recover, and stop

- If any hard constraint is contradicted, record the rejection and return to
  results/search to try the best untried candidate. Do not spend steps proving
  that a known mismatch is acceptable.
- If a requirement remains unknown, do not buy. Use one targeted detail check
  or one genuinely different query, then make a candidate decision.
- Buy immediately once category, all hard properties, exact variants, and price
  are supported on the current product page. Do not search, browse tabs, or
  revisit details after the gate is satisfied.
- Reserve enough steps for the final detail check, exact option selection, and
  `Buy Now`; by the midpoint, stop cycling through near-duplicate candidates.

### Tool-state contract

- Use `search_action` only when `Search available: yes`.
- Every `click_action` value must exactly match a currently available value,
  including `Buy Now`, option values, detail tabs, and `< prev`.
- After an invalid action, inspect the current observation before acting again.
