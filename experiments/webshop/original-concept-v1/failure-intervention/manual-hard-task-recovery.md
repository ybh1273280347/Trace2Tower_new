## Constraint-Gated Recovery Policy

Use this policy when a task has several hard constraints or when the first plausible
candidate does not clearly satisfy every requirement.

### 1. Build a hard-constraint checklist

- Before acting, separate the request into product identity or category, required
  properties, exact selectable variants, and maximum price.
- Treat every item as one of `supported`, `contradicted`, or `unknown` for the current
  candidate. An unknown requirement is not satisfied.
- Never relax or silently replace an explicit requirement.

### 2. Search with a non-repeating plan

- Search only when the search tool is available. Return to the search page before
  issuing a new query from any other page.
- Keep a mental ledger of queries already issued and candidates already rejected.
  Never repeat an exact query or reopen a rejected candidate unless new evidence has
  appeared.
- Start with the product identity plus the rarest requested identifiers. If that does
  not surface a plausible candidate, change the retrieval angle instead of making a
  cosmetic rewrite: use identity plus a rare property, then a clear synonym or a
  broader category plus the rare property.
- Use at most three distinct queries. Inspect plausible results before spending the
  remaining budget on more reformulations or pagination.

### 3. Verify candidates by evidence

- Reject a candidate immediately when its category, price, or any explicit property
  contradicts the checklist.
- Use the title, visible description, price, option list, and selected state as
  evidence. Open the most relevant detail tab only when a required property remains
  unknown.
- Do not infer that a related product, attractive bundle, broad category match, or
  marketing phrase satisfies an unshown requirement.
- Inspect at most three distinct candidates. After rejection, return to the results or
  search page and choose a candidate that has not already been tried.

### 4. Gate the purchase

- Do not select `Buy Now` while any hard constraint is unknown or contradicted.
- Select every required visible option exactly, then recheck the checklist against the
  current product page and selected state.
- Buy promptly once every hard constraint is supported and the price is within budget.

### 5. Protect the step budget

- Reserve enough actions to open the final candidate, inspect one needed detail view,
  set its options, and purchase.
- By the midpoint of the episode, stop repeating search patterns and choose between
  the best untried candidate and one genuinely different final query.
- After an invalid action, read the current state and choose a legal action. Never
  repeat the same invalid action.
