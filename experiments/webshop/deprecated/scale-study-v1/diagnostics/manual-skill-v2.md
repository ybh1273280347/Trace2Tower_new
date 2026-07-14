## Fast Exact-Match WebShop Policy

Goal: finish the purchase in the shortest legal path while preserving every explicitly requested constraint.

### 1. Search once with high-signal terms

- Extract the product category and the 2-4 rarest requested identifiers: exact brand/model, material or finish, color, size, flavor, pack count, or special feature.
- Call `search_action` only on the search page, using a compact phrase such as `loveseat flat packed wood finish`. Do not include the price ceiling in the query; check displayed prices instead.
- Do not search again merely to improve wording. Reformulate at most once, only if the first result page has no plausible candidate. From a results or product page, click `Back to Search` before reformulating.

### 2. Choose a candidate from results

- Read titles and prices. Ignore any item that is the wrong product category or already exceeds the budget.
- Prefer the result whose title contains the most requested identifiers. Click its exact visible product ID with `click_action`.
- Inspect at most two candidates. Do not cycle through many near-duplicates.

### 3. Verify without redundant exploration

- Treat the product title, visible description, displayed price, selected options, and option list as valid evidence. Do not reopen information that is already explicit.
- If a required color, size, finish, flavor, or pack option is listed, click the exact requested option before buying.
- Open at most one of Description, Features, Attributes, or Reviews, and only when a required non-variant property is not visible elsewhere. After reading it, use `< prev` to return.
- A generic marketing synonym may support a feature only when it clearly means the requested property; never invent an absent brand, model, size, color, or pack count.

### 4. Purchase decision

- Buy immediately when the product category, price, all exact variants, and all explicitly requested hard properties are supported by the current page evidence.
- If the candidate contradicts a hard constraint, return to search and inspect the next best candidate. Do not keep investigating a known mismatch.
- Use `click_action(value='Buy Now')` as soon as the match is established. Do not perform extra searches or tab visits after that point.

### 5. Tool-state discipline

- `search_action` is legal only when `Search available: yes`.
- Every `click_action` value must exactly match a currently available value.
- After an invalid action, do not repeat it. Read the current observation and choose a legal available action.
