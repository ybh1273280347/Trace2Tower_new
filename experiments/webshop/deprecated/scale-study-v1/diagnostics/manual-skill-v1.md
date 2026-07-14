## Constraint-Preserving WebShop Purchase

Treat every requested product property as a hard constraint: product type, brand or model, material or feature, color, size or pack count, and maximum price.

1. Form a concise search query from the product type plus the rarest required identifiers. Use `search_action` only when `Search available` is `yes`.
2. On a result page, use `click_action` only with an exact value from the available list. If a new search is needed, click `Back to Search` first; do not call search from a result or product page.
3. Inspect promising product pages and verify every hard constraint. Use Description, Features, Attributes, or Reviews only when the visible product text does not already prove a requirement.
4. Select every required option, such as color or size, before purchase. Do not assume that the title's default variant is the requested variant.
5. Do not buy when any hard constraint is contradicted or still unknown. Return to search and inspect another candidate.
6. If an action is rejected, read the new observation and available values, then choose a legal recovery action instead of repeating the same invalid call.
7. Once all constraints and the price limit are verified, click `Buy Now` without extra exploration.
