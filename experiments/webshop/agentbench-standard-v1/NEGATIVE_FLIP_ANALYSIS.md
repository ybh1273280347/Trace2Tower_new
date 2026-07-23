# AgentBench Standard Negative-Flip Analysis

## Diagnostic collection

`manifests/tower_negative_flips.jsonl` freezes the 33 tasks on which the P100 Tower reward is
strictly lower than the paired No-Skill reward on the AgentBench WebShop indices `0..199`.
It is a post-hoc diagnostic collection, not a training, selection, or tuning set.

- Manifest SHA-256: `457d61bf1eaf852e175e596b0b2c997cb348978d5932a9954431b5bcf7dd2047`
- No-Skill run: `webshop-agentbench-std-p100-noskill-flash-r0`
- Tower run: `webshop-agentbench-std-p100-tower-flash-r0`
- Recovery cell: `webshop-agentbench-std-p100-tower-flash-r0-recovery-167`

## Observed failure mode

The 33 negative-flip tasks have mean reward `0.6528` under No-Skill and `0.1611` under Tower.
All 33 No-Skill episodes complete; 20 Tower episodes reach the 20-round task limit. Tower adds
an average of 1.79 searches, 2.36 backtracks, and 1.21 detail inspections, while making 0.61
fewer purchases. Five No-Skill full successes become Tower zero-reward timeouts.

This is not explained by coarse task complexity. The negative-flip tasks average 2.88 explicit
attribute/option constraints, nearly the same as the 20 Tower-improvement tasks (2.80), and are
distributed across all five WebShop categories. The difference is the execution path after a
plausible candidate is reached: Tower continues checking, paging, or reformulating instead of
committing to purchase.

## Runtime boundary

The raw High cards are not directly injected into the agent prompt. The runtime first retrieves
three reference High cards, then `PlanRewriteTrace2TowerProvider._rewrite()` generates a new
`runtime_rewritten_high`. `format_tower_context()` injects that rewritten plan as the initial
`Strategy` section. The source cards therefore influence behavior, but they are not the literal
test-time text.

The WebShop rewrite adapter requires backtracking when a property cannot be verified and purchase
only after the product, price, and required options are verified. The frequently retrieved source
High cards carry the same verification-first policy. Thus the observed over-search is consistent
with both the source strategies and the rewrite contract.

Crucially, the execution loop has no state-policy transition after this initial plan:

1. `select_task()` uses task text to retrieve and rewrite the initial strategy.
2. `select_state()` returns an empty selection.
3. The agent receives evolving page observations, but no state-dependent replacement, commit gate,
   remaining-step policy, or rejection ledger.

The failure is therefore a runtime state-machine gap, expressed through a static rewritten High
plan. It is not justified to attribute it solely to the build-time High renderer. The historical
runs retain trajectories and selected IDs but not the exact renderer-produced plan text, so a
claim about a particular rewrite output would exceed the available evidence.

## Examples

- `webshop:56`: No-Skill searches once, opens one candidate, and buys; Tower repeatedly refines
  birthday-cake queries and reaches the task limit without a purchase.
- `webshop:48`: Tower selects the relevant variant and then repeatedly clicks the same option
  until the task limit.
- `webshop:144`, `webshop:127`, and `webshop:165`: No-Skill reaches a successful candidate;
  Tower continues result paging, backtracking, or re-querying and times out.

## Diagnostic implication

The next diagnostic should isolate the runtime stages on this frozen collection: raw reference
High without runtime rewrite, runtime rewrite without a state transition policy, and a predeclared
state-conditioned commit/recovery controller. These are mechanism diagnostics only; their results
must not replace the 200-task AgentBench-standard comparison.
