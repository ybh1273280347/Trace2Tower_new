# P100 Skill Construction LLM Usage

This audit compares the LLM chat usage recorded during skill construction.
It excludes trajectory collection, embeddings, runtime evaluation, and
currency cost estimates.

| Method | Source trajectories | Constructed skills | LLM calls | Input tokens | Output tokens | Total tokens | Cached input |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trace2Tower | 351 mixed | 9 Mid + 5 High | **14** | **1,269,049** | **7,147** | **1,276,196** | 129,536 |
| SkillX | 186 full-success | 51 planning + 2 atomic | 429 | 1,686,214 | 197,111 | 1,883,325 | 1,065,592 |

Relative to SkillX, Trace2Tower uses 96.7% fewer LLM calls, 24.7% fewer
input tokens, 96.4% fewer output tokens, and 32.2% fewer total chat tokens.
This is notable because the Trace2Tower artifact is built from more source
trajectories. The strongest supported construction-efficiency statement is
that graph-first aggregation compresses the evidence before generation: one
LLM render per selected Mid or High structure replaces SkillX's per-trajectory
extraction, validation, planning, and merge workflow.

The audit does not establish a billing-cost ratio. SkillX has substantially
more cached input, and after subtracting reported cached tokens Trace2Tower has
1,139,513 uncached input tokens versus SkillX's 620,622. Cached-token prices
are provider-specific, both reports omit embedding usage from this comparison,
and the two methods use different evidence-selection policies. The paper should
report calls and token volumes directly instead of translating them into money.

Trace2Tower's large input volume despite only 14 calls also exposes a concrete
optimization target: Mid rendering repeats large contrastive sibling profiles.
Reducing that prompt duplication could improve construction input usage without
changing the graph or runtime method, but it is not part of the measured result.

Machine-readable audit:
`experiments/webshop/original-concept-v1/construction-cost.json`.
