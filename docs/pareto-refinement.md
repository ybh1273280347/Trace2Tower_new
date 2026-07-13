# Pareto-Based Deployment Management

## Evidence Contract

Refinement consumes only training episodes and pairs each skill episode with the No-Skill episode having the same benchmark, sample ID, and repeat ID. Every injected skill receives one exposure from that pair. A formal audit binds the result files to one Tower snapshot, one skill method, the source run IDs, and the same agent model through hashed run metadata and reports.

For each exposed skill, the four maximized objectives are:

```text
performance_level = mean(skill score)
paired_reward_gain = mean(skill score - paired No-Skill score)
guarded_step_saving = mean(guard(no_skill_steps - skill_steps))
guarded_cost_saving = mean(guard(no_skill_chat_tokens - skill_chat_tokens))
```

Each saving is divided by the corresponding No-Skill value, with a denominator floor of one. When the skill score is lower than No-Skill, a positive raw saving is clamped to zero while a negative saving is preserved. Faster or cheaper regressions therefore receive no efficiency reward.

The frozen refinement config permits one round and defines cost as agent-chat prompt plus completion tokens. Retrieval embedding usage is excluded. Both chat token fields must be recorded by the run; missing evidence blocks ranking. The first round permits status downweighting but not physical deletion.

At least 10 paired exposures are required before a skill can be downweighted. Retrieval retains raw cosine diagnostics and applies a fixed `0.01` status band: a downweighted skill must exceed an active alternative by more than `0.01` to rank ahead. The same constants are fixed before success-only or mixed deployment results are inspected.

## Non-Dominated Sorting

Skills are compared only within the same benchmark, skill level, and refinement round. A skill dominates another when it is no worse on all four objectives and strictly better on at least one. Recursive front removal assigns `pareto_front_rank`, while each record also retains `dominated_by`, `dominates`, the full objective vector, exposure count, and exact paired episode keys.

Pareto rank does not create structural proposals. It only prioritizes proposals already proven legal by structural code:

- Split and Downweight prefer the deepest front, then lower paired gain, lower performance, higher exposure, and stable skill ID.
- Merge preserves two mutually non-dominating F1 skills. Other proposals retain overlap, centroid drift, combined exposure, and stable-ID ordering.
- Promote uses the exposure-weighted mean of child Mid objective vectors, keeps only F1 paths, then orders by contrastive path score, positive support, and path ID.
- First-round Downweight changes `active` to `downweighted`; it does not delete a skill.

No normalized objectives, scalar utility, learned weights, crowding distance, genetic operators, or NSGA-II search are used.

## Pilot Audit

The WebShop pilot audit bound `tower_e1ef1ba84546d8b6` to the Flash No-Skill run and Static Tower smoke. Snapshot ID, benchmark, run IDs, and `deepseek-v4-flash` all matched. The exact pair was `webshop:1000`, repeat 0.

| Evidence | No-Skill | Static Tower | Paired value |
|---|---:|---:|---:|
| Official reward | 1.0 | 1.0 | gain 0.0 |
| Steps | 4 | 9 | guarded saving -1.25 |
| Agent chat tokens | unavailable | unavailable | unavailable |

The Static episode injected one High and two Mid IDs. Because the historical results predate separate agent-chat token fields, the audit records reward and step evidence but emits `ranking_status: unavailable`, no Pareto ranks, and no downweight action. Historical aggregate input tokens are not reused because they may include retrieval embeddings.

New execution records preserve `chat_input_tokens` and `chat_output_tokens` separately from aggregate model usage. A Tower v1 refinement artifact is produced only from runs with complete chat-token evidence.
