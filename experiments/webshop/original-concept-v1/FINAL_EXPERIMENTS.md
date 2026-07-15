# Final Algorithm and Experiment Matrix

## Frozen final algorithm

The final WebShop algorithm is P100 Pareto T1 with the native Trace2Tower
renderer and graph-aware dynamic retrieval. It uses the frozen Test-A-selected
deployment configuration below.

| Component | Frozen value |
|---|---|
| Training evidence | P100 mixed pool plus the registered refinement feedback pool |
| Mid induction | Original-concept signed EigenTrace graph |
| High induction | Contrastive contiguous Mid paths |
| High maximum length | 4; the max6 structural check was identical |
| Refinement | One Pareto Split, one Promote, one High Downweight |
| Renderer | Native Trace2Tower |
| Retrieval | Per-step graph-aware retrieval from goal and current observation |
| High ranking | Goal relevance, active-node state relevance, event compatibility, path quality |
| Path expansion | Active Mid plus at most one directed successor |
| Total Mid budget | 3, including High path children |
| Event compatibility floor | 0.10 |
| Direct-Mid deduplication | cosine 0.92, MMR relevance weight 0.70 |
| Runtime config | `configs/experiments/webshop_trace2tower_final_runtime.yaml` |

This configuration was selected after Test-A diagnosis and is therefore a
post-hoc deployment configuration on Test-A. Test-B is the first frozen split
not used to design this retriever.

## Excluded from the final algorithm

- Legacy cosine-only High/Mid retrieval and full High-child expansion.
- Cap8 as the default deployment budget.
- High max6, because it produced byte-identical High paths under frozen support.
- SkillX-style rendering, which lost to the native renderer.
- Global E2E repeat extensions and extra Manual repeats.
- Token cost as a primary optimization objective; it remains an auxiliary metric.

## Execution matrix

| Priority | Experiment | Purpose | Status |
|---:|---|---|---|
| 1 | V0 graph-cap3 vs T1 graph-cap3, Flash Test-A repeat0 | Isolate Pareto refinement from retrieval | Running |
| 2 | T1 final graph-cap3, Flash Test-B repeat0 | Frozen-split robustness; compare reusable NoSkill and P100 SkillX | Running |
| 3 | T1 final graph-cap3, Flash Test-A repeat1/2 | Complete real repeat3 stability using reusable baseline repeats | Queued |
| 4 | T1 final graph-cap3, Pro Test-A repeat0 | Test whether the final Tower helps a stronger model | Queued |
| 5 | Pro repeat1/2 | Extend only after the repeat0 direction and cost are known | Conditional |
| 6 | Semantic-only with state-aware Mid retrieval and total cap3 | Remove graph induction and High paths under the final state/budget contract | Pending implementation |
| 7 | No-Mixed with graph retrieval cap3 | Test the value of failure evidence under the final runtime | Pending profile build |

## Completed evidence retained

- T1 graph cap3 versus graph cap8 on Test-A: reward-equivalent; cap3 uses fewer
  steps, invalid actions, and input tokens.
- T1 graph cap3 versus legacy T1: graph retrieval recovers reward and removes
  stage-inappropriate retrieval.
- High max4 versus max6: structurally identical, so no rollout is reported.
- Existing NoSkill, Manual, native P100 SkillX, legacy Tower, scale, renderer,
  and seen-task runs remain historical or baseline evidence under their exact
  recorded runtime contracts.
