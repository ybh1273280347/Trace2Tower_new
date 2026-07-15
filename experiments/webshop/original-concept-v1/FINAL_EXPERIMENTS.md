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
| 1 | V0 graph-cap3 vs T1 graph-cap3, Flash Test-A repeat0 | Isolate Pareto refinement from retrieval | Complete |
| 2 | T1 final graph-cap3, Flash Test-B repeat0 | Frozen-split robustness; compare reusable NoSkill and P100 SkillX | Complete |
| 3 | T1 final graph-cap3, Flash Test-A repeat1/2 | Complete real repeat3 stability using reusable baseline repeats | Running |
| 4 | T1 final graph-cap3, Pro Test-A repeat0 | Test whether the final Tower helps a stronger model | Running |
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

## First final-algorithm results

| Split / method | Mean reward | Full success | Mean steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| Test-A V0 graph cap3 | 0.70892 | 52% | 7.17 | 0.13 | 22,067 |
| Test-A final T1 graph cap3 | **0.71925** | **54%** | **6.81** | 0.16 | **20,059** |
| Test-B NoSkill | 0.73323 | 48% | 7.32 | 0.32 | 16,571 |
| Test-B P100 SkillX | 0.69573 | 47% | 7.50 | 0.34 | 27,330 |
| Test-B final T1 graph cap3 | **0.75123** | **53%** | **6.87** | **0.08** | **19,637** |

The Test-A isolation keeps graph retrieval and cap3 fixed, so the `+0.01033`
mean reward difference is attributable to the registered T1 refinement bundle
rather than to the retriever. Its paired 95% interval is
`[-0.00883, +0.03300]`, so the refinement gain is positive but not significant.

Test-B was not used to design graph retrieval. Final T1 exceeds NoSkill by
`+0.01800`, interval `[-0.02175, +0.05825]`, and P100 SkillX by `+0.05550`,
interval `[+0.01500, +0.10300]`. The SkillX comparison is significant under
the registered paired task bootstrap. Final T1 also exceeds legacy V0 by
`+0.03658`, interval `[-0.00300, +0.07825]`. Complete paired statistics are
recorded in `final-algorithm-results.json`.
