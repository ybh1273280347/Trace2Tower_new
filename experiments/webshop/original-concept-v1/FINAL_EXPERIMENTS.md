#  Final Algorithm and Experiment Matrix

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
| 3 | T1 final graph-cap3, Flash Test-A repeat1/2 | Complete real repeat3 stability using reusable baseline repeats | Complete |
| 4 | T1 final graph-cap3, Pro Test-A repeat0 | Test whether the final Tower helps a stronger model | Complete |
| 5 | T1 final graph-cap3, Pro Test-A repeat1/2 | Complete real Pro repeat3 after a positive repeat0 result | Running |
| 6 | Semantic-only with state-aware Mid retrieval and total cap3 | Remove graph induction and High paths under the final state/budget contract | Pending implementation |
| 7 | P100 No-Mixed vs V0 Mixed with graph retrieval cap3 | Test failure evidence within the same P100 rollout pool | Running |
| 8 | Test-B NoSkill repeat1 | Diagnose whether the high repeat0 baseline is stable | Complete |

## Completed evidence retained

- T1 graph cap3 versus graph cap8 on Test-A: reward-equivalent; cap3 uses fewer
  steps, invalid actions, and input tokens.
- T1 graph cap3 versus legacy T1: graph retrieval recovers reward and removes
  stage-inappropriate retrieval.
- High max4 versus max6: structurally identical, so no rollout is reported.
- Existing NoSkill, Manual, native P100 SkillX, legacy Tower, scale, renderer,
  and seen-task runs remain historical or baseline evidence under their exact
  recorded runtime contracts.

No-Mixed is compared with V0 Mixed rather than final T1. Both use the same P100
rollout pool and frozen graph-cap3 retriever; comparing No-Mixed directly with
T1 would also change the refinement feedback pool and Pareto lifecycle.

Test-B NoSkill repeat1 is a variance diagnostic triggered by the unusually high
repeat0 point estimate. It cannot replace repeat0. The report must show both
repeats and their two-repeat task mean regardless of direction; selecting only
the lower repeat would invalidate the robustness comparison.

The rerun confirms that Test-B NoSkill is genuinely high rather than a repeat0
outlier. Repeat0 scores `0.73323`; repeat1 scores `0.72798`; repeat1 minus
repeat0 is `-0.00525`, paired interval `[-0.03234, +0.01825]`. Their two-repeat
task mean is `0.73061`. Final Tower remains higher at `0.75123`, a `+0.02062`
delta from the NoSkill two-repeat task mean, interval
`[-0.01829, +0.05992]`. Test-B should therefore be described as an easier
split, not as evidence that skill injection is generally ineffective.

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

Pro repeat0 is also positive: final T1 scores `0.69392/51%`, compared with
NoSkill `0.62000/45%`, native P100 SkillX `0.67458/48%`, and legacy P100 Tower
`0.65733/50%`. Final T1 uses 25,305 input tokens on average, below NoSkill's
25,818 and far below legacy Tower's 41,187. This positive direction authorizes
the real Pro repeat1/2 extension in the execution matrix. Final T1 exceeds
NoSkill by `+0.07392`, paired 95% interval `[+0.02108, +0.12883]`, which is
significant. Its deltas over P100 SkillX and legacy Tower are respectively
`+0.01933`, interval `[-0.03717, +0.07517]`, and `+0.03658`, interval
`[-0.02217, +0.09658]`.

## Flash repeat3 result

| Method | Repeat3 reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68492 | 50.33% | 7.89 | 0.38 | 20,610 |
| Manual | 0.69158 | 50.67% | **6.53** | **0.06** | **19,259** |
| P100 SkillX | 0.70627 | 49.33% | 7.04 | 0.31 | 25,009 |
| **Final T1 graph cap3** | **0.71119** | **53.67%** | 6.83 | 0.16 | 20,461 |

Final T1 minus NoSkill is `+0.02628`, paired task-bootstrap interval
`[-0.02936, +0.08356]`. Final T1 minus SkillX is `+0.00493`, interval
`[-0.03094, +0.04072]`, with a `+4.33` point full-success difference, interval
`[-0.67, +9.67]` points. Reward superiority is directional rather than
significant, but final T1 retains the highest repeat3 reward and full-success
rate while using 18.2% fewer input tokens than SkillX. Complete results are in
`final-flash-repeat3.json`.
