# Original-Concept Trace2Tower Evidence Report

## Rebuilt artifact

| Item | Value |
|---|---:|
| Snapshot | `tower_1714fcabca14668e` |
| Training trajectories | 173 |
| Event segments | 947 |
| Graph edges | 9,018 |
| Observed transition edges | 774 |
| Cross-event candidate edges | 808 |
| Mid clusters | 18 |
| High paths | 13 |
| Weighted event purity | 0.8807 |

Event purity is an audit statistic, not a clustering constraint. The graph and K-means receive no event-type partition.

The first attempt using full goal and page text produced a disconnected task/product graph. It was rejected before rendering. The frozen artifact uses 34 unique compact behavioral signatures and a connected signed graph with nonzero low eigenvalues.

## Skill audit

The 18 Mid cards cover query formulation/refinement, candidate opening, option selection, purchase, description/features/attributes inspection, and detail/search backtracking. The 13 High paths contain query-to-candidate, candidate-to-option-to-purchase, option-to-purchase, and attribute-inspection/backtracking routines.

Retrieval uses the current observation on every step. Skill context is transient for that model call and does not accumulate in conversation history.

## Flash validation, repeat 0

Both conditions cover the same 100 task keys with zero errors.

This is a time-constrained single-repeat validation (`1 run/task`), not a repeat-3 estimate. Its interval measures variation across the 100 tasks and does not estimate run-to-run model variance.

| Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.65235 | 47% | 7.69 | 0.30 | 19,572 |
| Trace2Tower Full | **0.72343** | **51%** | **7.20** | **0.24** | 30,661 |

Paired Full minus NoSkill reward is `+0.07108`; the 10,000-sample task bootstrap 95% interval is `[+0.01675, +0.12909]`. There are 13 wins, 82 ties, and 5 losses. Full-success difference is `+4` percentage points with interval `[-1, +10]` percentage points.

## Decision

The fast mechanism gate passes. The original-concept Full Tower is not worse than NoSkill and has a positive paired reward interval on this one-repeat validation gate. Dynamic retrieval increases input-token cost by about 56.7%; later experiments must report this cost, but it does not block the mechanism gate.

Runs:

- `webshop-original-concept-v1-validation-flash-cap8-r1`
- `webshop-original-concept-v1-validation-flash-noskill-r1`

## Full cap selection

All three conditions use the rebuilt P50 Full artifact, Flash, the same 100 validation tasks, and one run per task.

| Direct Mid cap | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.70002 | 50% | 7.54 | 0.25 | 26,741 |
| 5 | 0.68260 | 49% | 7.74 | 0.27 | 30,432 |
| 8 | **0.72343** | **51%** | **7.20** | **0.24** | 30,661 |

Cap 8 is the empirical optimum and is frozen. Semantic-only uses the same cap without a separate selection run.

## P50 formal test

Every cell covers the same 100 test tasks at `repeat_id=0` with zero unresolved errors.

| Method | Flash reward | Flash full | Pro reward | Pro full |
|---|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 0.62000 | 45% |
| Manual | 0.68875 | 50% | **0.69042** | **51%** |
| Global E2E | 0.59475 | 43% | 0.56342 | 47% |
| SkillX | **0.70692** | 46% | 0.68500 | 49% |
| Semantic-only | 0.65025 | 49% | 0.54167 | 36% |
| P50 Full | 0.67625 | **51%** | 0.64083 | 48% |

Relative to NoSkill, Flash Full is `-0.00450` with task-bootstrap 95% interval `[-0.07200, +0.06258]`; Pro Full is `+0.02083`, interval `[-0.03833, +0.08034]`. The positive Flash validation gain therefore does not replicate on the P50 test.

The graph mechanism still improves over pure semantic clustering. Full minus Semantic is `+0.02600` on Flash, interval `[-0.01425, +0.06875]`, and `+0.09917` on Pro, interval `[+0.01383, +0.18633]`. Pro also gains 12 full-success points. Thus the relational graph and High hierarchy are useful relative to pure clustering, but the P50 library is too weak to consistently beat NoSkill.

Manual and SkillX are the only positive Pro baselines whose reward intervals exclude zero: Manual `+0.07042` and SkillX `+0.06500`. Global E2E is harmful on Flash (`-0.08600`, interval `[-0.15833, -0.01533]`).

## Seen-task diagnostic

Fresh Flash executions were run on the 50 P50 task IDs; training rollout rewards were not reused.

| Method | Seen reward | Delta vs seen NoSkill | Full success |
|---|---:|---:|---:|
| NoSkill | 0.65383 | - | 48% |
| SkillX | 0.65750 | +0.00367 | 46% |
| P50 Full | 0.66600 | +0.01217 | 46% |

Neither method shows strong task-specific memorization. This is consistent with the compact Tower signature deliberately excluding goals, products, prices, and page text.

## P100 scale diagnostic

P100 contains 100 tasks and four rollouts per task. The mixed selector retained 351 trajectories: 186 full success, 161 partial reward, and 4 same-task contrasts. Preprocessing produced 1,955 event segments and 47 unique compact behavior signatures, versus 34 at P50. The signed graph selected 9 Mid clusters and induced 5 High paths. Snapshot: `tower_9094918372ee0a39`.

| Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| P50 Full | 0.67625 | 51% | 7.58 | 0.22 | 33,384 |
| P100 Full | **0.72092** | **56%** | **7.17** | **0.19** | 32,360 |

P100 minus NoSkill is `+0.04017`, interval `[-0.01425, +0.09775]`. P100 minus P50 Full is `+0.04467`, interval `[-0.00350, +0.09700]`. The single-repeat intervals include zero, so this is a strong positive scale trend rather than a significance claim. Broader training coverage improves every reported execution metric and is the leading explanation for the P50 validation/test gap.

## P100 on Pro

P100 Full was also run on Pro using the same snapshot, cap 8, test manifest, and real `repeat_id=0`. All conditions below cover the same 100 task keys with zero unresolved errors.

| Model | Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---|---:|---:|---:|---:|---:|
| Flash | NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| Flash | P50 Full | 0.67625 | 51% | 7.58 | 0.22 | 33,384 |
| Flash | P100 Full | **0.72092** | **56%** | **7.17** | **0.19** | 32,360 |
| Pro | NoSkill | 0.62000 | 45% | 9.34 | 0.93 | 25,818 |
| Pro | P50 Full | 0.64083 | 48% | 9.19 | 1.06 | 42,738 |
| Pro | P100 Full | **0.65733** | **50%** | **8.58** | **0.76** | 41,187 |

On Pro, P100 minus NoSkill is `+0.03733`, interval `[-0.02433, +0.09967]`, with 14 wins, 79 ties, and 7 losses. Full success improves by 5 points, average steps fall by `0.76`, and input cost increases by 15,369 tokens per episode. P100 minus P50 Full is `+0.01650`, interval `[-0.03867, +0.07450]`, with 11 wins, 80 ties, and 9 losses.

The strongest mechanism comparison is against Semantic-only: P100 Full gains `+0.11567`, interval `[+0.04217, +0.19117]`, and 14 full-success points, interval `[+5, +23]` points. It also uses 17,407 fewer input tokens and 4.13 fewer steps. This strengthens the conclusion that relational graph structure and High induction help a stronger agent relative to pure clustering.

The evidence does not show that Pro benefits more than Flash: P100 minus NoSkill is `+0.03733` on Pro versus `+0.04017` on Flash, and both intervals include zero. P100 Pro also remains below native Manual (`0.69042`) and SkillX (`0.68500`) in reward point estimate. The defensible conclusion is that Full Trace2Tower can help the stronger model and clearly improves over semantic clustering, not that it dominates strong baselines or provides a larger model-strength interaction.

Run: `webshop-original-concept-v1-test-pro-p100-full-cap8-r1`.

## P100 cross-split robustness

Test-B was frozen with seed `20260720` before any Test-B rollout. It contains 100 tasks sampled from the 700 WebShop indices remaining after excluding validation, Test-A, and ablation. Test-A and Test-B use Flash, `repeat_id=0`, and the same P100 Tower and native P100 SkillX artifacts. P100 SkillX is built from all 186 successful P100 trajectories and contains 51 task plans plus 2 atomic skills.

| Test set | Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---|---:|---:|---:|---:|---:|
| Test-A | NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| Test-A | P100 SkillX | 0.71224 | 49% | **6.92** | 0.31 | 24,141 |
| Test-A | P100 Full | **0.72092** | **56%** | 7.17 | **0.19** | 32,360 |
| Test-B | NoSkill | **0.73323** | 48% | 7.32 | 0.32 | **16,571** |
| Test-B | P100 SkillX | 0.69573 | 47% | 7.50 | 0.34 | 27,330 |
| Test-B | P100 Full | 0.71465 | **49%** | **7.13** | **0.18** | 31,236 |

| Test set | Paired comparison | Reward difference | 95% interval | Full-success difference |
|---|---|---:|---:|---:|
| Test-A | Full minus NoSkill | +0.04017 | [-0.01425, +0.09775] | +5 points |
| Test-B | Full minus NoSkill | -0.01858 | [-0.05850, +0.02117] | +1 point |
| Test-A | SkillX minus NoSkill | +0.03149 | [-0.02318, +0.08924] | -2 points |
| Test-B | SkillX minus NoSkill | -0.03750 | [-0.09000, +0.01100] | -1 point |
| Test-A | Full minus SkillX | +0.00868 | [-0.03731, +0.05360] | +7 points, interval [+1, +14] |
| Test-B | Full minus SkillX | +0.01892 | [-0.03333, +0.07225] | +2 points |

Both skill methods change direction relative to NoSkill across the two test sets. The Test-B-minus-Test-A interaction is `-0.05875` for Full minus NoSkill, interval `[-0.12875, +0.01083]`, and `-0.06899` for SkillX minus NoSkill, interval `[-0.14509, +0.00400]`. These are substantial directional shifts but are not statistically significant at 95%. Full minus SkillX is directionally positive on both splits; its interaction is only `+0.01024`, interval `[-0.05826, +0.08073]`.

Pooling the two independently frozen test sets as 200 tasks, Full minus SkillX is `+0.01380`, interval `[-0.02062, +0.04869]`, with a `+4.5` point full-success difference, interval `[0, +9]`. Full minus NoSkill is `+0.01079`, interval `[-0.02425, +0.04621]`. The defensible conclusion is that benefit over NoSkill is split-sensitive, while Full has a more stable directional advantage over the primary P100 SkillX comparator; neither pooled reward difference excludes zero.

All four newly executed matrices cover 100/100 unique task keys with zero unresolved errors. Test-B SkillX and Full required 12 and 16 recoverable connection-error attempts respectively; checkpoint retries filled every missing key without duplicate results.

## P100 No-Mixed evidence ablation

No-Mixed uses the same P100 rollout pool as Full but keeps only all 186 full-success trajectories, versus Full's 351 selected mixed trajectories. Event extraction, compact signatures, relational graph construction, spectral clustering, High induction, native renderer, legacy retrieval, cap 8, Test-A, Flash, and `repeat_id=0` are fixed. The artifact has 19 Mid and 25 High skills, compared with Full's 9 Mid and 5 High skills; removing partial and failure evidence eliminates all negative adjacency mass and admits many more positive-support High paths.

| Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 7.98 | 0.36 | **21,388** |
| P100 SkillX | 0.71224 | 49% | **6.92** | 0.31 | 24,141 |
| P100 No-Mixed | 0.69192 | 52% | 8.26 | 0.44 | 35,609 |
| P100 Full | **0.72092** | **56%** | 7.17 | **0.19** | 32,360 |

No-Mixed minus Full reward is `-0.02900`, interval `[-0.08334, +0.02217]`, with 6 wins, 84 ties, and 10 losses. Full-success difference is `-4` points, interval `[-10, +2]`. No-Mixed takes `+1.09` steps, interval `[+0.36, +1.84]`, adds 0.25 invalid actions, and adds 3,250 input tokens per episode.

No-Mixed minus NoSkill is only `+0.01117`, interval `[-0.05209, +0.07575]`; No-Mixed minus P100 SkillX is `-0.02032`, interval `[-0.05533, +0.01067]`. The ablation does not support the hypothesis that mixed evidence weakens Tower. Reward trends in the opposite direction, and the step-efficiency degradation from removing mixed evidence excludes zero. The most defensible mechanism interpretation is that partial/failure evidence regularizes the graph and High library, reducing path proliferation and inefficient execution.

Run: `webshop-original-concept-v1-test-flash-p100-no-mixed-cap8-r1`, 100/100 unique keys, zero unresolved errors, 14 resolved TPM retry attempts.

## P100 renderer control

The graph, 9 Mid clusters, 5 High paths, retrieval policy, cap 8, model, and test keys were held fixed. Only the text renderer changed. The SkillX-style adapter used the upstream SkillX plan and functional-skill instructions but returned the existing Trace2Tower Mid/High schemas.

| Renderer | Mean reward | Full success | Steps | Invalid actions | Input tokens | Skill context chars |
|---|---:|---:|---:|---:|---:|---:|
| Native Trace2Tower | **0.72092** | **56%** | **7.17** | **0.19** | **32,360** | **76,989** |
| SkillX-style adapter | 0.68983 | 53% | 7.95 | 0.57 | 52,011 | 165,395 |

SkillX-style minus native reward is `-0.03108`. It more than doubles injected context and increases invalid actions, so the native renderer is frozen for P200. This is a renderer diagnostic, not a redefinition of either Trace2Tower or native SkillX.

## P200 scale diagnostic

P200 is a strict superset of P100. Collection produced 800 trajectories from 200 tasks; the mixed selector retained 710 trajectories: 378 full successes, 324 partial-reward trajectories, and 8 same-task contrasts. Preprocessing produced 3,962 event segments and 51 unique compact signatures. The signed graph contains 48,265 edges, including 3,252 observed transition edges and 3,306 cross-event edges. It selected 19 Mid clusters and induced 7 High paths without fallback. Snapshot: `tower_abe0cf9c83bf1a1d`.

| Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| P50 Full | 0.67625 | 51% | 7.58 | 0.22 | 33,384 |
| P100 Full | **0.72092** | **56%** | 7.17 | **0.19** | 32,360 |
| P200 Full | 0.69325 | 52% | **7.12** | 0.34 | 32,471 |

P200 minus NoSkill is `+0.01250`, interval `[-0.04825, +0.07409]`, with 15 wins, 74 ties, and 11 losses. P200 minus P100 is `-0.02767`, interval `[-0.07383, +0.01683]`. P200 therefore remains above NoSkill in point estimate but does not continue the P100 gain. The evidence supports a material, non-monotonic training-coverage effect; it does not support the stronger claim that reward increases monotonically with pool size.

P200 rendering required 26 construction calls, 2,534,338 input tokens (2,337,664 cached), and 4,762 output tokens. Runtime input remains close to P100 because cap 8 is fixed. Construction cost and deployment-time token cost are reported separately.

The P200 run covers all 100 manifest task keys at real `repeat_id=0`, with zero unresolved errors. Interrupted Windows parent processes caused duplicate attempts and one sample-only resume to land outside its normal shard; duplicate rows were removed by task key while retaining the first completed execution. Coverage and metrics are therefore audited over the global manifest-key set, not inferred from the overwritten shard invocation metadata.
