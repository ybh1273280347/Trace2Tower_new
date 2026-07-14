# Original-Concept Trace2Tower Fast Gate

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
