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

