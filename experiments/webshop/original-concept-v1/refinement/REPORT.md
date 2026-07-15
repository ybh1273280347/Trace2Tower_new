# WebShop Train-Feedback Refinement

## Feedback execution

The frozen `refinement-train` manifest contains 100 train-only tasks and real
repeat IDs 0, 1, and 2. DeepSeek V4 Flash produced 300/300 NoSkill results and
300/300 Tower v0 results. The Tower run retained 338 recoverable TPM-limit
attempts and has zero unresolved keys.

The refinement input contains 400 original P100 NoSkill trajectories, 300
fresh NoSkill feedback trajectories, and 300 fresh Tower v0 trajectories.
Preprocessing produced 6,485 event segments. Rebuilding the original-concept
EigenTrace graph selected 13 candidate Mid clusters, versus 9 in Tower v0.

## Pareto feedback

| High context bundle | Exposure | Performance | Paired reward gain | Guarded step saving | Mean steps | Pareto front |
|---|---:|---:|---:|---:|---:|---:|
| `high_efbf322a092b` | 282 | 0.7209 | +0.0283 | -0.1112 | 6.30 | 1 |
| `high_f4ff56f0acaa` | 18 | 0.5556 | -0.0778 | -0.2346 | 10.17 | 2 |

Chat tokens are recorded as a secondary diagnostic and are not part of the
primary Pareto rank.

## Interpretable structural decomposition

Raw member overlap makes all 9 old Mid clusters and 13 candidates one connected
component because small cross-cluster leakage edges act as bridges. Such a
9-to-13 jump is not an executable fifth action. The action planner retains an
edge only when it contains at least 5 historical segments and accounts for at
least 10% of both its old cluster and its candidate cluster's historical mass.
All raw edges remain in the audit.

The significant graph covers every old and candidate Mid and has five local
components. A local N-to-M relation is executed as Merge into a deterministic
intermediate node, followed by Split into the candidate Mids.

| Component | Old Mids | Candidate Mids | Atomic execution |
|---|---|---|---|
| 0 | `mid_0000` | `mid_0000` | continuation |
| 1 | `mid_0001, mid_0005, mid_0008` | `mid_0001, mid_0003, mid_0008, mid_0012` | Merge, then Split |
| 2 | `mid_0002` | `mid_0002` | continuation |
| 3 | `mid_0003` | `mid_0004, mid_0005, mid_0006, mid_0010` | Split |
| 4 | `mid_0004, mid_0006, mid_0007` | `mid_0007, mid_0009, mid_0011` | Merge, then Split |

This yields two Merge steps and three Split steps. The dominated V0 High bundle
`high_f4ff56f0acaa` is downweighted in the source lifecycle and disappears when
the High layer is re-mined, so V1 does not carry a dangling status penalty.
Promote is not selected separately in this round: all High paths are re-mined
under the unchanged original support and contrast rules.

## Materialized Tower V1

The final input contains 700 NoSkill and 300 prior-Tower feedback trajectories.
The scalable renderer preserves complete cluster membership and aggregate
statistics while showing the renderer at most eight deterministic examples per
Mid, balanced across successful and unsuccessful trajectories when available.

| Field | Value |
|---|---:|
| Source snapshot | `tower_9094918372ee0a39` |
| Refined snapshot | `tower_2e4d04d23287f600` |
| Training trajectories | 1,000 |
| Mid skills | 13 |
| High skills | 18 |
| Mid coverage | complete |
| High coverage | complete |

The Test-A deployment experiment uses DeepSeek V4 Flash, cap 8, and the V1
snapshot. The abandoned eight-episode downweight-only probe is not part of the
formal result.
