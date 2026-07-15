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

## Atomic action plan

The 9 old Mid clusters and 13 candidate Mid clusters form one global complex
repartition. There is no independently auditable one-to-many Split or
many-to-one Merge. Applying the whole repartition would bypass the four atomic
action contract, so it is rejected. Promote is also skipped because it would
depend on the rejected Mid repartition.

| Action | Legal proposals | Applied |
|---|---:|---|
| Split | 0 | none |
| Merge | 0 | none |
| Promote | 0 | none |
| Downweight | 1 | `high_f4ff56f0acaa` |

The runtime keeps the Tower v0 snapshot, cap 8, and raw cosine values. It
subtracts the frozen `0.01` status penalty from the dominated High bundle. The
next experiment evaluates this lifecycle v1 on Test-A with DeepSeek V4 Flash.
