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

All nine Mid skills were injected in all 300 Tower episodes. They therefore
share one exposure set and the same usage vector: performance `0.71094`, paired
gain `+0.02192`, and guarded step saving `-0.11862`. Usage Pareto cannot rank
the Mid structural sources, so it is used for Downweight only. Structural
actions are ranked independently from the rebuilt graph below.

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

The three non-continuation components are compared against a no-op vector on
outcome consistency gain, transition-role coherence gain, and EigenTrace
spectral compactness gain. All quantities use the same shared historical core
members before and after the proposed local transaction.

| Component | Outcome gain | Transition-role gain | Spectral gain | Source coverage | Front |
|---|---:|---:|---:|---:|---:|
| 1 | +0.00668 | +0.00287 | +0.63617 | 78.6% | 2 |
| **3** | **+0.02312** | **+0.00957** | **+0.70835** | **98.2%** | **1** |
| 4 | +0.01592 | 0.00000 | +0.52037 | 86.0% | 2 |
| No-op | 0 | 0 | 0 | - | 3 |

Component 3 is selected: split `mid_0003` into candidate descendants
`mid_0004`, `mid_0005`, `mid_0006`, and `mid_0010`. Weak historical leakage
from other old Mids remains with its original owner; new feedback segments in
the selected descendants join the split children. This produces a complete,
non-overlapping partial update.

## Pareto-refined Tower V1

The final input contains 700 NoSkill and 300 prior-Tower feedback trajectories.
The scalable renderer preserves complete cluster membership and aggregate
statistics while showing the renderer at most eight deterministic examples per
Mid, balanced across successful and unsuccessful trajectories when available.

| Field | Value |
|---|---:|
| Source snapshot | `tower_9094918372ee0a39` |
| Refined snapshot | `tower_bb3ec5295dfe207d` |
| Training trajectories | 1,000 |
| Mid skills | 12 |
| High skills | 6 |
| Mid coverage | complete |
| High coverage | complete |

The round applies one Split, promotes the highest-contrast new partial-Tower
path `high_69655a587d87`, and downweights usage-Pareto Front-2 bundle
`high_f4ff56f0acaa`. No Merge is applied in this round.

## Full-repartition diagnostic

Before structural Pareto was connected, the complete 9-to-13 candidate Tower
was run on the same Test-A tasks. It was not selected by the final optimization
rule and is retained only as evidence that applying every candidate change is
harmful. The abandoned eight-episode downweight-only probe is also excluded.

| Method | Mean reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| Tower V0 | **0.72092** | **56%** | **7.17** | 0.19 | 32,360 |
| Structural V1 | 0.68049 | 48% | 7.79 | **0.15** | 32,752 |

Structural V1 minus NoSkill is `-0.00026`, with paired task-bootstrap 95%
interval `[-0.05851, +0.05833]`. Structural V1 minus Tower V0 is `-0.04043`,
interval `[-0.09267, +0.01167]`. The full structural candidate therefore does
not pass the deployment gate and is not promoted over V0. Its lower invalid
action rate is insufficient to offset the reward and full-success loss.

TPM throttling required a resume. The raw V1 run contains duplicate attempts;
formal analysis freezes the first completed record in each sorted shard file
for every `(sample_id, repeat_id)` and retains all later rows in
`test-a-flash.json`. This rule is independent of outcome. The resulting paired
set contains exactly 100 unique task keys.

This diagnostic did not select the partial T1. The structural Pareto rule and
partial snapshot were frozen from train-only graph evidence before evaluating
the selected T1 on Test-A.

## Pareto T1 Test-A Result

The selected partial T1 completed all 100 Test-A keys with zero duplicate keys.

| Tower | Mean reward | Full success | Mean steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| V0 | **0.72092** | **56%** | **7.17** | 0.19 | 32,360 |
| Pareto T1 | 0.67983 | 52% | 7.66 | 0.25 | 35,273 |

The selected graph Split plus Promote and High Downweight did not improve
deployment reward. The offline structural gains are therefore diagnostic
evidence for proposal quality, not evidence that the action should be deployed
without an outcome gate.

The rendered C3 children are behaviorally distinct rather than simple copies of
the old `mid_0003` card. The old card was a generic detail-tab inspection;
the new cards specialize in Features inspection/backtracking, Attributes
inspection, Description verification, and a bounded generic detail check. The
promoted High `high_69655a587d87` composes Description verification followed by
Features verification.

## High Path Length Authority

The authoritative original material specifies retrieving Top-1 High, but does
not prescribe a maximum High path length of 4. The `max_high_path_length: 4`
value comes from the current project configuration and validation contract, not
from `Trace2Tower原始资料.md`. It must therefore be treated as an experiment
configuration choice, not an algorithmic rule from the original concept.

## Retrieval Diagnosis

The deployed retriever was dynamic in time but not graph-aware. The agent
refreshed the selection at every environment step, while each refresh still
used independent cosine Top-K over High and Mid card text. The learned Mid
transition graph, High path position, and positive-versus-negative path quality
did not participate in deployment ranking.

Per-step replay found three concrete failures:

- High selection used the task goal alone, so the same generic High usually
  remained selected across search, result, product, and detail states.
- `direct_mid_top_k` bounded only direct Mid matches. Expanding all children of
  the selected High raised actual per-step context to 9-12 skills at cap8 and
  5-8 skills at cap3.
- The Mid query concatenated the task goal with the current observation. On a
  product page, six or more near-duplicate search/open cards could outrank the
  detail and option skills needed by the current state.

The complete per-step audit is frozen in `RETRIEVAL_DIAGNOSTIC.md`. A separate
detail-state replay in `RETRIEVAL_DETAIL_STATE_DIAGNOSTIC.md` confirms that the
promoted detail High was never reached by legacy goal-only High retrieval.

| Legacy T1 | Mean reward | Full success | Mean steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| cap8 | 0.67983 | 52% | 7.66 | 0.25 | 35,273 |
| cap3 | 0.69283 | 53% | 7.16 | 0.21 | 30,242 |

Cap3 is cheaper and slightly better in this run, but it does not repair the
retrieval contract. It still expands the entire selected High path outside the
nominal direct-Mid cap.

## Graph-aware Dynamic Retrieval

The replacement retriever preserves `K_t = Retrieve(g_t, o_t, T)` and gives
each signal one responsibility:

- goal-to-High semantic similarity represents task relevance;
- observation-to-Mid similarity represents current-state relevance;
- the event distribution learned from each Mid's training segments filters
  small mixed-event tails that are not executable in the current page state;
- the selected High path supplies the active Mid and one directed successor;
- High contrastive score carries success/failure consistency into ranking;
- `mid_context_budget` limits all injected Mid cards, including path children.

The three-task execution probe verified an actual maximum of one High plus
three Mids per step. On Description and Attributes pages it selects promoted
High `high_69655a587d87`; on search and result pages it selects search paths.
New trajectories persist the exact retrieved skill IDs at every step.

## High max=6 Structural Check

Using the same P100 pool, refined Mid clusters, support ratio 0.10, and Pareto
selection, High mining was repeated with maximum path lengths 4 and 6. Both
runs mined 10 candidates and produced identical final six High paths. The two
`high-paths.json` files have the same SHA-256:
`b5a2eeacf07dd65a37f91cfa69118dd2f8ece382a6020afc6f273e5b65503617`.

The length-four ceiling is therefore not active under the frozen P100 support
contract. A Flash rollout would compare identical skill content and is not an
informative max-length experiment. Producing length-five or length-six paths
would also require changing minimum support, which must be reported as a
separate two-variable experiment rather than attributed to the length ceiling.
