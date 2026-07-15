# Train-Feedback Deployment Refinement

## Active contract

This experiment follows the original recursive refinement protocol. A frozen
Tower v0 is deployed on a fresh train-only feedback set. The resulting
skill-conditioned trajectories are paired with fresh NoSkill trajectories on
the same keys, then merged with the original construction trajectories to
rebuild the graph, Mid clusters, and High paths. Test-A is never an input to
this stage.

A Full episode normally injects one High path, its child Mid cards, and direct
Mid retrieval. Following the original utility update, every actually injected
skill receives observational usage evidence from that episode. This does not
claim a causal marginal effect: when skills always co-occur, their objective
vectors are identical and structural selection must abstain.

The direct Mid cap is fixed at 8. The Pareto objectives are exactly:

```text
performance_level = mean(Tower reward)
paired_reward_gain = mean(Tower reward - paired NoSkill reward)
guarded_step_saving = mean(guarded relative step saving)
```

The guard sets positive step saving to zero when the Tower reward is below its
paired NoSkill reward; negative saving is retained. Skills are compared within
their Tower level by deterministic non-dominated sorting. Pareto rank then
orders already-legal Split, Merge, Promote, and Downweight proposals; it never
creates a structural relation. A structural proposal is ineligible when all
source Mids have the same co-injected exposure set, because no source-specific
utility ordering exists. The raw cosine and cap 8 remain unchanged.

Chat input plus output tokens remain in the raw result and are reported as a
secondary cost diagnostic. They do not participate in the main Pareto rank.

Structural proposals have a separate Pareto scope because cap 8 co-injects all
nine V0 Mids and makes their usage vectors identical. On each local lineage
component, old and candidate partitions are compared over the same shared
historical members using three maximized graph-derived gains:

```text
outcome_consistency_gain
transition_role_coherence_gain
spectral_compactness_gain
```

The no-op vector `(0, 0, 0)` participates in sorting. A structural transaction
is selectable only when it is no worse on all three axes and strictly better
on at least one. This structural Pareto uses the already rebuilt train graph;
it does not run candidate-specific trajectories.

## Scope and limitation

The feedback manifest must be a pre-registered train-only selection disjoint
from P100 construction, validation, Test-A, Test-B, and ablation manifests.
Run fresh Flash NoSkill and Tower v0 episodes on exactly the same
`(sample_id, repeat_id)` keys. The current protocol target is 100 tasks and
three real repeats, or 600 episodes across the paired methods. No additional
rollout is performed for each structural candidate: the frozen V0 feedback is
the deployment evidence used to update utility and select legal actions.

The old Test-A bundle table is retained as a preflight diagnostic only. It is
not refinement evidence and cannot produce Tower v1.

## Recursive update rule

For refinement round `r`, let `T_r` be the frozen Tower and let `F_r` be a
fresh train-only feedback manifest:

```text
B_r       = NoSkill rollout(F_r)
S_r       = Tower(T_r) rollout(F_r)
E_r       = pair(B_r, S_r) by (benchmark, sample_id, repeat_id)
C_r       = construction trajectories used by T_r
P_r       = preprocess(C_r + B_r + S_r)
G_r       = rebuild EigenTrace graph and candidate Mid clusters from P_r
H_r       = mine candidate High paths from G_r
Q_r       = Pareto-rank legal Split / Merge / Promote / Downweight proposals
T_(r+1)   = apply one frozen refinement round to T_r using Q_r
```

`B_r` is used both as the paired baseline for utility and as ordinary
NoSkill evidence in the rebuilt graph. `S_r` is used both as skill-conditioned
feedback and as ordinary trajectory evidence. Injected skills receive usage
credit from the episode, but identical exposure sets are explicitly detected
and cannot drive source-specific structural selection.

The recursion is across fresh feedback sets, not repeated fitting on one set:

```text
T_0 -> F_0 -> T_1 -> F_1 -> T_2 -> ...
```

`F_0`, `F_1`, ... are disjoint and are never validation or test data. After a
round is applied, `T_(r+1)` is frozen before the next feedback rollout. The
current experiment runs one round only, so the output is `T_1`; Test-A is run
only after `T_1` is frozen.
