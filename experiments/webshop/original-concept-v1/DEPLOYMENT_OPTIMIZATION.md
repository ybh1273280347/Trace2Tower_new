# Train-Feedback Deployment Refinement

## Active contract

This experiment follows the original recursive refinement protocol. A frozen
Tower v0 is deployed on a fresh train-only feedback set. The resulting
skill-conditioned trajectories are paired with fresh NoSkill trajectories on
the same keys, then merged with the original construction trajectories to
rebuild the graph, Mid clusters, and High paths. Test-A is never an input to
this stage.

A Full episode normally injects one High path, all of that path's child Mid
cards, and the direct Mid retrieval. Reward, steps, and model usage are
therefore observations of a co-injected context bundle, not independent
observations of each card. The old per-card exposure mapping is not used here.

The direct Mid cap is fixed at 8. The deployment unit is the High-context
bundle: the selected High path together with its child Mid cards and direct
Mid retrieval that are actually injected. The Pareto objectives are exactly:

```text
performance_level = mean(Tower reward)
paired_reward_gain = mean(Tower reward - paired NoSkill reward)
guarded_step_saving = mean(guarded relative step saving)
```

The guard sets positive step saving to zero when the Tower reward is below its
paired NoSkill reward; negative saving is retained. Bundles are compared only
on these three primary objectives by deterministic non-dominated sorting. Only
a dominated High bundle can be downweighted; at most one bundle is changed in
the first round. The deepest dominated front is preferred, then lower paired
gain, lower performance, higher exposure, and stable ID. The raw cosine and
cap 8 remain unchanged.

Chat input plus output tokens remain in the raw result and are reported as a
secondary cost diagnostic. They do not participate in the main Pareto rank.

## Scope and limitation

The feedback manifest must be a pre-registered train-only selection disjoint
from P100 construction, validation, Test-A, Test-B, and ablation manifests.
Run fresh Flash NoSkill and Tower v0 episodes on exactly the same
`(sample_id, repeat_id)` keys. The current protocol target is 100 tasks and
three real repeats, or 600 episodes across the paired methods. Mid-card
downweighting remains disabled unless independent variation identifies a
card's marginal effect.

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
feedback and as ordinary trajectory evidence. The pairing does not assign the
whole episode independently to every co-injected Mid card; bundle provenance
is preserved and card-level updates require identifiable evidence.

The recursion is across fresh feedback sets, not repeated fitting on one set:

```text
T_0 -> F_0 -> T_1 -> F_1 -> T_2 -> ...
```

`F_0`, `F_1`, ... are disjoint and are never validation or test data. After a
round is applied, `T_(r+1)` is frozen before the next feedback rollout. The
current experiment runs one round only, so the output is `T_1`; Test-A is run
only after `T_1` is frozen.
