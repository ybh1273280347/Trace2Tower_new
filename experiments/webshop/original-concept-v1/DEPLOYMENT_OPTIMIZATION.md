# Test-A Deployment Optimization

## Active contract

This experiment uses the original Trace2Tower deployment idea, but fixes an
identification problem in the old lifecycle implementation. A Test-A Full
episode normally injects one High path, all of that path's child Mid cards,
and the direct Mid retrieval. Reward, steps, and model usage are therefore
observations of a co-injected context bundle, not independent observations of
each card. The old per-card exposure mapping is not used here.

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

The cap-8 bundles are evaluated on the existing 200 real Test-A episodes
(repeat IDs 1 and 2) with `deepseek-v4-flash`, paired to the existing NoSkill
episodes by `(sample_id, repeat_id)`. This is a deployment-policy diagnostic
on Test-A, not a new held-out generalization claim. Mid-card downweighting is
disabled because the current retrieval bundles do not provide enough
single-card variation to identify a Mid card's marginal effect.
