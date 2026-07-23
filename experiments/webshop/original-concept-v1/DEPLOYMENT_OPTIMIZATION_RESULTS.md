# Test-A Deployment Preflight (Not Refinement Evidence)

Configuration: `deepseek-v4-flash`, WebShop Test-A, direct Mid cap 8, existing
real repeat IDs 1 and 2. The 200 Tower episodes are paired with the 200
NoSkill episodes by `(sample_id, repeat_id)`.

| High bundle | Exposure | Performance | Paired reward gain | Guarded step saving | Mean steps | Mean chat tokens | Pareto front |
|---|---:|---:|---:|---:|---:|---:|---:|
| `high_efbf322a092b` | 180 | 0.7101 | +0.0082 | -0.0816 | 7.28 | 31,609 | 1 |
| `high_f4ff56f0acaa` | 20 | 0.6483 | +0.0950 | +0.0314 | 5.80 | 21,325 | 1 |

The primary Pareto axes are performance level, paired reward gain, and guarded
step saving. Both bundles remain on Front 1, so there is no evidence-based
downweight action. The cap-8 runtime is therefore unchanged. Chat tokens are
reported as a secondary diagnostic only and do not change the front.

This is only a preflight diagnostic of the existing Test-A cap-8 result. It is
not a deployment refinement result: Test-A must not be used to create Tower
v1. The absence of a downweight decision here therefore does not freeze or
validate a lifecycle mutation. The actual refinement report must be built from
the separate train-only feedback manifest described in
`DEPLOYMENT_OPTIMIZATION.md`.

Mid-level per-card pruning remains disabled because the bundles co-inject the
High path, its child Mids, and direct Mids, so their individual marginal
effects are not identified.

Machine-readable evidence: `artifacts/experiments/webshop/original-concept-v1/deployment-pareto.json`.
