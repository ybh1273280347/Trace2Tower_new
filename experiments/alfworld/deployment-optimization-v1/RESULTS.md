# ALFWorld Deployment Optimization v1 Results

## Frozen protocol

- Base Tower: `tower_d2c2d0090ed9b6b4`
- Runtime: `plan_rewrite / budgeted_v2 / renderer`
- Feedback: 450 train tasks, repeat 0
- Gate: 120 disjoint train tasks, repeat 0
- Holdout: 120 disjoint train tasks, repeat 0
- Token cost is not an optimization objective.
- Modified candidates must satisfy paired success delta `>= 0` and one-sided 95% lower bound
  `>= -0.03` on gate.

The cleanup-era `high_to_mid` feedback runs are invalid diagnostics and are not included.

## Feedback

| Method | Success | Mean steps |
|---|---:|---:|
| Frozen No-Skill | 58.89% | 14.32 |
| Tower v0 | 70.22% | 12.40 |

Tower v0 has a paired success gain of +11.33 percentage points, with 102 wins, 51 losses,
and 297 ties. The deployment Pareto report proposed downweighting `high_1a65ad05b781`.

Natural graph rebuilding produced 42 candidate Mid clusters from 18,344 segment instances
collapsed into 6,430 embedding nodes. Significant lineage contained 17 continuations, two
splits, one merge, five recompositions, and one new Mid. The no-op structural gate retained
one Split source, `mid_0028`.

## Candidates

- `downweight`: apply a calibrated 0.03 retrieval score penalty to
  `high_1a65ad05b781`. On feedback queries this changes Top-3 exposure from 50 to 26.
- `split`: reversible shadow Split `tower_5b7a48265b8e5cbb`, retaining `mid_0028` for
  existing High-path compatibility and adding two directly retrievable child Mid skills.
- `full`: shadow Split plus the downweight policy.

The shadow Split covers all 308 historical source members. The 38 members outside the two
significant lineage overlaps are deterministically assigned to the nearest child centroid.

## Gate

| Candidate | Success | Delta vs v0 | One-sided 95% lower | Mean steps | Pass |
|---|---:|---:|---:|---:|---:|
| v0 | 80.00% | 0.00pp | n/a | 10.95 | baseline |
| downweight | 80.83% | +0.83pp | -4.17pp | 11.22 | no |
| split | 78.33% | -1.67pp | -8.33pp | 11.41 | no |
| full | 75.00% | -5.00pp | -10.83pp | 11.51 | no |

No modified candidate passes the frozen noninferiority gate. The final selection is therefore
`selection_42750f790fef4155`: `v0_noop`, with no deployment policy. This is not a fallback
chosen after holdout; it is the deterministic result of the predeclared gate.

## Holdout

Because final v1 is identical to v0, one Tower rollout represents both and avoids duplicate
rewrite calls.

| Method | Success | Mean steps |
|---|---:|---:|
| Frozen No-Skill | 59.17% | 14.18 |
| Tower v0 / final v1 | 80.83% | 10.58 |

Final Tower has a paired success gain of +21.67 percentage points, with 32 wins, 6 losses,
and 82 ties. Guarded step saving is +0.1180.

## Artifacts

- Feedback report: `artifacts/trace2tower/alfworld/deployment-optimization-v1/feedback/pareto-report-correct.json`
- Structural report: `artifacts/trace2tower/alfworld/deployment-optimization-v1/proposals/structural-pareto-correct.json`
- Gate report: `artifacts/trace2tower/alfworld/deployment-optimization-v1/gate/report.json`
- Selection: `artifacts/trace2tower/alfworld/deployment-optimization-v1/selected/selection.json`
- Holdout report: `artifacts/trace2tower/alfworld/deployment-optimization-v1/selected/holdout-report.json`
- Correct preprocessed input SHA-256:
  `0555dbd910db687f9bfdc3dc995c6223014a651976bb5c7863fbaf603d3c7d8c`
