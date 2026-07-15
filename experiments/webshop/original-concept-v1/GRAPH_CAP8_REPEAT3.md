# Graph Cap8 Repeat3

The Flash Test-A graph-cap8 extension is complete. It combines the existing
real `repeat_id=0` run with the new real repeat IDs 1 and 2. Every condition
covers the same 100 tasks at all three repeats.

| Method | Repeat3 reward | Full success | Steps | Invalid actions | Runtime input tokens |
|---|---:|---:|---:|---:|---:|
| Graph cap3 | **0.71119** | 53.67% | **6.83** | **0.16** | **20,461** |
| Graph cap8 | 0.70464 | 53.00% | 7.58 | 0.21 | 27,844 |
| Legacy cap8 | 0.70958 | **54.67%** | 7.14 | 0.17 | 32,334 |
| P100 SkillX | 0.70627 | 49.33% | 7.04 | 0.31 | 25,009 |
| NoSkill | 0.68492 | 50.33% | 7.89 | 0.38 | 20,610 |

Graph cap8 minus graph cap3 is `-0.00656` reward with task-bootstrap 95%
interval `[-0.03522, +0.01656]`. Graph cap8 minus legacy cap8 is `-0.00494`
with interval `[-0.04033, +0.03017]`. Neither difference excludes zero.

Increasing the graph retrieval budget from three to eight does not improve
reward or full success. It adds about 7,383 runtime input tokens per episode
and 0.75 steps relative to graph cap3. Graph cap3 therefore remains the final
retrieval configuration. Graph cap8 is retained as a budget control showing
that the graph-aware gain does not come from injecting more skills.

Machine-readable analysis:
`experiments/webshop/original-concept-v1/final-flash-graph-cap8-repeat3.json`.
