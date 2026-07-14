# Trace2Tower Static Execution

## Tower v0 Contract

A Tower v0 snapshot is content-addressed and contains the complete executable structure: official Low templates, Mid clusters and cards, High paths and cards, both semantic indexes, frozen method configuration, and training trajectory provenance.

The builder rejects a formal snapshot unless:

- every source record explicitly has `split=train` and `trajectory_method=no_skill`;
- every Mid cluster and High path has a corresponding card;
- Mid membership and High child order match the deterministic structure;
- every member segment and supporting trajectory belongs to declared training provenance;
- Mid and High indexes exactly cover their cards and carry matching card-text hashes;
- all five source artifacts have lowercase SHA-256 identities;
- the snapshot ID matches the canonical snapshot content.

Pilot snapshots:

| Benchmark | Snapshot | Train trajectories | Mid | High |
|---|---|---:|---:|---:|
| ALFWorld | `tower_db9fd4d00d9425c9` | 10 | 6 | 2 |
| WebShop | `tower_e1ef1ba84546d8b6` | 5 | 3 | 2 |

Both snapshots have complete Mid and High coverage.

## Execution Boundary

The public agent resets the benchmark before skill retrieval. The provider therefore receives the official task goal and actual initial observation, embeds both retrieval queries, and returns one `SkillSelection` containing injected IDs, formatted context, and selection-model token usage. The old precomputed `skill_context` entry remains supported.

Retrieval input tokens are added to the episode input total. The final injected IDs and context character count are persisted in the existing result contract.

## End-to-End Smoke

No frozen test task was used. One ALFWorld dev episode and one WebShop train episode were run with `deepseek-v4-flash`, thinking disabled, and the complete pilot snapshots.

| Benchmark | Result | Steps | Invalid | Injected skills | Context chars |
|---|---:|---:|---:|---:|---:|
| ALFWorld dev | 0.0 | 20 | 0 | 4 | 2,949 |
| WebShop train | 1.0 | 9 | 1 | 3 | 3,192 |

Both runs completed the execution pipeline without infrastructure errors and wrote official results plus full trajectories. The ALFWorld episode reached the step limit, so this smoke proves plumbing and persistence, not skill quality or method performance.

Rebuilding each snapshot from identical inputs produced the same snapshot ID and byte-identical file hash. Re-running both smoke commands skipped the completed episode from checkpoint and made no additional model calls.
