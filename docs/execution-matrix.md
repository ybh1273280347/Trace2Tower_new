# Unified Execution Matrix

## Executable Methods

`scripts/experiments/run_matrix.py` currently exposes only methods with a verified execution provider and a complete artifact boundary:

- `no_skill`
- `flat_skill_summary`
- `skillx`
- `trace2tower_static`

Semantic-Only, Full, and the three edge ablations retain build configurations but are not execution choices until their complete benchmark artifacts exist. This prevents a config name from being mistaken for a runnable method.

Every skill method requires one explicit `BENCHMARK=PATH` artifact assignment per selected benchmark. The matrix parses the full content-addressed library or snapshot, validates its benchmark and internal identity, and records its SHA-256 in `resolved-config.yaml`. Static retrieval counts must match the embedded Tower configuration.

Evaluation reports WebShop's official continuous mean reward and a derived full-success rate at `primary_score >= 0.999`. Completion rate remains separate: a completed purchase may receive partial reward. Reward and full-success pairwise intervals both cluster repeats by task before bootstrap resampling.

Static High retrieval additionally accepts an execution-time cosine threshold recorded in the resolved method configuration. Threshold `-1.0` preserves the original always-select-Top-1 behavior. Diagnostic retrieval reports retain the Top-1 candidate even when the threshold rejects it, so absence of a High skill is distinguishable from an empty High index.

The optional WebShop event-stratified Tower is executable through the same Static provider because its graph-build choice and complete provenance are embedded in the content-addressed snapshot. It is not the selected default artifact: despite pure stage cards, its four-task calibration failed before fresh validation.

Static execution also records whether High child-card bodies are included in model-visible context. Disabling them never removes proof IDs from episode results. The default is enabled because the compact-context calibration did not establish behavioral equivalence.

All methods use the frozen `deepseek-v4-flash` agent policy unless an explicit diagnostic override is recorded. Credentials are loaded from `.env` only for real execution.

## Execution And Recovery

Results use the common path:

```text
artifacts/runs/{run_id}/{benchmark}/{split}/{method}/shard-XX/
```

Each shard owns official results, error attempts, atomic trajectory files, and run metadata. Benchmarks and shards share the global API semaphore and episode semaphore, while each episode creates an independent environment.

A run ID cannot be reused with a different resolved configuration. Resume skips completed episode keys. Shard metadata records both the latest invocation summary and cumulative official result count, result SHA-256, error-attempt count, and trajectory count, so a successful run remains provable after a skip-only resume.

Dry-run validates manifests, method config, artifact identities, sample selection, and shard assignment without loading credentials, creating checkpoints, repairing partial files, or writing a run directory.

`--repeat-id` explicitly expands selected samples into independent checkpoint keys without changing the on-disk manifest. The override is recorded under `selection.repeat_ids` in the immutable resolved config and in matrix metadata. Omitting it preserves the manifest's existing repeat IDs and legacy resolved-config shape.

## Dry-Run Evidence

The ten-shard training dry-run selected every task exactly once:

- ALFWorld: 3,553 total, distributed as 356 for shards 0-2 and 355 for shards 3-9.
- WebShop: 11,000 total, exactly 1,100 per shard.

Flat and Static artifacts were validated for both benchmarks. The current SkillX execution library is WebShop-only and was validated on its exact source sample. None of the four dry-run IDs created an artifact directory.

## Two-Task Matrix Smoke

One real Static matrix run used `tower_e1ef1ba84546d8b6` on WebShop training samples `webshop:1000` and `webshop:10009`. Both episodes persisted successfully; an immediate rerun skipped both and reported two cumulative official results with the same result hash.

| Method | Mean reward | Mean steps | Invalid rate | Input tokens |
|---|---:|---:|---:|---:|
| No-Skill pair | 1.0 | 7.5 | 0.0667 | 29,489 total |
| Static Tower | 0.5 | 14.5 | 0.1034 | 109,404 total |

The paired reward difference was -0.5 with a two-pair bootstrap interval of [-1.0, 0.0]. Sample `1000` tied at 1.0; sample `10009` reached the 20-step limit with reward 0 while retrieving the same one High and two Mid IDs. This is a pilot quality gate, not a performance estimate. The small Tower is not used for a full rollout; the next Tower evaluation requires a larger Flash-generated training pool and a rebuilt structure.
