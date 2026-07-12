# Trace2Tower implementation decisions

## Repository boundary

- Top-level source documents, local datasets, credentials, third-party repositories, checkpoints, and experiment artifacts are not tracked.
- Project-owned conclusions under `docs/` are tracked.
- AgentBench, SkillX, Princeton WebShop, ALFWorld, and TextWorld remain external source checkouts; source-lock records their revisions and dirty state.

## Dataset contract

- The local Parquet files and runtime artifacts under `Datasets/` are the authoritative experiment data.
- The local ALFWorld data does not match AgentBench's bundled `train_valid` and `new_std` lists. Only 2714 of 3150 AgentBench training paths are present, while every path referenced by the local Parquet files is present.
- ALFWorld uses local `train` (3553) for training, `valid_seen` (140) for development smoke tests, and `valid_unseen` (134) for final testing.
- WebShop preserves the planned global goal ranges: training `[1000, 12000)` and testing `[0, 200)`. These ranges are available in the local 12087-goal dataset.
- AgentBench's action tools, interaction limits, and official score semantics remain the behavioral reference. Full AgentBench Docker images are not required.
- ALFWorld runs through a pinned Python 3.9 sidecar containing the exact local ALFWorld and TextWorld revisions. The local game data is mounted read-only; no full AgentBench image is built.
- WebShop uses the upstream reward implementation and a resumable SQLite FTS5/BM25 index over the same local `id + contents` search corpus. Its ranking is not bit-identical to Lucene, but the fixed index and adapter are shared by every compared method.

## Model contract

- `qwen3-embedding-8b` provides the shared 4096-dimensional embedding space.
- `deepseek-v4-flash` with thinking disabled executes training and test episodes.
- `gpt-5.4` with `reasoning_effort=none` performs skill extraction, merge, and card rendering.
- Provider credentials live only in the ignored `.env` file.

## Recovery contract

- An episode is complete only after an official result with a non-null primary score is durably appended.
- Provider and network errors are written to a separate error JSONL and remain eligible for rerun.
- Resume uses `(benchmark, split, method, sample_id, repeat_id)` as the completion key.
- A partial final JSONL line left by process interruption is discarded on startup before execution resumes.
- All Trace2Tower-owned model calls share one configured API semaphore. The common runtime disables SDK retries and applies the experiment retry policy itself.
- Episode concurrency and provider concurrency are separate limits. Both default to 10 in the frozen common configuration.
- `billable_tokens` remains null when a provider does not explicitly return a billable-token field; it is not inferred from total tokens.
- No-Skill training rollout writes one atomic episode file before completing its result checkpoint, then deterministically materializes the shard's episode files into the shared `shard-XX.jsonl` pool. This prevents partial lines and duplicate trajectories while preserving independent shard recovery.
- `--max-episodes` limits the already-selected shard and is only a bounded verification control; omitting it executes the full fixed shard.
- ALFWorld serializes only TextWorld game loading because its module-level Tatsu parser is not thread-safe. Active environment steps and model calls retain their configured concurrency.

## Method boundaries

- PUE was accidental source text and is not part of the method or experiments.
- WebShop event classification follows the supplemental deterministic page-state rules and consecutive equal events are merged.
- Test data is frozen: skill construction, parameter selection, and refinement use training data only.
