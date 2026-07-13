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
- The paired trajectory pilot selected `deepseek-v4-flash` for the shared No-Skill training pool. `deepseek-v4-pro` remains diagnostic-only and is not mixed into the pool; all skill methods therefore consume trajectories from one fixed generator policy.

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
- ALFWorld serializes TextWorld calls that reach its module-level Tatsu parsers, including game loading and environment steps. Model calls remain concurrent, and the matrix launcher shares one episode semaphore and one API semaphore across all shards.

## Method boundaries

- WebShop event classification follows the supplemental deterministic page-state rules and consecutive equal events are merged.
- ALFWorld change-point segmentation uses within-segment SSE over L2-normalized transition embeddings. Candidate segment costs are precomputed once, and penalty calibration minimizes distance from the target median segment length of 3 with a maximum segment length of 6.
- Transition embeddings are cached by `(embedding model, SHA-256 transition text)` as little-endian float32 vectors. Provider batch size is fixed at 16 after a 64-item full-transition batch repeatedly returned HTTP 500 during the pilot.
- Outcome smoothing uses `(own_score + sum(semantic_similarity * neighbor_score)) / (1 + sum(semantic_similarity))`; the fixed unit prior therefore belongs to the segment's own trajectory outcome.
- Eigengap selection is restricted to 2 through 20 Mid clusters, then clamped by node count and the available non-degenerate eigenvectors. Ties select the smaller K. Semantic-Only consumes the Full build's selected K rather than running eigengap itself.
- High paths use per-trajectory presence support after consecutive Mid duplicate compression. Full success is `primary_score >= 0.999` for contrastive mining only; WebShop's official metric remains continuous reward.
- Mid and High renderer prompts use versioned cache keys, stable tool schemas, static prefixes longer than the automatic-cache threshold, and compact deterministic evidence suffixes. Builder-owned IDs, membership, support, and order are never renderer outputs.
- Preprocessed trajectory records persist their source split and generation method. Tower v0 accepts only explicit `train` and `no_skill` provenance, and its content-addressed snapshot validates complete card coverage, source artifact hashes, structural membership, path support provenance, and card-text-hashed retrieval indexes before execution.
- Skill retrieval runs after environment reset so it consumes the official task goal and actual initial observation. Retrieval embedding input tokens are included in episode input cost, while injected IDs and context size use the existing result fields.
- Flat Skill Summary uses one strict card per fully successful shared training trajectory and Top-3 semantic retrieval. Stable source IDs, prompt and text hashes, library identity, and index membership are builder-owned. Its unchanged pilot rebuild reused every card and vector with zero new calls.
- SkillX runs from pinned unmodified commit `36747f424a17ea041e476adf2ff976a206ec9c30`; a checker protects 19 pipeline, prompt, extraction, filtering, clustering, and inference files. Trace2Tower only adapts trajectories, LLM messages, and embeddings to the shared runtime.
- The one-trajectory SkillX pipeline smoke completed, but the sole retained functional skill contains a likely precondition/action mismatch and cost 75,059 reported input tokens. Its exact-task execution replay shortened the path from 11 to 8 steps while reducing reward from 1.0 to 0.3333, so full SkillX pilot extraction is not expanded. This single replay is a cost gate, not a performance estimate.
- SkillX execution uses a content-addressed projection with exact source-library and source-record hashes, separate plan and skill indexes, Top-3 plan retrieval, and Top-4 per-plan-step skill retrieval at the upstream 0.45 threshold. Unavailable benchmark tools are rejected before execution.
- Deployment refinement will use deterministic Pareto non-dominated sorting over performance level, paired reward gain, guarded step saving, and guarded cost saving. Pareto rank prioritizes already-legal structural proposals; it does not generate Split, Merge, Promote, lineage, or path relationships, and no NSGA-II search is introduced.
- Pareto cost uses only provider-reported `billable_tokens`. The runtime preserves that field when explicitly present, but refinement rejects ranking when it is absent; input plus output tokens are not used as a fallback. The first round may downweight status but never physically deletes a skill.
- Formal refinement evidence binds result hashes to the exact Tower snapshot, run IDs, method, benchmark, and agent model before pairing. Partial reward and step evidence remains visible even when missing cost prevents a four-dimensional rank.
- Final aggregation requires exact manifest coverage within the declared full or pilot subset. ALFWorld reports success rate, WebShop reports mean reward, and invalid action rate is pooled over steps. Token summaries preserve per-field observation coverage.
- Pairwise inference uses candidate-minus-No-Skill differences on identical episode keys with 10,000 paired bootstrap resamples, seed 42, and a 95% percentile interval. Resolved error attempts are excluded from the final failure report.
- Unified matrix execution currently exposes only No-Skill, Flat Skill Summary, SkillX, and Static Tower because these methods have verified providers and complete artifacts. Full, Semantic-Only, and edge ablations are not runnable merely because configuration files exist.
- The two-task WebShop Static matrix smoke produced mean reward 0.5 versus the paired No-Skill 1.0 and one task-limit failure. The pilot Tower is therefore not promoted to a full rollout; it must be rebuilt from a larger Flash-generated training pool before broader evaluation.
- The WebShop Flash shard-0 pool was extended from 5 to 20 complete episodes under the original immutable run contract. The rebuilt `tower_999171fa3b9f880b` is byte-reproducible, but its two-task pool-external gate still lost to No-Skill by 0.3333 mean reward and added 2.5 steps. Every mined High path has only one successful supporting trajectory, so full rollout remains blocked pending a fresh-sample test of confidence-gated High retrieval.
- Test data is frozen: skill construction, parameter selection, and refinement use training data only.
