# SkillX Baseline

## Upstream Boundary

SkillX is executed from the unmodified checkout at commit `36747f424a17ea041e476adf2ff976a206ec9c30`. The upstream checker rejects a changed commit or diffs in 19 protected algorithm and prompt files and records their committed SHA-256 values.

Trace2Tower owns only three adapters:

- trajectories are converted to SkillX's `task_history` format while preserving the real ALFWorld and WebShop tool names and arguments;
- official prompt messages are passed unchanged through the shared `gpt-5.4` renderer endpoint, with validation delegated to SkillX's own regex parser;
- DBSCAN embedding requests are routed through the shared normalized `qwen3-embedding-8b` endpoint.

Credentials remain in the ignored `.env`. No AgentBench or SkillX image is required.

## Minimal Official-Pipeline Experiment

One fully successful WebShop Flash trajectory was passed through the official hybrid pipeline with shortest-plan extraction, omission-mode atomic extraction, pre-merge filtering, one epoch, and expansion disabled.

| Evidence | Result |
|---|---|
| Source trajectory | `webshop:train:no_skill:webshop:10009:0` |
| Source score / steps | 1.0 / 11 |
| Extracted / filtered / final skills | 5 / 1 / 1 functional |
| Planning skills | 1 |
| Renderer calls | 14 |
| Input / cached / output tokens | 75,059 / 44,800 / 3,908 |
| Runtime | 171.6 seconds |
| Library SHA-256 | `2653c0d6cc7912cb0503a021a836447b42a450e637b8ad1043c5bc2f026b7d8a` |

This proves the pinned official pipeline and adapters execute end to end. Because only one skill survived filtering, upstream correctly skipped DBSCAN and the run used zero embedding tokens. A separate contract test drives the pinned DBSCAN implementation through the Trace2Tower embedding adapter and proves the expected two-cluster result with controlled vectors.

## Quality Assessment

The retained functional skill is relevant to the successful rug task, but its declared precondition says execution begins on the product page while its first instruction calls `click_action('< prev')`. That action belongs to a detail subsection in the source trajectory and may be invalid under the stated precondition. Passing SkillX's official filters therefore does not prove executable skill quality.

The single-source cost and this precondition mismatch did not justify scaling extraction before execution evidence. The baseline is kept for comparison, and the generated library was therefore passed through a retrieval-and-execution gate before any larger build.

## Retrieval And Execution Gate

The execution projection preserves the exact source library SHA-256, pinned upstream commit, each source record SHA-256, stable semantic IDs, card-text hashes, and separate plan and skill indexes. Retrieval reproduces SkillX's inference order with the shared asynchronous embedding runtime:

1. retrieve Top-3 plans for the task at similarity threshold 0.45 and select the first;
2. split that plan with the upstream line rules;
3. retrieve Top-4 skills per plan step at the same threshold;
4. deduplicate by stable skill ID and cap the final list at 10.

The one-plan, one-skill WebShop projection is `skillxlib_8e2ea4328cb70e0f`. Its first build embedded two texts using 1,204 input tokens; the immediate rebuild reused both vectors with zero new calls.

The execution smoke replayed the exact source task with `deepseek-v4-flash`:

| Method trajectory | Score | Steps | Invalid actions | Injected context |
|---|---:|---:|---:|---:|
| Source No-Skill | 1.0 | 11 | 1 | 0 |
| SkillX replay | 0.3333 | 8 | 0 | 5,874 chars |

The injected replay selected `french vanilla sundara` instead of the requested `french cellar`, inspected Features and Attributes, and purchased it. This is one paired task, not a performance estimate or proof that SkillX caused the entire difference. It is sufficient as a cost gate: the extracted guidance was executable and shortened the path, but did not preserve the task's decisive constraint. Full SkillX pilot extraction is therefore not expanded at this stage.

The first two smoke attempts also exposed undeclared WebShop runtime dependencies before any episode execution. `click` and the exact `en_core_web_sm` 3.8.0 model are now locked project dependencies so official reward semantics no longer depend on packages installed in a global Python environment. Re-running the completed smoke skipped the episode with zero additional model calls.
