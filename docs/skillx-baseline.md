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

The single-source cost and this precondition mismatch do not justify scaling SkillX extraction yet. The baseline is kept for comparison, but the next evidence gate is retrieval plus one execution smoke using the generated library. Full pilot extraction remains deferred until that smoke shows the retained skills can be injected without degrading the agent trajectory.
