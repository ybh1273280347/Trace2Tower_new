# Agent rollout model pilot

## Protocol

- Models: `deepseek-v4-flash` and `deepseek-v4-pro`.
- Fixed settings: temperature 0, thinking disabled, identical prompt, tools, manifests, and step limits.
- Paired tasks: 10 ALFWorld training tasks from shards 0 and 1, plus 5 WebShop training tasks from shard 0.
- Full success means `primary_score >= 0.999`. Partial WebShop rewards remain in the mean benchmark score.
- Provider credentials remained in `.env`; pilot trajectories and raw results remain in ignored `artifacts/` paths.
- Early ALFWorld attempts exposed a TextWorld parser concurrency failure. Those attempts were retried from checkpoint after serializing parser access and are not counted as model failures.

## Results

| Metric | deepseek-v4-flash | deepseek-v4-pro |
|---|---:|---:|
| Paired episodes | 15 | 15 |
| Full-success trajectories | 11 | 11 |
| Full-success rate | 73.33% | 73.33% |
| ALFWorld full success | 7 / 10 | 7 / 10 |
| WebShop full success | 4 / 5 | 4 / 5 |
| Total input + output tokens | 409,051 | 562,089 |
| Tokens per full success | 37,186 | 51,099 |
| Mean steps | 9.13 | 10.60 |
| Valid-action rate | 99.27% | 96.86% |
| Mean episode latency | 16.16 s | 21.93 s |

The paired score comparison was also neutral: Pro won one ALFWorld task, Flash won one, and the remaining 13 tasks tied. WebShop scores tied on every task.

Qualitative inspection favored Flash for trajectory mining. Pro produced successful but unnecessarily long shelf scans, repeated object manipulation, and multi-page forward/backward search. Flash also produced exploratory failures, but its successful paths were generally shorter and contained fewer invalid actions. Since Trace2Tower must segment transitions and induce reusable skills, this difference reduces noise rather than merely reducing cost.

## Decision

Use `deepseek-v4-flash` as the sole generator for the shared No-Skill training trajectory pool. Keep `deepseek-v4-pro` available for diagnostics, but do not use it as an automatic fallback or mix its trajectories into the shared pool. This preserves one fixed rollout policy across all downstream methods and retains genuine Flash failures for outcome and contrastive modeling.
