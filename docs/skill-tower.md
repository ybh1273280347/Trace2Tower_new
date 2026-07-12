# Trace2Tower Skill Tower

## Deterministic Boundary

- Low skills are benchmark-owned primitive action templates. They are not extracted or retrieved.
- A Mid skill is fixed to one `MidCluster`. Code writes its ID, complete member list, support count, evidence, and primitive distribution.
- High paths are mined from ordered Mid IDs after consecutive duplicate compression. Code writes path IDs, Mid order, support values, score, and supporting trajectory IDs.
- GPT-5.4 renders only names, applicability, procedures, constraints, and Mid grounding actions through one required tool call.
- Renderer output is rejected if it adds fields or uses a grounding action absent from both the benchmark Low set and the cluster evidence.

## High Path Definition

For each trajectory, contiguous paths of length 2 through 4 are enumerated after consecutive duplicate Mid IDs are compressed. A path is counted at most once per trajectory and must contain at least two distinct Mid IDs.

The pilot and path miner use the already frozen full-success threshold `primary_score >= 0.999`. Support is the fraction of positive or negative training trajectories containing the path. Paths require positive support of at least `0.02` and are ordered by:

```text
positive_support * log((positive_support + 1e-6) / (negative_support + 1e-6))
```

The threshold is used only to form the positive and negative trajectory sets for contrastive path mining. WebShop's official evaluation metric remains its continuous mean reward.

## Pilot Validation

The selected `deepseek-v4-flash` pilot trajectories were reused without another rollout.

| Benchmark | Trajectories | Mid clusters | High paths | Real cards rendered |
|---|---:|---:|---:|---:|
| ALFWorld | 10 | 6 | 2 | 2 Mid + 1 High |
| WebShop | 5 | 3 | 2 | 2 Mid + 1 High |

The top ALFWorld path has positive support `0.286`, negative support `0`, and score `3.589`. The top WebShop path has positive support `0.5`, negative support `0`, and score `6.561`.

All six GPT-5.4 calls returned valid constrained tool payloads after using `tool_choice="required"`, which is compatible with the configured proxy. The named-function `tool_choice` form was rejected by that proxy on a High render and is not used.

WebShop's top path forms a coherent search, variant verification, and purchase strategy. The small ALFWorld pilot produced a broader top path that combines a heat-and-discard behavior with a subsequent placement behavior. The renderer correctly preserved that structure. High path quality should therefore be judged again after expanding the Flash training trajectory pool; renderer text must not be used to repair structural evidence.

## Prompt Cache

Renderer requests follow the OpenAI prompt-caching guidance: the fixed instructions and stable tool schema precede compact, deterministically serialized variable evidence. Mid requests use a versioned cache key per benchmark, while High requests use one versioned High-renderer key. The Mid tool enum contains the benchmark's complete official Low action set so its schema does not change between clusters; code still rejects any returned action absent from the specific cluster evidence.

The common runtime records `cached_input_tokens` and `cache_write_input_tokens` when the provider returns them. A two-request ALFWorld experiment with different clusters measured:

| Request | Input tokens | Cached input tokens | Latency |
|---|---:|---:|---:|
| First Mid | 7,237 | not reported | 8,502 ms |
| Second Mid | 7,354 | 4,992 | 5,092 ms |

The second request therefore reported a `67.9%` input-cache hit. The configured GPT-5.4 proxy did not report cache-write tokens. The implementation follows the official [Prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).
