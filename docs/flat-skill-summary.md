# Flat Skill Summary Baseline

## Contract

Flat Skill Summary maps each fully successful shared training trajectory to exactly one card. The builder accepts only `split=train`, `method=no_skill`, and scores at or above the frozen success threshold. Builder-owned stable IDs, source trajectory IDs, benchmark, prompt SHA-256, card text hashes, library ID, and embedding index membership remain outside model control.

The renderer receives the task, the initial observation once, and a compact action/result chain. It emits a strict structured card containing a name, summary, preconditions, ordered procedure, checks, and tools. Card checkpoints are atomic. Re-running with the same prompt reuses completed cards and only embeds cards whose stable ID or text hash changed.

Retrieval embeds `task goal + initial observation`, selects Top-3 by cosine similarity, resolves ties by stable skill ID, and records the selected IDs and context size in the normal episode result.

## Pilot Libraries

| Benchmark | Successful / input | Cards | Library |
|---|---:|---:|---|
| ALFWorld | 7 / 10 | 7 | `flatlib_89e81e46883f44de` |
| WebShop | 4 / 5 | 4 | `flatlib_d4f4c1fcba5270e9` |

Both formal compact libraries use prompt SHA-256 `deb1bead2e7dd29d87249a6591a1ba42a874b7cf040e2aad64f13b0d6c30b7b2`. An immediate rebuild reused all 11 cards and all 11 embeddings with zero model calls.

## Execution Smoke

| Benchmark | Split | Score | Steps | Invalid | Top-3 context chars |
|---|---|---:|---:|---:|---:|
| ALFWorld | dev | 1.0 | 13 | 0 | 2,597 |
| WebShop | train | 1.0 | 9 | 1 | 3,533 |

These two episodes prove retrieval, context injection, execution, and persistence. They are not performance estimates.

## Renderer Cache Experiment

| Prompt/evidence variant | Calls | Reported input | Reported cached input | Reported non-cached input |
|---|---:|---:|---:|---:|
| Original concise | 11 | 26,477 | 0 | 26,477 |
| Extended cache prefix | 11 | 84,643 | 52,608 | 32,035 |
| Compact action/result chain | 11 | 57,142 | 0 | 57,142 |

The extended prompt did trigger cache reporting, but its non-cached input alone exceeded the entire concise run, so it was rejected. The compact chain removes directly observable duplicate observations and remains the formal implementation, but identical routing and prompt hashes produced inconsistent provider token accounting; no billing reduction is claimed from that comparison. The proven cost reduction is deterministic card and embedding reuse, which makes unchanged rebuilds free of new model calls.

## Repeated Execution Gate

The formal compact WebShop library `flatlib_d4f4c1fcba5270e9` was evaluated on four source-external training tasks with repeat IDs `0`, `1`, and `2`. Its four card sources are `1000`, `10009`, `10018`, and `10027`; none overlap evaluation tasks `1012`, `1019`, `1022`, and `1027`.

Flat achieved mean reward `0.5556` versus repeated No-Skill `0.4306`. The task-clustered difference was `+0.125` with interval `[-0.1111, 0.375]`; 12 episode pairs contained 5 wins, 5 ties, and 2 losses. It reduced steps by 0.83 per episode while adding 1,625.5 reported input tokens and 0.25 invalid actions. Billable-token coverage remained zero.

Per-task repeated differences were `+0.2222`, `+0.5`, `0`, and `-0.2222`. Unlike both Tower variants, Flat selected different card combinations across goals. This is directional evidence for task-specific successful-trajectory retrieval, but four independent tasks cannot establish a positive interval. The next gate expands the library from the same 50-trajectory Flash pool and evaluates it on fresh repeated tasks.
