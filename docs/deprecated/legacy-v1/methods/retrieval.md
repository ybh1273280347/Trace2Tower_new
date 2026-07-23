# Trace2Tower Retrieval

Trace2Tower builds separate semantic indexes over rendered Mid and High cards using the shared `qwen3-embedding-8b` space. Card IDs, card-text SHA-256 values, and embedding vectors are persisted together and validated for uniqueness and a fixed nonzero dimension. A vector is reused only when both its stable ID and text hash match.

For each task:

1. Embed the task goal and retrieve Top-1 High.
2. Expand every child Mid from that High in its fixed order.
3. Embed `task goal + initial observation` and retrieve Top-2 direct Mid.
4. Deduplicate Mid IDs by first occurrence, preserving High children before direct matches.
5. Inject the High card followed by the final Mid cards.

When no High card exists, retrieval falls back to Top-2 direct Mid. Cosine ties are resolved by stable skill ID. `SkillMatch` records both the selected ID and its raw cosine similarity so retrieval choices remain auditable. Existing `skill_ids`, rendered context, and context character count are retained.

## Pilot

The completed pilot tower contains every discovered Mid and High card.

| Benchmark | Index | Cards | Embedding dimension | Retrieved IDs | Context chars |
|---|---|---:|---:|---|---:|
| ALFWorld | Mid / High | 6 / 2 | 4096 | 1 High + 3 unique Mid | 2,949 |
| WebShop | Mid / High | 3 / 2 | 4096 | 1 High + 2 unique Mid | 3,192 |

An immediate repeat index build reused all 13 card vectors and made zero new embedding calls. The end-to-end smoke selected one of two High candidates in each benchmark, expanded its children, and removed duplicate direct Mid matches.
