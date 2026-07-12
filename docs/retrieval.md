# Trace2Tower Retrieval

Trace2Tower builds separate semantic indexes over rendered Mid and High cards using the shared `qwen3-embedding-8b` space. Card IDs and embedding vectors are persisted together and validated for uniqueness and a fixed nonzero dimension.

For each task:

1. Embed the task goal and retrieve Top-1 High.
2. Expand every child Mid from that High in its fixed order.
3. Embed `task goal + initial observation` and retrieve Top-2 direct Mid.
4. Deduplicate Mid IDs by first occurrence, preserving High children before direct matches.
5. Inject the High card followed by the final Mid cards.

When no High card exists, retrieval falls back to Top-2 direct Mid. Cosine ties are resolved by stable skill ID. `SkillMatch` records both the selected ID and its raw cosine similarity so retrieval choices remain auditable. Existing `skill_ids`, rendered context, and context character count are retained.

## Pilot

The index builder reused the six already rendered pilot cards and made no agent or environment calls.

| Benchmark | Index | Cards | Embedding dimension | Retrieved IDs | Context chars |
|---|---|---:|---:|---|---:|
| ALFWorld | Mid / High | 2 / 1 | 4096 | 1 High + 2 unique Mid | 1,971 |
| WebShop | Mid / High | 2 / 1 | 4096 | 1 High + 2 unique Mid | 2,780 |

Both queries selected the available High card, expanded both children, and removed duplicate direct Mid matches. This validates the retrieval contract but is not a relevance evaluation because the pilot tower contains only one rendered High candidate per benchmark.
