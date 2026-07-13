# Deterministic transition preprocessing

## ALFWorld

ALFWorld actions are parsed from the upstream command templates into the fixed primitive vocabulary. The local environment's `move ... to ...` surface form and upstream `put ... in/on ...` form both map to `PUT`; `use ...` maps to `TOGGLE`. Unknown tools or commands remain `INVALID`, while the exact raw action is retained.

Each transition is embedded from the frozen text signature containing goal, observation before, primitive plus raw action, and observation after. Segmentation minimizes:

```text
sum(normalized_embedding_within_segment_SSE) + penalty * segment_count
```

Dynamic programming uses closed step intervals and a maximum segment length of 6. All candidate segment SSE values are precomputed once before the calibration search. Calibration selects the smallest tested penalty whose resulting global segment-length median is closest to 3. If the input trajectories are too short to reach 3, it returns the closest reachable median rather than failing or looping.

The Flash pilot contained 10 trajectories and 111 transitions. Qwen3-Embedding-8B produced 4096-dimensional vectors. Pilot calibration produced penalty `0.09937398503188888`, 34 segments, median length 3, and the following length distribution:

```text
1: 4, 2: 12, 3: 5, 4: 3, 5: 4, 6: 6
```

This pilot penalty validates the implementation only. The final penalty must be calibrated once from the complete shared training trajectory pool and written to `artifacts/trace2tower/segmentation-calibration.json`.

## WebShop

WebShop classification uses the supplemental state machine and fixed priority:

```text
search -> buy -> back to search -> page-specific navigation
-> detail entry -> product link -> product option -> OTHER_CLICK
```

DOM-derived clickable kinds take precedence. Text and ASIN fallbacks apply only when structural information is absent. The same `< prev` action is `RESULT_NAVIGATION` on a results page and `DETAIL_BACKTRACKING` on an item-detail page. Classification happens per step before consecutive equal events are merged into closed intervals.

The five-trajectory Flash pilot produced 26 transitions and 26 event segments. It covered query formulation/refinement, candidate and option selection, attribute inspection, detail backtracking, and purchase. No action required `OTHER_CLICK`; the lack of multi-step segments reflects this pilot's absence of consecutive equal event types, while the focused option-selection test verifies merging over steps 2-3.

## Persistence

`preprocess_trajectories.py` atomically materializes one JSONL record per source trajectory with typed transitions and segments. Every segment preserves its source transition IDs, raw actions, closed boundaries, first observation, final observation, trajectory score, and 4096-dimensional mean embedding. The pilot audit proved exact transition coverage and serialization round trips for both benchmarks.
