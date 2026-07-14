# Validation Result Reuse

Results are reusable only when the validation manifest, task key, repeat ID, Flash model, method artifact, retrieval behavior, and execution configuration match exactly.

| Condition | Reuse status | Evidence |
|---|---|---|
| Original-concept Full, cap8 | Reuse | `webshop-original-concept-v1-validation-flash-cap8-r1`, 100/100 keys, zero errors |
| NoSkill | Reuse | `webshop-original-concept-v1-validation-flash-noskill-r1`, 100/100 keys, zero errors |
| Old event-stratified Tower repeat3 | Reject | Wrong snapshot and reset-time retrieval |
| Old Semantic repeat3 | Reject | Old full-content segment embeddings and reset-time retrieval |
| P50 formal test, six methods x two models | Reuse | All 12 conditions cover 100/100 test keys at real `repeat_id=0`, zero unresolved errors |
| P100 Full Flash test | Reuse | `webshop-original-concept-v1-test-flash-p100-full-cap8-r1`, 100/100 keys, zero unresolved errors |
| P50 seen diagnostic | Reuse only as seen evidence | NoSkill, SkillX, and Full each cover the same 50 train task IDs with fresh executions |
| Partial Semantic cap sweep | Reject and deleted | Semantic cap was fixed to 8 without a separate sweep; incomplete cap3/5/8 runs were removed |

For a reusable repeat-3 run, only the real `repeat_id=0` row may be selected for the single-repeat protocol. Repeat IDs must not be averaged and must not be relabeled.
