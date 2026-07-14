# Validation Result Reuse

Results are reusable only when the validation manifest, task key, repeat ID, Flash model, method artifact, retrieval behavior, and execution configuration match exactly.

| Condition | Reuse status | Evidence |
|---|---|---|
| Original-concept Full, cap8 | Reuse | `webshop-original-concept-v1-validation-flash-cap8-r1`, 100/100 keys, zero errors |
| NoSkill | Reuse | `webshop-original-concept-v1-validation-flash-noskill-r1`, 100/100 keys, zero errors |
| Old event-stratified Tower repeat3 | Reject | Wrong snapshot and reset-time retrieval |
| Old Semantic repeat3 | Reject | Old full-content segment embeddings and reset-time retrieval |
| Manual Skill | Missing | No exact-key run on the current manifest |
| Global E2E GPT | Missing | No exact-key run on the current manifest |
| SkillX | Missing | No exact-key run on the current manifest |

For a reusable repeat-3 run, only the real `repeat_id=0` row may be selected for the single-repeat protocol. Repeat IDs must not be averaged and must not be relabeled.

