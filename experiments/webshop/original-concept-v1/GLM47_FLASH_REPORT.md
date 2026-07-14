# GLM-4.7 Flash Test-A Report

This report covers one real execution per Test-A task (`n=100`), using
`glm-4.7-flash`. The SkillX retry run replaces the eight transport-failed
tasks from the primary run; it is not an additional repeat.

## Aggregate Results

| Method | N | Mean reward | Exact reward=1 | Mean steps | Invalid action rate | Mean input tokens | Mean context chars | Completed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NoSkill | 100 | 0.5536 | 31% | 6.39 | 8.29% | 14,431 | 0 | 85% |
| Manual skill | 100 | 0.7060 | 44% | 4.36 | 2.29% | 9,015 | 2,552 | 97% |
| SkillX P100 | 100 | 0.3750 | 27% | 12.16 | 61.43% | 34,988 | 6,679 | 50% |
| Trace2Tower cap8 | 100 | 0.3587 | 28% | 12.36 | 58.66% | 59,774 | 132,969 | 48% |

All values are computed from the raw `results.jsonl` records. Reward is the
WebShop `primary_score`; invalid action rate is pooled invalid actions divided
by executed steps.

## Paired Differences

The comparison is paired by the same Test-A `sample_id` against NoSkill.

| Method | Mean reward delta | Wins | Ties | Losses |
| --- | ---: | ---: | ---: | ---: |
| Manual skill | +0.1524 | 28 | 65 | 7 |
| SkillX P100 | -0.1786 | 12 | 46 | 42 |
| Trace2Tower cap8 | -0.1949 | 15 | 43 | 42 |

## Interpretation

1. GLM can use a compact, fixed instruction: Manual skill improves over
   NoSkill by `+0.1524` reward and reduces invalid actions. This confirms that
   the model is not simply unable to benefit from guidance.
2. Both automatically retrieved libraries produce negative transfer on this
   model. SkillX and Trace2Tower have nearly the same failure pattern: roughly
   half of episodes reach the step limit and invalid actions exceed 58%.
3. Trace2Tower is substantially more expensive in this run. Its mean input is
   about 4.1x NoSkill and its mean injected context is about 133k characters.
   This is a context-following stress case, not evidence that the graph method
   is intrinsically inferior to SkillX.
4. The defensible paper framing is to use DeepSeek Flash/Pro as the primary
   effectiveness models and report GLM-4.7 Flash as a model-transfer stress
   test. The GLM result establishes a boundary: compact manual guidance works,
   while long automatically retrieved execution context can cause negative
   transfer. It must not be presented as a positive Tower result or hidden.

## Provenance

- NoSkill: `artifacts/runs/webshop-original-concept-v1-test-a-glm47flash-noskill-r1`
- Manual: `artifacts/runs/webshop-original-concept-v1-test-a-glm47flash-manual-r1`
- Tower: `artifacts/runs/webshop-original-concept-v1-test-a-glm47flash-tower-cap8-r1`
- SkillX primary: `artifacts/runs/webshop-original-concept-v1-test-a-glm47flash-skillx-p100-r1`
- SkillX retry: `artifacts/runs/webshop-original-concept-v1-test-a-glm47flash-skillx-p100-r1-retry`

The primary SkillX run had `92` successful transport results and `8` transient
connection failures. The retry completed exactly those eight task IDs with
`0` errors, yielding the final `n=100` SkillX table above.
