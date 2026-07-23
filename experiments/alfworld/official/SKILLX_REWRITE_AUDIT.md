# ALFWorld SkillX Rewrite Audit

## Open-source behavior

The frozen source is `third_party/SkillX-native-36747f4` at commit
`36747f424a17ea041e476adf2ff976a206ec9c30`.

In `inference/skill_usage.py`, `SkillUsageService.prepare_prompt()` executes the
following `plan_with_skill` path:

1. retrieve the top three plans and select the first plan;
2. rewrite that plan when `rewrite_plan=true`;
3. retrieve Function/Atomic skills from the rewritten plan;
4. optionally use the LLM selector to limit the skill list;
5. pass the rewritten plan to `format_system_prompt(..., plan=plan)` together
   with the selected skill library.

Therefore, the open-source implementation directly injects the rewritten plan
into the final system prompt. It does not use rewrite only as an internal skill
retrieval query.

## ALFWorld runs

| Contract | Run | Success rate | Status |
|---|---|---:|---|
| Historical no-rewrite, max skills 8 | `alfworld-test-v1-flash-skillx-global-p310-r0` | 81.34% | Retained baseline, explicitly labeled no-rewrite |
| Open-source rewrite path, max skills 8 | `alfworld-test-v1-flash-skillx-native-rewrite-global-p310-r0` | 67.91% | Complete diagnostic: 134 unique results, zero errors |
| Open-source rewrite path, max skills 10 | `alfworld-test-v1-flash-skillx-native-rewrite-max10-global-p310-r0` | - | Aborted by user after 7 partial results; invalid for analysis |

The complete rewrite diagnostic uses the same P310 execution library, test
manifest, Flash agent, Plan Top-3, skills-per-step 4, similarity threshold 0.45,
and ALFWorld action environment as the historical run. Enabling rewrite changed
success by -13.43 percentage points relative to no-rewrite, with task-bootstrap
95% CI `[-20.90, -6.72]` percentage points.

## Reporting rule

The 81.34% result may be reported only as `SkillX no-rewrite`. It must not be
described as the default open-source rewrite contract. The 67.91% run is a
runtime diagnostic showing that the repository's rewrite-and-inject path is
harmful under this ALFWorld adaptation.
