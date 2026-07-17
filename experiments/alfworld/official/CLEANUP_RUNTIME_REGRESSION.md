# ALFWorld Cleanup Runtime Regression

## Status

The source cleanup replaced the formal ALFWorld deployment provider instead of only moving it.
Results produced through the replacement provider are diagnostic-only and must not be used in
the paper.

## Formal contract

The original `118/134` Full run used:

- snapshot `tower_d2c2d0090ed9b6b4`;
- `retrieval_strategy: plan_rewrite`;
- `rewrite_contract_version: budgeted_v2`;
- Top-3 High references;
- the structured `submit_task_plan` rewrite tool;
- rewritten-procedure Mid retrieval and `select_supporting_skills` filtering.

Commit `61f9dcf` deleted `plan_rewrite_provider.py` and replaced the configuration with the
SkillX-native `HighToMidSkillProvider`. The replacement takes only the top High as its rewrite
reference and uses different rewrite and Mid-selection prompts. It is not behavior-preserving.

## Invalid runs

The following runs used the replacement `high_to_mid` contract and are retained only for audit:

- `alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r1`
- `alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r2`
- `alfworld-deployment-v1-feedback-pilot-tower-v0-r0`
- `alfworld-deployment-v1-feedback-remaining-tower-v0-r0`

The two invalid test repeats scored `72/134` and `80/134`. Across repeat 0 and these two runs,
all tasks had the same goal, initial observation, and primary High, but no task had an identical
final context bundle or context hash. This is evidence of the provider substitution, not a valid
estimate of the formal method's repeat variance.

## Repair

The original provider contract has been restored under the organized inference and ALFWorld
adapter directories. Future formal analyzers reject runs that do not bind to the original
`plan_rewrite`/`budgeted_v2` contract. Correct repeats use new run IDs and never overwrite or
delete the invalid diagnostic artifacts.

## Correct Repeat Results

The repaired formal contract was rerun on all 134 `valid_unseen` tasks. Repeat 1 scored
`106/134` (79.10%) and repeat 2 scored `108/134` (80.60%), with zero failed episodes. Including
the original repeat 0 (`118/134`, 88.06%), the three-repeat mean is 82.59% with a sample
standard deviation of 4.80 percentage points. The pooled result is `332/402` (82.59%), above
the strongest recorded ablation, D2 (`81/134`, 60.45%), by 22.14 percentage points. The full
per-repeat record is `ALFWORLD_FULL_REPEAT_RESULTS.json`.
