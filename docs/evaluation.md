# Evaluation And Paired Bootstrap

## Formal Protocol

`scripts/evaluate_results.py` consumes the fixed episode JSONL protocol directly. Every result is reconstructed as an `EpisodeResult`, so benchmark score semantics, ALFWorld success consistency, token nullability, step counts, and the absence of errors are validated before aggregation.

Each invocation covers one benchmark and split. The selected manifest keys are authoritative, including repeat ID. A method is aggregatable only when it has exactly one official result for every selected key, no unexpected or duplicate keys, no scope mismatch, and `official_result_coverage = 1.0`. `--sample-id` creates an explicit pilot subset; it does not weaken the coverage requirement inside that subset.

The frozen evaluation config uses No-Skill as the baseline, 10,000 paired bootstrap resamples, seed 42, and a 95% confidence interval.

## Aggregate Metrics

- ALFWorld reports official success rate.
- WebShop reports mean continuous reward.
- Invalid action rate is pooled as total invalid actions divided by total steps.
- Steps, completion, latency, skill injection, and context size are summarized per method.
- Input, output, and billable tokens each report an observation coverage rate, observed total, and observed mean. Missing provider billing remains missing rather than becoming zero cost.
- Construction cost is a separate optional contract containing extraction calls, renderer tokens, explicit billable tokens, embedding work, latency, and final skill counts. It is never mixed into episode execution totals.

Pairwise comparison requires identical benchmark, split, sample ID, and repeat ID keys. The reported primary difference is candidate minus No-Skill. Bootstrap resamples the paired differences, not the two methods independently. The report also includes wins/ties/losses, paired step and invalid-action differences, and coverage-aware token differences.

`failures.jsonl` contains only unresolved infrastructure errors. An earlier error is removed from the final failure view once the same complete episode key exists in official results.

Outputs are written atomically as `aggregate.json`, `aggregate.md`, `pairwise.json`, and `failures.jsonl`, with manifest, config, and result-file SHA-256 identities.

## Pilot Verification

Two one-pair training subsets verified plumbing only; their confidence intervals are not statistical evidence.

For `webshop:1000`, No-Skill, Flat Skill Summary, and Static Tower all scored 1.0 with complete coverage. Flat and Static each used 9 steps versus No-Skill's 4, so both paired step differences were +5. Their reward differences and bootstrap intervals were exactly 0. Flat used 24,767 input tokens, Static 24,304, and No-Skill 6,022; billable coverage was zero for all methods.

For `webshop:10009`, SkillX scored 0.3333 versus No-Skill's 1.0. The paired reward difference and single-pair interval were -0.6667. SkillX used 8 steps versus 11 and one fewer invalid action, confirming the earlier observation that a shorter legal path can still violate the decisive task constraint. Two prior dependency errors for the same SkillX episode were correctly excluded because the resumed official result exists.
