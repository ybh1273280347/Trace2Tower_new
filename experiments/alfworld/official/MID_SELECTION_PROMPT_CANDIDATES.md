# Mid Selection Prompt Candidates

## V3 Supporting Coverage

Historical text replaced by V4 on 2026-07-16 14:22 BJT. Cache key:
`trace2tower:mid-self-filter:{domain}:v3`.

```text
Select the smallest useful set of supporting skills for the current task-specific plan. A selected skill must materially help execute at least one plan step. Reject skills tied to a wrong object, operation, destination, or state change. Prefer complementary coverage over duplicate search or transport variants. Return at most the requested number of IDs with the required tool. Return an empty list when every candidate would add misleading object, operation, or location details.
```

Diagnostic: `alfworld-diagnostic-r0-pre-strict-mid-filter-r01`.

## V4 Strict Incremental

Historical main-experiment text. Cache key:
`trace2tower:mid-self-filter:{domain}:v4`.

```text
The task-specific High plan is already complete and has priority. Select the smallest set of Mid skills that adds operational information absent from that plan. A selected Mid must contribute at least one new concrete action pattern, observable precondition, recovery rule, or completion check. Repeating the plan's search, verification, selection, transport, or completion steps at the same level of specificity is not useful. Reject skills tied to a wrong object, operation, destination, state change, or option. Return at most the requested number of IDs with the required tool; an empty list is the normal answer when the High plan is already sufficient.
```

Diagnostics: `alfworld-diagnostic-r0-recovered-mid-filter-r01` and the V4 rerun requested after this snapshot.

Discarded skill-user prompt diagnostic: `alfworld-diagnostic-v4-user-hierarchy-r01`.
The selector remained V4; only a High/Mid relationship prefix was added to the
user context when one or more Mid cards were selected. On the fixed 24-task
diagnostic sample it completed 20/24 tasks, selected 0.38 Mid cards per task,
and selected no Mid on 19 tasks. The prefix was subsequently removed, so this
run is retained only as a discarded diagnostic artifact.

## SkillX-Inspired Relevance

Diagnostic alternative based on SkillX's relevance and redundancy criteria. Cache key remained V4 during the diagnostic.

```text
The task-specific High plan has priority. Review every plan step and select Mid skills whose descriptions best match the current task and support reliable execution of that step. A selected Mid may provide useful search, precondition, action, recovery, or completion guidance even when the High plan mentions the same phase. If multiple candidates overlap, prefer the skill most relevant to the task with the least unnecessary functionality. Reject skills tied to a wrong object, operation, destination, state change, or option. Return an empty list only when no candidate is relevant to the task and plan.
```

Diagnostic: `alfworld-diagnostic-skillx-style-mid-filter-r01`.

## V5 Hierarchical Execution Coverage

Diagnostic candidate derived from the Mid skills selected by r0. Cache key:
`trace2tower:mid-self-filter:{domain}:v5`.

```text
The High plan provides task-level strategy; Mid skills provide reusable execution routines. Review every plan step and select the smallest useful set of Mid skills that makes execution reliable. Select a Mid when it provides concrete support for locating, inspecting or opening, retrieving, required state transformation, transport, recovery, or verification. A Mid can be useful even when the High plan names the same phase, if it supplies an executable procedure or guard at a more concrete level. Prefer complementary coverage over duplicate variants. Reject skills tied to a wrong object, operation, destination, state change, or option. Return an empty list only when no candidate can support execution of any plan step.
```

Diagnostic: `alfworld-diagnostic-v5-hierarchical-mid-filter-r01`.

## V6 High/Mid Relationship Clarification (Not Applied)

Candidate considered after the 2026-07-23 backup. It is not applied to the
selector; the active cache key remains:
`trace2tower:mid-self-filter:{domain}:v4`.

The candidate cache key would have been:
`trace2tower:mid-self-filter:{domain}:v6`.

```text
The High plan is the authoritative task-level strategy: it defines the goal, ordered stages, and task-specific object or destination choices. Mid skills are reusable execution routines that augment, never replace, that strategy. Select the smallest complementary set of Mid skills that adds a more concrete action pattern, observable precondition, recovery guard, or completion check for a High-plan stage. Sharing a search, transport, verification, or transformation stage is expected and is not redundant when the Mid supplies a more operational procedure or safeguard. Reject a Mid only when it conflicts with the task-specific object, operation, destination, state change, or option, or adds no detail beyond the same-level High instruction. Prefer complementary coverage across material execution risks. Return an empty list only when every candidate is conflicting or a same-level duplicate. Return at most the requested number of IDs with the required tool.
```

This selector clarification is not being evaluated. The v4 selector and the
original skill-user card-injection format remain frozen.
