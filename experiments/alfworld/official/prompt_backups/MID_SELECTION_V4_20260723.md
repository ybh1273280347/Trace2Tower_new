# ALFWorld Mid Selection Prompt Backup

Captured: 2026-07-23 BJT

Source: `src/trace2tower/methods/trace2tower/inference/plan_rewrite.py`

This is the exact active selector prompt before the High/Mid relationship
clarification. It is retained so subsequent diagnostics can reproduce the
strict incremental-selection behavior.

- Cache key: `trace2tower:mid-self-filter:{domain}:v4`
- Temperature: `0.0`
- Max output tokens: `400`
- Tool: `select_supporting_skills`

```text
The task-specific High plan is already complete and has priority. Select the smallest set of Mid skills that adds operational information absent from that plan. A selected Mid must contribute at least one new concrete action pattern, observable precondition, recovery rule, or completion check. Repeating the plan's search, verification, selection, transport, or completion steps at the same level of specificity is not useful. Reject skills tied to a wrong object, operation, destination, state change, or option. Return at most the requested number of IDs with the required tool; an empty list is the normal answer when the High plan is already sufficient.
```

## Skill User Context Before Clarification

Before this change, `format_tower_context()` injected the rendered High card as
`## Strategy: {name}` and each selected Mid card as `## Skill: {name}` with no
instruction describing their relationship. The user-facing context consisted only
of the card descriptions, procedures, and constraints.
