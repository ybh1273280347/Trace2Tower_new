# Failure-Set Skill Intervention Protocol

## Question

The experiment asks whether the common zero-reward Test-A failures are caused partly
by missing decision guidance, even though Trace2Tower retrieves many skills for them.

## Frozen failure set

The sample set was selected after inspecting repeat 0. It is the intersection of
zero-reward tasks for Final T1 graph-cap3 and SkillX:

`webshop:232`, `webshop:492`, `webshop:499`, `webshop:664`, `webshop:951`, and
`webshop:969`.

This is a post-hoc diagnostic set. Results must not be presented as a held-out estimate
of generalization or as a replacement for the main Test-A and Test-B conclusions.

## Intervention

The fixed manual skill is `manual-hard-task-recovery.md`. It contains no sample IDs,
product names, target answers, or candidate identifiers. It adds four kinds of guidance
that are weak or absent in the repeatedly retrieved graph skills:

1. explicit hard-constraint state: supported, contradicted, or unknown;
2. a purchase gate that treats unknown constraints as blocking;
3. query and candidate ledgers that prohibit repeated failed actions;
4. a bounded recovery plan for a 20-step episode.

## Comparators

- Final T1 graph-cap3, repeats 0, 1, and 2;
- SkillX p100, repeats 0, 1, and 2;
- the existing generic manual WebShop skill, repeats 0, 1, and 2;
- the new hard-task recovery skill, repeats 0, 1, and 2.

The statistical unit is the task mean across the three real repeat IDs. With only six
post-hoc tasks, the report uses descriptive paired differences and wins/ties/losses,
not significance claims.

## Interpretation boundary

Improvement would show that these trajectories contain a recoverable guidance gap. It
would motivate learning or retrieving event-derived skills that encode stateful
recovery and constraint gates. It would not justify embedding WebShop product rules in
the general pipeline.

No improvement would leave several explanations open, including search-corpus
coverage, environment observability, model execution limits, or an intervention that
still fails to express the needed control policy.
