# Public Skill Snapshots

These files contain reusable Trace2Tower Low/Mid/High skill cards only. They do not contain
raw trajectories, task manifests, training observations, source hashes, or experiment metadata.

The snapshot schema is `trace2tower.skill_snapshot.v1`:

- `low_skills`: primitive action templates;
- `mid_skills`: reusable local procedures and constraints;
- `high_skills`: ordered compositions of Mid skills and their task-level procedures.

Load a snapshot as application data, or generate one from a private Tower with:

```text
trace2tower extract-skills --tower private-tower.json --output skills.json
```
