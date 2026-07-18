# Trace2Tower

Trace2Tower organizes reusable agent experience into a Tower of Low, Mid, and High skills.
This public snapshot contains the Tower algorithm, the LLM agent/runtime contracts, and
sanitized skill snapshots.

## CLI

Install the package, then inspect the two public entry points:

```text
trace2tower build-tower --help
trace2tower extract-skills --help
```

`build-tower` validates and assembles a Tower from caller-owned preprocessing artifacts.
`extract-skills` exports only Low/Mid/High cards, removing trajectory provenance before the
skills are shared with an application.

## Package Layout

- `src/trace2tower/methods/trace2tower`: graph construction, spectral induction, Tower artifacts,
  rendering contracts, and retrieval;
- `src/trace2tower/components`: the LLM runtime and tool-using agent evaluator;
- `skill_library`: sanitized example skill snapshots.

Datasets, raw trajectories, experiment runners, benchmark matrices, credentials, and diagnostic
outputs are intentionally not part of this public snapshot.
