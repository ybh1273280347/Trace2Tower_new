# Trace2Tower Public Package

This directory is the public distribution of Trace2Tower. It contains only:

- the Tower graph, spectral, induction, retrieval, and artifact contracts;
- the LLM agent/runtime interfaces;
- sanitized Low/Mid/High skill snapshots.

It intentionally excludes datasets, raw trajectories, experiment runners, benchmark matrices,
provider credentials, and diagnostic outputs.

## CLI

```text
trace2tower build-tower --help
trace2tower extract-skills --help
```

The package layout follows responsibilities rather than experiment history:

```text
trace2tower/
  agent/          LLM transport and tool-using evaluator
  adapters/       domain event and action adapters
  preprocessing/  trajectory-to-transition/segment transforms
  graph/          EigenTrace graph and spectral decomposition
  skills/         Mid/High induction and skill card models
  retrieval/      embedding index and search primitives
  artifacts/      validated Tower snapshots
  inference/      plan rewrite and context formatting
  rendering/      LLM-backed skill rendering contracts
  core/           shared contracts and Tower models
```
