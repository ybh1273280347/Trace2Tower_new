# Official ALFWorld Experiment

The final protocol uses one global trajectory pool and the complete `valid_unseen` test cohort. SkillX and Trace2Tower are both built and retrieved globally; no task-family partition is part of the contract.

- `train/manifest.json`: trajectory pool and skill-construction artifacts.
- `validation/manifest.json`: official validation partition and its declared non-use for cap selection.
- `test/manifest.json`: formal Flash and Pro run matrix.

Only artifacts referenced by these manifests may be included in the final report.
