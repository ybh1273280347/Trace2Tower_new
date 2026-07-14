# Deprecated ALFWorld Artifacts

The following outputs are superseded or structurally invalid and must not enter analysis:

- `artifacts/trace2tower/towers/alfworld-success-only-v1.json`
- `artifacts/trace2tower/towers/alfworld-mixed-v1.json`
- `artifacts/trace2tower/towers/alfworld-success-only-family-v1.json`
- `artifacts/trace2tower/towers/alfworld-mixed-family-v1.json`
- `artifacts/trace2tower/towers/alfworld-success-only-family-cap8-v1.json`
- `artifacts/trace2tower/towers/alfworld-mixed-family-cap8-v1.json`
- all family-unaware ALFWorld High cards and snapshots
- `artifacts/trace2tower/alfworld/*/skills-final2`, which used the overly strict `0.075` High support threshold and usually retained only one High per task family

The official Tower snapshots are the unpruned `family-v2` cap3 files listed in `../official/train/manifest.json`.
