# Diagnostic Artifacts

These artifacts are retained for implementation diagnosis only and are excluded from formal metrics:

- `artifacts/trace2tower/alfworld/*/skills-final3`: prompt-v3 High rendering probes.
- `artifacts/trace2tower/alfworld/*-family-v3`: interrupted manual-pruning copies; no official snapshot or run references them.
- historical smoke runs touching the protocol exclusions in `configs/experiments/alfworld_protocol_v1.json`.
- transient failure reports and recovery checkpoints under `artifacts/skillx/alfworld-success-family-v1/*/upstream/recovery`.
- `artifacts/experiments/alfworld/diagnostic/failed-skillx-tool-contract`: pre-episode run directories created before fully qualified upstream ALFWorld tool names were projected to the runtime tool name.

The final report must resolve run IDs through `../official/test/manifest.json`, never by recursively scanning all artifacts.
