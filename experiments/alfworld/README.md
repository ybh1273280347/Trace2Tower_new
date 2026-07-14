# ALFWorld Experiment Registry

This directory is the authoritative entry point for the ALFWorld paper experiment.

- `official/` lists the only training, validation, and test inputs and runs allowed in the final report.
- `diagnostic/` lists probes and debugging artifacts that may explain implementation decisions but are not evidence for method performance.
- `deprecated/` lists superseded or contaminated artifacts that must never enter analysis.

Large generated files stay under `artifacts/`. The manifests here bind their paths and roles without duplicating them.
