# Curated Experiment Configurations

This directory contains the active configurations used by the final ALFWorld
and WebShop experiment chains.  Files retain their source names so they can be
used directly with the current experiment CLI.

- `shared/` contains common runtime and evaluation settings.
- `alfworld/` contains the final Full build/runtime configuration, the G1-G4
  build configurations, the D1-D3 runtime configurations, and SkillX.
- `webshop/` contains the constraint-branch build/runtime configuration and
  the No-Skill, SkillX, and Expert-Crafted Skills baseline configurations.

The old WebShop isomorphic configuration and all deprecated configurations are
deliberately excluded.  `alfworld_ablation_semantic_only.yaml` is retained as
the corrected G1 rerun configuration, but its historical output is not a
reusable artifact because it ignored duplicate-embedding collapse.
