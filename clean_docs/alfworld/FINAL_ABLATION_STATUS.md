# ALFWorld Final Ablation Status

The final Full reference is the collapsed-embedding P310 snapshot
`tower_d2c2d0090ed9b6b4`, constructed from 1,240 No-Skill trajectories
(310 train tasks, repeats 0-3).  It completed 118 of 134 `valid_unseen`
tasks (88.06%) in the formal evaluation.

Reusable build ablations are G2 No Transition, G3 No Outcome, and G4 No
Contrastive.  Their matched completion counts were 69/134, 76/134, and
81/134, respectively.  They share the same P310 preprocessed input and
enable `collapse_duplicate_embeddings: true`.

G1 Semantic-Only is intentionally not included as a reusable snapshot.  Its
historical builder clustered all 13,724 segment instances directly, despite
the configuration requesting duplicate-embedding collapse.  The corrected
configuration is preserved under `clean_configs/alfworld/` and must be rerun
from the retained P310 preprocessed pool before it can enter the ablation
matrix.

The deployment variants D1-D3 reuse the Full snapshot and only change rewrite
and Mid injection.  Their completion counts were 78/134, 81/134, and 78/134,
respectively.  Their current runtime configurations are retained under
`clean_configs/alfworld/`.
