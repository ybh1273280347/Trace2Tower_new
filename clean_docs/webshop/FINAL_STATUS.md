# WebShop Final Experiment Status

The final reusable Trace2Tower artifact is the P200
`constraint-branch-v1` snapshot.  It contains the graph, the compact
two-High/three-Mid snapshot, and the generalized single High used for the
primary deployment result.  The final evaluations use the frozen 100-task
validation and TestA manifests.  The generalized High obtained mean rewards
of 0.70402 on validation and 0.71432 on TestA, compared with No-Skill at
0.65235 and 0.68075.  The compact hierarchy is retained as an ablation
snapshot, with corresponding rewards of 0.68785 and 0.69092.

The SkillX baseline uses the frozen P100 execution library
`skillxlib_5346d9c7cc996337`.  The retained baseline configurations are
No-Skill, SkillX, and Expert-Crafted Skills; `expel` remains an enum-only
placeholder in code and has no experiment artifact.

Earlier WebShop Tower versions, the failed isomorphic deployment, and
deployment-refinement diagnostics are deliberately excluded from this clean
set.
