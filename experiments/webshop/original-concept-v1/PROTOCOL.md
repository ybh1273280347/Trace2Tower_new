# Trace2Tower WebShop Original-Concept Protocol

## Authority

`Trace2Tower原始资料.md` is the sole algorithm authority. Later implementation notes and the deprecated event-stratified variant do not define Trace2Tower.

## Full mechanism

1. Convert each trajectory into event segments.
2. Encode each WebShop segment as a compact behavioral signature. Product names, goals, prices, and full page text are excluded.
3. Build a sparse graph whose mask is the union of global semantic kNN edges and observed consecutive trajectory transitions. Event labels never restrict graph edges or clusters.
4. Compute `S_uv = max(0, cosine(h_u, h_v))`.
5. Compute `T_uv = #(type(u) -> type(v)) / #(type(u))` on observed directed transitions.
6. Estimate `rho_u = P(success | u)` with cosine-weighted semantic neighbors and compute `O_uv = 1 - |rho_u - rho_v|`.
7. Use equal weights for the otherwise unspecified `alpha`, `beta`, and `gamma`: `B_uv = M_uv * mean(S_uv, T_uv, O_uv)`.
8. Lift the shared graph into success and failure mass: `W+_uv = B_uv * sqrt(rho_u rho_v)` and `W-_uv = B_uv * sqrt((1-rho_u)(1-rho_v))`.
9. Compute `W_CE = W+ - W-`, its signed normalized Laplacian, the EigenTrace representation, and K-means Mid clusters.
10. Map trajectories to Mid sequences and mine contrastive contiguous paths as High skills.
11. At deployment, retrieve from `(goal, current observation)` at every agent step. High Top-1 and its child Mid skills are injected with direct Mid retrieval.

`semantic_clustering` is the graph-structure ablation: it uses the same segments and compact embeddings but skips `S/T/O`, contrastive graph construction, spectral decomposition, and High induction, then directly applies K-means.

## Fast mechanism gate

- Training evidence: P50 mixed pool, 173 trajectories from 50 tasks.
- Evaluation: the frozen 100-task validation manifest.
- Agent: `deepseek-v4-flash`.
- Repeats: one execution per task, represented by `repeat_id=0` only.
- Tower direct Mid cap: 8.
- Comparator: identical-key NoSkill.
- Gate: Full Trace2Tower mean reward must not be below NoSkill.

## Validation execution policy

All subsequent validation conditions use only `deepseek-v4-flash`, the same 100-task manifest, and one execution per task. Reports and tables must label this as `single-repeat` or `1 run/task`. The statistical unit is the task, and paired task bootstrap may quantify variation across tasks. A result must never be copied into synthetic repeat IDs or reported as three independent executions.

Full Trace2Tower was compared at direct Mid caps 3, 5, and 8 with the rebuilt original-concept artifact. Cap 8 has the highest empirical validation reward and is frozen for all Tower runs. Semantic-only also uses cap 8 without a separate sweep so Full, Semantic, and SkillX share the same retrieval budget. NoSkill, Manual, Global E2E, and SkillX have no validation-selected setting.

## Formal test policy

The frozen 100-task test manifest is evaluated once per task. The P50 main table contains NoSkill, Manual, Global E2E, SkillX, Semantic-only, and Full on both Flash and Pro. These runs were started only after Full cap 8 was frozen. Semantic-only is a graph-structure baseline, not a one-variable ablation.

After the P50 test exposed a validation/test gap, two post-hoc diagnostics were registered:

1. A seen-task diagnostic reruns the 50 P50 training task IDs with fresh Flash executions and compares NoSkill, SkillX, and P50 Full. It measures task-specific memorization and is not held-out evidence.
2. A scale diagnostic rebuilds Full from the nested P100 pool, keeps cap 8 and the same test manifest, and runs Flash only. It measures whether broader training coverage improves held-out performance.

The P100 Full scale follow-up is evaluated on both Flash and Pro with the same snapshot, cap 8, 100-task test manifest, and `repeat_id=0`. P200 remains Flash-only. Any SkillX-style renderer experiment must keep graph, clusters, paths, retrieval, and cap fixed and be reported only as a renderer diagnostic; it does not redefine Full Trace2Tower.

The renderer control was run on the P100 structure. The native Trace2Tower renderer outperformed the SkillX-style adapter and is frozen for P200. P200 is a strict superset of P100, uses four Flash collection rollouts per training task, and is evaluated on the same 100-task Flash test with cap 8. Because the renderer was selected after observing this test set, the P200 result is a post-hoc scale diagnostic rather than new confirmatory test evidence. It must not be described as prompt tuning or as an independent held-out confirmation.

Scale is not assumed to be monotonic. P50, P100, and P200 differ in the evidence pool and in graph-induced Mid/High structure; a larger pool can improve behavioral coverage while worsening structural compression. The scale diagnostic therefore reports the realized artifact and execution result rather than treating pool size alone as the causal variable.

Existing results may be reused by selecting their real `repeat_id=0` rows only when manifest, model, method artifact, retrieval behavior, and execution configuration match exactly. The current reuse audit is recorded in `REUSE.md`.
