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
