# Experiment Scripts

当前入口只服务 `webshop-event-tower-v2`。

## 数据协议

```powershell
uv run python -m scripts.experiments.data.prepare_webshop_event_tower_v2
```

该命令按固定 seed 生成 validation/test manifests 和 `configs/experiments/webshop_event_tower_v2.json`。协议已冻结后不得更改 seed、样本或 repeats。

## 构建边界

阶段 1 训练池审计：

```powershell
uv run python -m scripts.experiments.data.audit_webshop_training_pools
```

阶段 2 artifact 冻结审计：

```powershell
uv run python -m scripts.experiments.build.audit_webshop_stage2_artifacts
```

- Global E2E：`scripts.experiments.build.build_global_e2e_skills`
- SkillX：现有 SkillX 构建链，GPT 并发不超过 4
- Trace2Tower：`preprocess_trajectories` → `build_trace2tower_graph` → `build_trace2tower_skills` → `build_trace2tower_index` → `build_tower_snapshot`

Full 和 No-mixed 的 WebShop Tower 必须通过 event stratification 契约。Semantic Clustering 不构建 signed graph 或 High，只作为 baseline。No-event 保留 signed graph 与 High，仅关闭事件分层。Mid-only 复用 Full snapshot 并显式关闭 High。

Tower 的直接 Mid cap 是运行时参数。正式运行必须传入 `--direct-mid-top-k 3`、`5` 或 `8`；SkillX 的 `max_skills: 8` 是其原生能力，不复用这套 Tower 选择规则。

阶段 3 validation 按机器协议顺序运行全部 12 个条件，并写入可恢复 ledger：

```powershell
uv run python -m scripts.experiments.run.run_webshop_stage3_validation
```

阶段 3 覆盖审计、bootstrap 和 cap 冻结：

```powershell
uv run python -m scripts.experiments.analyze.analyze_webshop_stage3_validation
```

旧 scale-study、Flat、Static Tower 和 cap sweep 命令见 `experiments/webshop/deprecated/scripts-experiments-readme-v1.md`，不得用于 v2。
