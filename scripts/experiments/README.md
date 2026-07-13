# Experiment Scripts

实验脚本按阶段组织。所有命令从仓库根目录运行，推荐模块形式：

```powershell
uv run python -m scripts.experiments.<category>.<module> --help
```

凭据和模型 endpoint 只从被 Git 忽略的 `.env` 读取。`artifacts/` 保存可续跑检查点，不提交到 Git。

## 目录

| 目录 | 职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| `data/` | manifest、固定 selection、轨迹池和 evidence 筛选 | dataset、原始 trajectory | manifest、审计后的 pool |
| `run/` | NoSkill、统一矩阵、模型 pilot、smoke | manifest、method artifact | result shard、trajectory、metadata |
| `build/` | 分段、图、卡片、索引、Tower/Flat/SkillX artifact | train trajectory pool | 冻结 method artifact |
| `analyze/` | 完整性、指标、bootstrap、上游审计 | result shard、metadata | report、audit |
| `deployment/` | Pareto exposure 审计和 lifecycle 配置 | 配对 train results、Tower v0 | refinement report、优化配置 |
| `freeze/` | 已封版证据的哈希与覆盖验证 | 正式 results、reports | freeze manifest |

`run/rollout_no_skill_train.py` 还提供脚本共用的 YAML/JSON 原子写入 helper。其他模块通过完整包路径导入，不依赖当前工作目录注入 `sys.path`。

## ALFWorld 主链

生成固定协议：

```powershell
uv run python -m scripts.experiments.data.prepare_alfworld_protocol
```

统一 rollout：

```powershell
uv run python -m scripts.experiments.run.run_matrix `
  --benchmark alfworld --split train --method no_skill `
  --shard-id all --num-shards 10 --run-id <run-id> `
  --agent-model deepseek-v4-pro
```

池完成后依次执行：

```text
analyze.audit_alfworld_pool
build.calibrate_segmentation
build.preprocess_trajectories
build.build_trace2tower_graph
build.build_trace2tower_skills
build.build_trace2tower_index
build.build_tower_snapshot
```

正式数据边界见 `docs/protocols/alfworld-experiment.md`。

## WebShop 部署优化主链

1. 用 Pro 在相同 50 个 train 任务、相同 repeats 上运行 NoSkill、success-only Tower cap3 和 mixed Tower cap3。
2. 分别审计 success-only 与 mixed：

```powershell
uv run python -m scripts.experiments.deployment.audit_pareto_refinement --help
```

3. 生成绑定 refinement report 的 lifecycle 配置：

```powershell
uv run python -m scripts.experiments.deployment.build_pareto_deployment_config --help
```

4. 用 `run.run_matrix` 在验证集比较 v0/v1；规则冻结后才生成新的 held-out selection。

完整协议见 `docs/protocols/webshop-deployment-optimization.md`。

## 冻结验证

WebShop 已封版主实验：

```powershell
uv run python -m scripts.experiments.freeze.freeze_webshop --verify
```

该命令应验证 29 个条件、19,500 个 episode。部署优化是新增实验，不修改旧 freeze manifest。
