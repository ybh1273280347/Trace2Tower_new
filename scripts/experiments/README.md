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

3. 将初始 NoSkill 池与审计后的 skill-conditioned 轨迹合并为结构重建输入：

```powershell
uv run python -m scripts.experiments.data.materialize_refinement_pool --help
```

4. 生成绑定 refinement report 的 lifecycle 配置：

```powershell
uv run python -m scripts.experiments.deployment.build_pareto_deployment_config --help
```

完整的 Split / Merge / Promote / Downweight 单轮结构更新使用：

```powershell
uv run python -m scripts.experiments.deployment.build_full_refinement --help
```

该命令输出 lineage、四类 proposal 决策、变化卡片、索引、runtime states 和只读 Tower v1。单独的 lifecycle 配置只用于 Downweight 消融，不代表 Trace2Tower-Full。

5. 用 `run.run_matrix` 比较 v0/v1；规则冻结后才生成新的 held-out selection。

固定的 Pro 验证矩阵可直接续跑：

```powershell
uv run python -m scripts.experiments.deployment.run_webshop_validation
```

该命令串行运行 `webshop:50-149`、3 repeats 的 NoSkill、success-only v0/v1 和 mixed v0/v1。可用 `--condition mixed_v1` 只运行单个条件。

完整协议见 `docs/protocols/webshop-deployment-optimization.md`。

## 冻结验证

WebShop 已封版主实验：

```powershell
uv run python -m scripts.experiments.freeze.freeze_webshop --verify
```

该命令应验证 29 个条件、19,500 个 episode。部署优化是新增实验，不修改旧 freeze manifest。

## Skill 注入诊断

只根据已选 skill ID 展开最终注入文本，不运行 embedding、检索或 rollout：

```powershell
uv run python -m scripts.experiments.analyze.expand_skill_ids `
  --artifact artifacts/trace2tower/towers/<snapshot>.json `
  --skill-id <high-or-mid-id> `
  --skill-id <another-id> `
  --output artifacts/diagnostics/expanded-skill.md
```

脚本自动识别 Tower、Flat 和 SkillX artifact，并只格式化明确传入的 ID。Tower High 卡引用的子 Mid 会作为元数据列出，但只有显式传入对应 Mid ID 才会展开其文本。

当前 Tower 结果中的 `skill_ids` 同时承担检索归因：当 `include_high_child_context: false` 时，它仍包含 High 引用的全部子 Mid，但这些子 Mid 不一定进入实际 context。因此仅凭历史结果的 `skill_ids` 无法严格还原实际注入集合；还需结合 `skill_context_chars` 和运行时检索细节判断。

## 全局归纳 Flat

旧 Flat 是一条成功轨迹生成一张卡。更接近常见 skill 获取方式的全局归纳版本，将完整 success-only 轨迹语料一次提交给 GPT-5.4，输出一个无层级 skill 集合：

```powershell
uv run python -m scripts.experiments.build.build_corpus_flat_skills `
  --benchmark webshop `
  --trajectory-glob artifacts/trajectories/webshop/multirepeat/webshop-flash50-repeat4-pool-v1.jsonl `
  --output-dir artifacts/flat_skill_summary/webshop-p50-global-gpt54-v1
```

构建使用 `.env` 中的 renderer 和 embedding endpoint，并强制 `RENDERER_MODEL=gpt-5.4`。旧 Flat library 不会被覆盖；新 library 仍可由 `run.run_matrix --method flat_skill_summary` 直接执行。

高层端到端版本要求每张 skill 独立覆盖搜索到购买的完整闭环，运行时固定 top-1：

```powershell
uv run python -m scripts.experiments.build.build_corpus_flat_skills `
  --benchmark webshop `
  --trajectory-glob artifacts/trajectories/webshop/multirepeat/webshop-flash50-repeat4-pool-v1.jsonl `
  --induction-mode end_to_end `
  --output-dir artifacts/flat_skill_summary/webshop-p50-e2e-gpt54-v2
```

执行时使用 `configs/experiments/flat_skill_summary_top1.yaml`，其 legacy top-1 检索保证每个 episode 恰好注入一张完整 skill。
