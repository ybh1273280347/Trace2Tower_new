# ALFWorld 部署反馈结构优化实施文档

状态：实验阶段 1 已启动；manifest、反馈指标、Pareto 和 lineage 核心已开始实现，尚未
接入活动推理策略。

## 1. 目标与边界

本实验验证部署反馈能否在不使用模型微调的前提下，持续改进 Trace2Tower 的外部技能
结构。优化必须同时覆盖两类对象：

1. 使用状态：识别稳定有害或低效的 High context bundle，并可逆 Downweight。
2. 层级结构：根据新反馈轨迹产生 Mid Split、Mid Merge 和 High Promote。

以下约束已经冻结：

- 只在 ALFWorld `train` 任务上生成 feedback、候选和选择决策。
- `valid_seen` 与 `valid_unseen` 不参与反馈、候选排序或超参数选择。
- 部署效用只建模绝对表现、配对收益增量和 guarded step saving，不建模 token 成本。
- 不使用等权标量 utility；结构候选和部署候选分别执行 Pareto 管理。
- 第一轮不物理 Prune，不修改原 v17 snapshot；所有变更生成新 artifact。
- 图结构关系由确定性代码产生，LLM 只渲染已经确定的技能结构。

当前不属于本实验的内容：

- G1 Semantic-Only 重跑；它仍是独立构建消融。
- rewrite、Mid presence 等 D1-D3 机制消融；它们不参与本轮参数搜索。
- WebShop deployment refinement。
- 在线 bandit、NSGA-II、模型微调和 token 成本优化。

## 2. 数据协议

### 2.1 固定来源

- 基础 Tower：ALFWorld P310 Full v17，`tower_d2c2d0090ed9b6b4`。
- 基础构建池：310 个 train task，repeat 0/1/2/3，共 1,240 条 No-Skill 轨迹。
- 可用扩展任务：P1000 manifest 中未进入 P310 的 690 个 train task。

不得从既有 `valid_seen` 或 `valid_unseen` 结果反向选择任务、技能或阈值。正式运行前，
必须将 690 个任务按 `sample_id` 分组，用冻结 seed 在 task 层均匀随机划分。六个 ALFWorld
task family 只用于划分后的分布审计，不参与任务选择、图构建、结构候选或 Pareto 排名。

### 2.2 建议划分

以下数量是实施默认值，生成 manifest 前仍可调整；一旦生成并记录 SHA-256 后不得根据
结果改动：

| 子集 | task 数 | 用途 |
|---|---:|---|
| `deployment_feedback` | 450 | 产生反馈轨迹、效用统计和结构候选 |
| `deployment_gate` | 120 | 筛选候选 Tower，不回写效用或结构 |
| `deployment_holdout` | 120 | 方案冻结后一次性评测 |

另从 `deployment_feedback` 用独立冻结 seed 均匀抽取一个 60-task pilot。它只用于核对实际成本、rewrite
合同和结果完整性，是 feedback 的子集，不是第四个调优 split。pilot 完成后不得自动扩跑；
是否继续剩余 390 个 task 必须单独确认。

正式扩跑时生成 `deployment_feedback_remaining`，并强制满足：它与 pilot 交集为零，二者
并集严格等于 `deployment_feedback`。最终分析读取两个独立 run，禁止重跑或重复计入 pilot。

划分脚本必须输出：选择 seed、各 task family 数量、三个 manifest 的 SHA-256、与 P310
及彼此之间的交集审计。

### 2.3 Feedback rollout

`deployment_feedback` 上使用与主实验一致的 agent model、temperature、最大步骤和运行时
rewrite 合同。所有优化集只使用 `repeat_id=0`，不以重复执行估计运行方差。

No-Skill 不重新 rollout。它直接复用已审计的 P1000 全局轨迹池中对应 task 的
`repeat_id=0` 记录；冻结输入文件 SHA-256，并逐 task 校验 method、split、score 和 step 数。
只对冻结 Tower v0 运行一次在线 rollout，因此完整 feedback 的 rewrite 硬上限为 450，pilot
阶段为 60。

按 `(benchmark, sample_id, repeat_id)` 严格配对。缺失、重复、
未知 skill ID、snapshot 不一致、agent model 不一致或 manifest 越界都必须阻断 refinement。

失败样本可以形成额外的 proposal-discovery 视图，但不得单独用于 Pareto 目标估计；否则
会把困难任务过采样偏差写入效用。

## 3. Refinement 输入

候选结构的输入为：

```text
D_refine =
    P310 原始 No-Skill 轨迹
  + deployment-feedback Tower-v0 轨迹
```

No-Skill feedback 只作为配对控制，不进入结构重建。P310 的原始 1,240 条构建轨迹保持
不变，pilot 只追加 60 条 Tower-v0 部署轨迹；因此不会因重新抽样基础池而引入额外构建
差异，新增反馈也不会在 episode 数量上淹没基础图。

候选重建复用现有活动算法，不在本目录复制实现：

```text
preprocessing -> eigen_trace -> induction -> rendering -> artifacts
```

必须继续启用 `collapse_duplicate_embeddings=true`，并保持 Full 的图信号、随机种子、High
支持阈值和 renderer 合同。部署优化不得顺便改变主构建算法。

## 4. 新旧结构 Lineage

历史 segment ID 必须保持稳定。根据旧 Mid 与候选 Mid 共享的历史 segment 建立局部 overlap
graph，并在每个连通分量内分类：

```text
one old -> one new      continuation
one old -> many new     split
many old -> one new     merge
many old -> many new    recomposed
none -> one new         new_mid
one old -> none         disappeared_mid
```

主要 lineage 证据是：

- `shared_member_count`；
- `old_retention`；
- `new_historical_purity`。

Centroid similarity 只能作为 overlap 相同情况下的 tie-break 和 drift 诊断，不能单独产生
Split 或 Merge。完全由新增 segment 组成的 Mid 是 `new_mid`，不能伪装为历史 continuation。

## 5. 双层 Pareto

### 5.1 结构 Pareto

结构 Pareto 只判断 graph edit 是否值得成为候选，不使用部署 reward 替代图证据。每个局部
lineage proposal 计算三个越大越好的增量：

```text
outcome_consistency_gain
transition_role_coherence_gain
spectral_compactness_gain
```

no-op 向量固定为 `(0, 0, 0)`。合法候选必须在三个维度均不低于 no-op，并至少一个维度
严格提高。原始值、增量、支配关系和局部 lineage component 都必须进入报告。

结构 Pareto 只过滤由 lineage 代码证明合法的 Split/Merge；不得从 Pareto 排名反向发明
成员关系。

### 5.2 部署 Pareto

效用归因单元首先是实际检索到的原始 High ID 及其共同注入的 Mid bundle。Mid 没有独立
随机 exposure 时，不允许把完整 episode gain 解释为单张 Mid 的因果效用。

三个越大越好的目标为：

```text
performance_level = mean(tower_success)
paired_success_gain = mean(tower_success - no_skill_success)
guarded_step_saving = mean(guarded_step)
```

其中：

```text
raw_step = (no_skill_steps - tower_steps) / max(no_skill_steps, 1)

if tower_success < no_skill_success:
    guarded_step = min(raw_step, 0)
else:
    guarded_step = raw_step
```

使用 task bootstrap；每次 bootstrap 重新计算 Pareto front，并记录 `front_1_probability` 与
`dominated_probability`，避免用稀疏 exposure 的点估计直接修改技能状态。这里 bootstrap
重采样的是不同 task，并不要求为同一 task 额外生成 repeat。

建议默认门槛：

- exposure `< 10`：不进入排名；
- exposure `10..19`：只报告，不自动动作；
- exposure `>= 20`、`dominated_probability >= 0.90` 且
  `front_1_probability <= 0.10`：允许生成 Downweight proposal。

这些门槛属于执行前可调整项，冻结后不得根据 gate 结果改变。

## 6. 四类结构动作

### 6.1 Split

候选必须来自 `one old -> many new` lineage，且通过结构 no-op gate。排序优先考虑：

1. source Mid 的部署稳定性差；
2. 结构增量处于第一 Pareto front；
3. 子 Mid 具有足够历史 retention 和新增 feedback 支持；
4. stable ID tie-break。

### 6.2 Merge

候选必须来自 `many old -> one new` lineage，且通过结构 no-op gate。两个部署 Pareto F1、
功能互补的 Mid 不得仅因语义相似而合并。排序优先考虑共享历史 overlap，再以结构 front、
centroid drift 和 stable ID tie-break。

### 6.3 Promote

Promote 只能来自候选 Mid 有向转移图中新增或增强的 High path。路径必须满足 Full 的长度、
成功支持和 contrastive score 合同；其 child Mid 必须存在于候选 snapshot。排序优先考虑
结构合法、child bundle 部署稳定且 contrastive score 高的路径。

### 6.4 Downweight

Downweight 只改变部署策略，不改变 Tower 中已经发现的结构事实。它必须存放在引用具体
snapshot ID 的 deployment policy artifact 中。第一轮不删除卡片、不修改 embedding，也不
把 `status` 字段塞进 `MidCluster`、`HighPath` 或 `TowerSnapshot`。

## 7. 单轮预算与候选矩阵

每轮最多选择：

- Top-1 Split；
- Top-1 Merge；
- Top-1 Promote；
- Top-1 Downweight。

没有合法候选的动作直接跳过。先分别物化可归因候选，再物化兼容组合：

| 候选 | 内容 |
|---|---|
| `v0` | 冻结原 Tower |
| `split` | 仅 Top-1 Split 及必要重渲染 |
| `merge` | 仅 Top-1 Merge 及必要重渲染 |
| `promote` | 仅 Top-1 Promote |
| `downweight` | 仅部署策略状态变化 |
| `full` | 通过冲突检查的结构动作与 Downweight 组合 |

若动作作用于同一 lineage component 或使对方引用失效，则不得直接组合；必须记录冲突并
保持单动作候选。

## 8. Gate 与最终选择

`deployment_gate` 不参与图重建或效用更新。全部候选只执行 `repeat_id=0`；明显被 v0
支配的候选停止，不追加 repeat。

全局候选层面，绝对成功率与配对成功增量高度相关，因此成功表现作为非劣硬约束，不能
重复计权。建议默认 gate：

```text
paired success point estimate >= 0
one-sided 95% lower bound >= -0.03
```

满足 gate 的候选再按三目标 Pareto front 管理；需要唯一选择时，依次使用：

1. guarded step saving 的置信下界；
2. 修改动作更少；
3. stable candidate ID。

唯一选择规则必须在 rollout 前冻结。`deployment_holdout` 只运行 No-Skill、v0 和最终 v1，
不得依据结果继续修改 v1。`valid_seen`/`valid_unseen` 只允许在全部冻结后做确认性复评，且
必须声明它们已被历史实验观察，不作为新的未见泛化证据。

## 9. 代码隔离与职责

未来 Python 实现只能放在本目录，建议最小结构如下：

```text
deployment_optimization/
├── IMPLEMENTATION_PLAN.md
├── models.py       # 动作、lineage、proposal、report 的闭集模型
├── feedback.py     # manifest/run 审计、配对、三目标与 bootstrap
├── lineage.py      # 新旧 Mid overlap graph 与确定性分类
├── structural.py   # 三个结构增量和合法 proposal 生成
├── pareto.py       # 显式向量的非支配排序，不生成结构关系
└── refinement.py   # 调用现有构建阶段、物化候选与输出报告
```

职责边界：

- `core/` 继续只保存稳定 Tower 领域模型；不得提前加入 refinement 状态。
- `eigen_trace/` 与 `induction/` 继续提供通用算法，不感知实验 split 或 lifecycle。
- `deployment_optimization/` 拥有 lineage、proposal、Pareto 决策和 round orchestration。
- `artifacts/tower.py` 继续验证不可变结构 snapshot；候选 snapshot 仍走同一完整性合同。
- `inference/` 只消费已冻结 deployment policy，不负责计算效用或产生动作。
- manifest 生成、批量运行和报告命令放在 `scripts/experiments/` 的对应分层目录，不能把
  实验 CLI 塞入方法包。

生成数据不得写入 `src/`。建议 artifact 根目录：

```text
artifacts/trace2tower/alfworld/deployment-optimization-v1/
├── manifests/
├── feedback/
├── lineage/
├── proposals/
├── candidates/
├── gate/
└── selected/
```

## 10. 实施顺序

1. 冻结并审计三个 train-only manifest。
2. 定义真实消费者需要的 action、lineage、proposal 和 report 模型。
3. 实现 feedback 配对、guarded step 和 bootstrap Pareto。
4. 实现 deterministic lineage 与结构三指标。
5. 复用现有构建阶段生成候选结构，先不接 inference 状态。
6. 物化单动作候选并完成结构完整性测试。
7. 增加 deployment policy overlay，再接入 inference tie-break。
8. 运行 feedback，冻结 proposal 和候选矩阵。
9. 运行 gate，冻结唯一 v1。
10. 一次性运行 holdout，并生成最终报告。

## 11. 必须具备的测试

- 三个 manifest 与 P310 两两不相交，pilot 是 feedback 的严格子集。
- feedback 结果严格配对，错误 snapshot、模型、split、重复 key 会失败。
- guarded step 不奖励更快失败。
- bootstrap 以 task 为单位而不是 episode 为单位。
- continuation、Split、Merge、new/disappeared Mid lineage 均有确定性 fixture。
- 结构候选必须通过 no-op Pareto gate。
- Pareto 排序不归一化、不标量化，也不创建结构关系。
- 单轮每类动作最多一个，冲突动作不能组合。
- Downweight 不修改 Tower snapshot 内容或 snapshot ID。
- 候选 snapshot 满足 provenance、结构引用、card/index coverage 和哈希合同。
- 同配置、同输入、同 seed 生成完全相同的 proposal、候选 ID 和报告。

## 12. 执行前开放决策

正式落代码前仍需最终确认并写入配置：

- 60-task pilot 后是否扩展到完整 450-task feedback；
- manifest seed；
- feedback task/policy source 的图权重公式；
- 结构三个增量的精确定义与数值容差；
- exposure 和 bootstrap probability 门槛；
- success 非劣 margin；
- renderer 是否只处理新增/变化技能，或完整重渲染候选 Tower；
- gate 第一阶段的提前停止规则；
- Split/Merge/Promote/Downweight 的兼容性判定。

这些项目未冻结前，不创建活动实验配置，不运行 feedback，也不将本目录导出为稳定包 API。
