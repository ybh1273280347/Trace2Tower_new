# WebShop 验证与正式测试口径

本文固定 Trace2Tower WebShop 的配置选择依据。验证集只用于选择配置，不作为最终性能证据；正式测试结果不得反向修改 cap、证据策略、技能 artifact 或检索参数。

## 验证集与统计口径

- 样本：`webshop:50-149`，共 100 个任务。
- 重复：每个任务使用 `repeat_id = 0, 1, 2`，每个条件共 300 个 episode。
- 执行模型：`deepseek-v4-flash`。
- 主要指标：mean reward 与满分成功率；满分定义为 `primary_score >= 0.999`。
- 统计单位：先在同一任务内平均 3 次重复，再以 100 个任务为 cluster 做 10,000 次 bootstrap，置信水平 95%。
- 选择顺序：满分成功率优先，其次 mean reward，最后比较输入 token、步数和无效动作成本。

## 数据完整性审计

本次审计直接读取各 run 的 `results.jsonl`、`resolved-config.yaml` 和绑定 artifact，不复用旧汇总文件。审计覆盖 NoSkill、Flat cap3/cap8，以及 success-only/mixed Tower 的 cap3/5/8/12，共 11 个条件。

| 检查项 | 结果 |
|---|---|
| Episode 覆盖 | 每个条件均为 300/300 |
| Key 集合 | 每组都严格等于 100 个任务 × 3 repeats |
| 重复或越界 key | 0 |
| 未解决 error | 0 |
| Agent 模型 | 全部为 `deepseek-v4-flash` |
| 输入 token 缺失 | 0/3,300 |
| Flat 可比性 | cap3 与 cap8 使用同一 library，除 `flat_top_k` 外检索配置相同 |
| Tower 可比性 | 同一证据策略的四个 cap 使用完全相同的训练轨迹、技能卡、聚类和向量索引，仅 cap 契约不同 |
| 训练来源 | Flat 与 success-only 使用 94 条满分训练轨迹；mixed 使用 173 条训练轨迹；全部来自 train split |
| 训练/验证重叠 | 0 |

因此主要 rollout 数据是干净且可配对的，不需要重新调用模型。旧版 Flat CI 与本文声明的 task-cluster 口径不一致；本文已从原始 episode 重新计算并更正，不能沿用旧的显著性表述。

## Flat 配置选择

Flat staged retrieval 固定使用 Top-100 候选、绝对相似度阈值 `0.45`、相对最佳分数 margin `0.08`、近重复阈值 `0.95` 和 MMR relevance weight `0.75`，只比较最终注入上限 3 与 8。

| Cap | Mean reward | 满分成功率 | Completion | 平均步数 | 每 episode 输入 token |
|---:|---:|---:|---:|---:|---:|
| 3 | **0.7276** | **47.3%** | **94.7%** | **8.18** | **25,082** |
| 8 | 0.6871 | 44.3% | 91.0% | 8.42 | 35,119 |

cap3 相对 cap8 的 reward 差为 `+0.0405`，95% CI `[-0.0014, +0.0843]`；满分成功率差为 `+3.0%`，95% CI `[-1.0%, +7.0%]`。两项区间都跨零，因此不能声称 cap3 已显著优于 cap8。

配置选择仍然明确：cap3 的满分率和 reward 点估计都更高，每 episode 少 10,038 个输入 token，平均步数也更少。按预先固定的选择顺序，最终冻结 `flat_top_k: 3`。这里的结论是“cap3 在验证目标和成本上占优”，不是“更多卡片已被统计证明有害”。

## Tower 配置选择

下表所有数字均由原始 300 个 episode 重算。纵向格式避免把 evidence strategy 与 cap 两个变量混在同一单元格中。

| Evidence | Cap | Mean reward | 满分成功率 | Completion | 平均步数 | 每 episode 输入 token |
|---|---:|---:|---:|---:|---:|---:|
| Success-only | 3 | 0.6966 | **48.7%** | 93.3% | **7.48** | **25,035** |
| Success-only | 5 | **0.7043** | 45.0% | 93.0% | 7.64 | 27,187 |
| Success-only | 8 | 0.6760 | 44.3% | 90.3% | 8.09 | 32,088 |
| Success-only | 12 | 0.6756 | 44.0% | 91.0% | 7.99 | 32,286 |
| Mixed | 3 | 0.6748 | 44.0% | 93.0% | **7.39** | **23,693** |
| Mixed | 5 | 0.6798 | 45.0% | **94.0%** | 7.43 | 23,804 |
| Mixed | 8 | **0.6862** | **45.7%** | 93.7% | 7.50 | 24,268 |
| Mixed | 12 | 0.6748 | 43.7% | 93.3% | 7.58 | 25,276 |

### Cap 选择

Success-only 是标准控制。cap3 的满分成功率 `48.7%` 为该策略最高值：

- 相对 cap8，cap3 满分率高 `4.3%`，95% CI `[+0.7%, +9.0%]`；reward 高 `0.0206`，CI `[-0.0189, +0.0624]`；每 episode 少 7,053 个输入 token。
- 相对 cap5，cap3 满分率高 `3.7%`，CI `[0.0%, +8.0%]`；reward 低 `0.0076`，CI `[-0.0493, +0.0340]`；每 episode 少 2,151 个输入 token。

cap5 只有 reward 点估计略高，满分率更低且成本更高。按“满分成功率优先”的冻结规则，Tower 最终选择 cap3。

## Token 口径

输入 token 是 Agent 在一个 episode 中各步模型请求的累计输入量，包括固定提示、任务、逐步交互历史和重复携带的技能上下文；它不是技能构建成本，也不是入选训练轨迹数。

Success-only 与 mixed 会形成不同的技能卡和向量分布。经过相似度阈值、去重和 MMR 后，两者实际注入的卡数、文本长度和执行步数都可能不同，因此 success-only 的累计输入 token 高于 mixed 并不矛盾。

## 冻结配置

| 组件 | 正式配置 |
|---|---|
| Flat | staged/diverse retrieval，cap3 |
| Tower | staged/diverse direct-Mid retrieval，cap3 |
| High | top-k 1 |

cap5、cap8、cap12、self-filter、Mid-only 和交叉 High 变体均不进入正式主矩阵。

## 正式测试边界

正式测试使用预注册 selection `selection_32248afcaee8da76`，包含 300 个此前未出现在任何 result 或 error 中的任务，每任务重复 3 次。执行 manifest SHA-256 为 `b055ef5458374c0b8e34935dd59d83f1a90d023bf93fb6a7d2c27c61bcd8fc3e`。

正式矩阵固定为两个模型（Flash、Pro）和五种方法（NoSkill、Flat cap3、官方 SkillX、Success-only Tower cap3、Mixed Tower cap3）。SkillX 是只使用成功轨迹的外部 baseline，不参与 cap 或 evidence 选择。正式结果只评估冻结方法的泛化表现，不再用于选择新配置。

WebShop 实验线已在正式 baseline、Mid-only 和 High 交叉实验完成后封版。完整结论见 [Trace2Tower WebShop 完整实验报告](webshop-complete-experiment-report.md)，可验证状态见 `webshop-freeze-manifest.json`。
