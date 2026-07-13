# WebShop 验证与正式测试口径

本文固定 Trace2Tower WebShop 实验的配置选择依据，并作为后续正式实验报告的验证协议章节。本文只使用主要配置验证集说明调参决策，不讨论轨迹池扩容过程或其他诊断实验。

## 主要配置验证集

- 样本：`webshop:50-149`，共 100 个独立任务。
- 重复：每个任务使用 `repeat_id = 0, 1, 2`，每个方法共 300 个 episode。
- 主要指标：mean reward 与满分成功率；满分定义固定为 `primary_score >= 0.999`。
- 统计单位：先在同一任务内平均三次重复，再以 100 个任务为 cluster 做 10,000 次 bootstrap，置信水平 95%。三次重复不作为三个独立样本。
- 配置选择顺序：满分成功率优先，其次 mean reward，最后比较输入 token、步数和无效动作成本。
- 验证集只用于选择一次配置；配置冻结后，不得根据正式测试结果修改 cap、证据策略、技能 artifact 或检索参数。

## Flat Skill 配置选择

Flat 使用 staged retrieval：先取 Top-100 候选，再应用绝对相似度阈值 `0.45`、相对最佳分数 margin `0.08`、相似度高于 `0.95` 的近重复去除，以及 MMR relevance weight `0.75`。

验证集上，最多注入八张卡的 staged-cap8 相对 Top-3 reward 下降 `0.0404`，95% CI `[-0.0795, -0.0059]`。GPT-5.4 self-filter 虽然缩短了上下文，但没有恢复 reward。更多卡片因此构成可重复的上下文噪声，而不是额外收益。

最终冻结：

- `retrieval_strategy: diverse`
- `flat_candidate_top_k: 100`
- `flat_similarity_threshold: 0.45`
- `flat_relative_margin: 0.08`
- `flat_dedup_similarity_threshold: 0.95`
- `flat_mmr_lambda: 0.75`
- `flat_top_k: 3`

## Tower Cap 与证据策略选择

主要验证集同时比较 success-only 与 mixed 两种证据策略，以及直接 Mid cap 3、5、8、12。

| Cap | Success-only reward | Success-only 满分率 | Mixed reward | Mixed 满分率 | Success-only 每 episode 累计输入 token | Mixed 每 episode 累计输入 token |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.6966 | 48.7% | 0.6748 | 44.0% | 25,035 | 23,693 |
| 5 | 0.7043 | 45.0% | 0.6798 | 45.0% | 27,187 | 23,804 |
| 8 | 0.6760 | 44.3% | 0.6862 | 45.7% | 32,088 | 24,268 |
| 12 | 0.6756 | 44.0% | 0.6748 | 43.7% | 32,286 | 25,276 |

这里的输入 token 是 Agent 完成一个 episode 时各步模型请求的累计输入量，包含固定提示、任务、逐步历史 observation/action 和重复携带的技能上下文；它不是技能构建成本，也不是入选训练轨迹数。Success-only 与 mixed 会形成不同的 Mid 聚类、卡片长度和向量分布，因此即使 cap 相同，经过阈值、去重和 MMR 后实际注入的卡数与文本长度也不同。以 cap8 为例，success-only 平均技能上下文为 7,285 字符、记录 7.95 个唯一 Mid ID、运行 8.09 步；mixed 分别为 3,845 字符、5.35 个 Mid ID和 7.50 步，所以 success-only 的累计输入 token 更高。

按“满分成功优先”的冻结规则，success-only cap3 的满分率 `48.7%` 为标准控制中的最高值。相对 success-only cap8，cap3 的满分率高 `4.3%`，95% CI `[0.7%, 9.0%]`，并且每个 episode 少约 7,053 个输入 token。cap5 的 reward 略高，但满分率低 `3.7%` 且成本更高，不能取代 cap3。

在 cap3 下，mixed 相对 success-only 的 reward 差为 `-0.0218`，95% CI `[-0.0661, 0.0208]`；满分成功率低 `4.7%`，95% CI `[-9.0%, -1.0%]`。因此 mixed 不能作为默认策略。它仍然保留在正式实验矩阵中，因为“保留有信息价值的错误轨迹”是核心消融问题；保留该方法不代表使用验证集宣称它优于 success-only。

最终冻结：

- Tower 默认直接 Mid cap：`3`
- High top-k：`1`
- 检索策略：与 Flat 相同的 staged/diverse retrieval
- 默认证据策略：`success-only`
- 关键消融：`mixed`，使用同一 cap3 预算
- 不进入正式主矩阵：cap5、cap8、cap12、self-filter、Mid-only 或交叉 High 变体

## 正式测试预注册

正式测试在配置冻结后，从抽样前从未出现在任何实验 result 或 error 中的 WebShop 样本里抽取。选择协议为 `selection_32248afcaee8da76`：

- 三个固定抽样 seed：`20260713`、`20260714`、`20260715`。
- 每个 seed 无放回抽取 100 个任务，三个集合之间也不重叠。
- 合计 300 个独立任务；每任务重复三次；每个模型-方法组合共 900 个 episode。
- 执行 manifest SHA-256：`b055ef5458374c0b8e34935dd59d83f1a90d023bf93fb6a7d2c27c61bcd8fc3e`。

正式矩阵固定为两个执行模型：

- `deepseek-v4-flash`
- `deepseek-v4-pro`

每个模型固定运行四种方法：

- NoSkill
- Flat Skill cap3
- Success-only Tower cap3
- Mixed Tower cap3

报告先分别给出三个 seed 的 100-task/300-episode 结果，再给出合并后的 300-task/900-episode 结果。合并置信区间以 300 个 `sample_id` 为 cluster；技能方法只与同模型、相同 `(sample_id, repeat_id)` 的 NoSkill 做配对比较。

正式测试结果只用于评估冻结方法的泛化表现，不再用于选择新配置。若正式测试显示某个未冻结变体可能更优，该观察只能成为后续独立实验的假设，不能回写本轮主结果。
