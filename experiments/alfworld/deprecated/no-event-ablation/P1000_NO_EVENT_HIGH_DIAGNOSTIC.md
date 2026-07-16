# ALFWorld P1000 No-Event High 路径消融

## 结论

在保持 `high_min_support_ratio = 0.02`、`max_high_path_length = 4` 且不启用 High fallback 的正式配置下，将 ALFWorld 全局训练池从 310 个任务补到 1000 个任务后，仍未产生自然 High 路径。

这次实验排除了“旧训练池太小”作为唯一原因。P1000 中确实出现了成功偏置路径，但最强路径只覆盖 2 个成功样本，而 2% 正式门槛要求至少覆盖 14 个成功样本；达到 2% 支持率的公共路径又全部更偏向失败轨迹，因此对比得分为负。

## 训练池构建

P1000 保留原有 310 个已审计任务，并以固定 seed `2903` 从未使用的完整 train split 中全局抽取 690 个任务。抽样不按任务族分层或过滤。每个任务保留 repeat `0,1,2,3`。

| 指标 | 原训练池 | P1000 |
|---|---:|---:|
| 任务数 | 310 | 1000 |
| 轨迹数 | 1240 | 4000 |
| 成功轨迹数 | 850 | 2489 |
| 成功率 | 68.55% | 62.23% |
| Transition 数 | - | 56,355 |
| Segment 数 | 6,098 | 19,268 |
| Mid cluster 数 | 9 | 7 |
| 正式自然 High 数 | 0 | 0 |
| 使用 High fallback | 否 | 否 |

新增 rollout 原始结果含 2766 行、2760 个唯一 `(sample_id, repeat_id)`。6 条重复记录的分数和步数均一致，合并时确定性去重，最终 P1000 恰好包含 4000 条轨迹。

## High 支持率诊断

诊断只统计正式图上的候选支持率，没有改变正式 2% 阈值，也没有将低支持路径写入 Tower。

| 指标 | 结果 |
|---|---:|
| 至少有一次观测的候选路径 | 1250 |
| 达到 2% 成功支持率的路径 | 6 |
| 对比得分为正的路径 | 26 |
| 同时满足两项的正式 High | 0 |
| 成功样本数 | 661 |
| 失败样本数 | 412 |
| 2% 所需成功样本数 | 14 |
| 最强成功偏置路径覆盖 | 2 / 661（0.30%） |

同一任务的四次 rollout 可能同时包含成功和失败，因此成功样本集合与失败样本集合允许重叠；这里沿用正式 High 挖掘实现按 `sample_id` 分别去重的统计口径。

支持率最高的路径 `mid_0000 -> mid_0003` 覆盖 19/661 个成功样本，但也覆盖 121/412 个失败样本，其成功支持率为 2.87%，失败支持率为 29.37%，对比得分为负。其余五条达到 2% 的路径也全部更常见于失败样本。

## 判断

P1000 已覆盖完整 3553 个 train 任务中的全局随机 1000 个任务。现有差距不是从 1.8% 补到 2% 这样的边界波动，而是成功偏置路径仅有 0.30% 支持率。继续随机补少量任务不太可能改变结论；直接扩到全量则还需要 2553 个任务、约 10,212 次 rollout，成本明显，但没有证据表明同一成功偏置路径会从 2 个样本增长到正式门槛。

因此，后续若继续保持 2% 阈值，应优先检验“精确连续 Mid ID 路径是否把同一策略拆散”，而不是把降低阈值作为补救。这个判断针对路径表示和支持率统计，不否定事件抽取作为通用建图步骤。

## 产物

- 合并轨迹：`artifacts/trajectories/alfworld/alfworld-pool-p1000-global.jsonl`
- 合并审计：`artifacts/trajectories/alfworld/alfworld-pool-p1000-global.audit.json`
- 预处理结果：`artifacts/trace2tower/alfworld/deprecated/original-concept-v1/p1000/preprocessed-pool.jsonl`
- P1000 图：`artifacts/trace2tower/alfworld/deprecated/original-concept-v1/p1000/graph`
- 结构挖掘：`artifacts/trace2tower/alfworld/deprecated/original-concept-v1/p1000/skills-structure`
