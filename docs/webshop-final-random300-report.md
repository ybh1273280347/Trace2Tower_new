# WebShop Random-300 正式测试报告

## 实验口径

本报告只汇报冻结配置在正式测试集上的结果，不使用测试结果反向选择配置。配置选择依据见 [WebShop 验证与正式测试口径](webshop-validation-and-final-test-protocol.md)。

正式测试使用预注册 selection `selection_32248afcaee8da76`：从此前未出现在任何 result 或 error 中的候选样本里，以三个固定随机种子各抽取 100 个任务，共 300 个互不重复的任务。每个任务执行 3 次，因此每个模型-方法条件包含 900 个 episode。

- 模型：`deepseek-v4-flash`、`deepseek-v4-pro`
- 方法：NoSkill、Flat cap3、Success-only Tower cap3、Mixed Tower cap3
- 满分成功：`primary_score >= 0.999`
- Completion：episode 以 `finish_reason = completed` 正常结束；它反映流程完成度，不等同于满分成功
- 置信区间：先在任务内平均 3 次重复，再对 300 个任务做 10,000 次 task-cluster bootstrap

8 个条件均完成 900/900 episode，共 7,200 个 episode；没有缺失 key、重复 key、越界 key或未解决错误。执行 manifest SHA-256 为 `b055ef5458374c0b8e34935dd59d83f1a90d023bf93fb6a7d2c27c61bcd8fc3e`。

## 主要结果

### 任务表现

| 模型 | 方法 | Mean reward | 满分成功率 | 相对 NoSkill reward 差（95% CI） | 相对 NoSkill 满分率差（95% CI） |
|---|---|---:|---:|---:|---:|
| Flash | NoSkill | 0.7037 | 49.7% | - | - |
| Flash | Flat cap3 | 0.7050 | 51.2% | +0.0013 `[-0.0242, +0.0276]` | +1.6% `[-1.7%, +4.9%]` |
| Flash | **Success-only Tower cap3** | **0.7186** | 51.3% | +0.0148 `[-0.0106, +0.0410]` | +1.7% `[-1.9%, +5.3%]` |
| Flash | Mixed Tower cap3 | 0.7153 | **51.7%** | +0.0116 `[-0.0137, +0.0379]` | +2.0% `[-1.7%, +5.8%]` |
| Pro | NoSkill | 0.6543 | 46.8% | - | - |
| Pro | Flat cap3 | 0.6930 | **52.9%** | +0.0387 `[+0.0069, +0.0715]` | +6.1% `[+2.6%, +9.8%]` |
| Pro | **Success-only Tower cap3** | **0.7029** | 51.9% | **+0.0486** `[+0.0176, +0.0801]` | **+5.1%** `[+1.3%, +8.9%]` |
| Pro | Mixed Tower cap3 | 0.6845 | 51.2% | +0.0303 `[-0.0021, +0.0634]` | +4.4% `[+0.7%, +8.2%]` |

Success-only Tower 在两个模型上都取得了当前主矩阵中最高的 mean reward。Flash 上的优势是点估计，区间仍跨零；Pro 上相对 NoSkill 的 reward 和满分成功率提升均得到 95% CI 支持。

### 完成度与执行成本

| 模型 | 方法 | Completion | 平均步数 | 每 episode 累计输入 token |
|---|---|---:|---:|---:|
| Flash | NoSkill | 90.1% | 7.45 | 18,886 |
| Flash | Flat cap3 | 88.9% | 8.54 | 27,892 |
| Flash | **Success-only Tower cap3** | 92.2% | **7.11** | 22,764 |
| Flash | Mixed Tower cap3 | **92.9%** | 7.19 | **22,140** |
| Pro | NoSkill | 81.4% | 8.69 | 23,664 |
| Pro | Flat cap3 | 84.7% | 9.12 | 30,500 |
| Pro | **Success-only Tower cap3** | **87.1%** | **8.12** | 26,709 |
| Pro | Mixed Tower cap3 | 85.8% | 8.22 | **26,518** |

输入 token 是 Agent 在一个 episode 中各步请求的累计输入量，包含固定提示、任务、交互历史和技能上下文，不包含技能构建成本。NoSkill 不注入技能，因此 token 最低；公平的结构与成本比较主要是 Tower 对 Flat。

## 主方法相对 Flat

| 模型 | Reward 差（95% CI） | 满分率差（95% CI） | Completion 差 | 平均步数差 | 输入 token 差 |
|---|---:|---:|---:|---:|---:|
| Flash | +0.0136 `[-0.0130, +0.0403]` | +0.1% `[-3.4%, +3.7%]` | +3.3% | -1.42 | -5,128（-18.4%） |
| Pro | +0.0099 `[-0.0175, +0.0377]` | -1.0% `[-4.0%, +1.9%]` | +2.4% | -1.00 | -3,791（-12.4%） |

Success-only Tower 相对 Flat 的 reward 差在两个模型上都是正值，但置信区间跨零，因此不能声称 reward 已显著优于 Flat。它的执行形态则一致更好：两个模型上 completion 都更高，平均步数都更少，输入 token 分别降低 18.4% 和 12.4%。

这一结果说明 Tower 的潜在价值不只在最终分数。Flat 将一组全局摘要整体注入上下文，增加了输入长度，却没有帮助模型更快结束任务；Tower 按当前状态检索少量相关技能，在保持当前最高 mean reward 的同时，减少了交互步数和上下文成本。效率指标目前作为描述性结果报告，未对其另做显著性检验。

## 结论

1. **Success-only Tower 是当前主方法。** 它在 Flash 和 Pro 上都取得主矩阵最高 mean reward，并且相对 Flat 有更高 completion、更少步骤和更低 token 成本。
2. **Pro 上的有效性已经建立。** Success-only Tower 相对 NoSkill 的 mean reward 提高 0.0486，满分成功率提高 5.1 个百分点，两项 95% CI 均不跨零。
3. **Flash 上保留谨慎表述。** Tower 的 reward、completion 和效率点估计更好，但 reward 与满分成功率区间跨零，尚不能宣称稳定性能提升。
4. **Tower 尚未显著击败 Flat，但效率优势清楚。** 两个模型上的 reward 点估计均高于 Flat；直接差异未显著，同时平均少 1.00 至 1.42 步，并节省 12.4% 至 18.4% 输入 token。
5. **Mixed 没有带来额外收益。** 它没有显著优于 Success-only Tower，因此保留为错误轨迹消融，默认配置仍使用 Success-only Tower cap3。

本 Random-300 只用于冻结配置的正式测试。后续配置改进必须在新的验证样本上完成，再使用新的未见测试样本评估。
