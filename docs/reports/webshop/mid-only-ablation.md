# WebShop Mid-only 机制消融

## 实验设计

本消融用于定位 Tower 中 Mid 与 High 两层对 success-only/mixed 差异的影响。它复用正式 Random-300 的 300 个任务、3 次重复、cap3 artifact 和执行模型，只把 `high_similarity_threshold` 从 `0.45` 提高到 `1.0`，使 High 卡不被注入。技能库、Mid 检索参数和 episode key 均保持不变。

新增四个 Mid-only 条件：success-only/mixed × Flash/Pro。每个条件均完成 900/900 episode，没有重复 key、缺失 key或未解决 error。置信区间仍先在任务内平均 3 次重复，再对 300 个任务做 10,000 次 task-cluster bootstrap。

这是在主矩阵完成后追加的机制消融，不用于反向修改正式配置。

## 汇总结果

| 模型 | Evidence | High | Mean reward |     满分成功率 | Completion |     平均步数 |   输入 token |
|---|---|---|------------:|----------:|-----------:|---------:|-----------:|
| Flash | Success-only | Full |      0.7186 | **51.3%** |  **92.2%** | **7.11** |     22,764 |
| Flash | Success-only | Mid-only |  **0.7190** |     50.9% |      91.8% |     7.41 | **22,442** |
| Flash | Mixed | Full |  **0.7153** |     51.7% |  **92.9%** | **7.19** | **22,140** |
| Flash | Mixed | Mid-only |      0.6937 |     49.6% |      88.6% |     8.08 |     24,266 |
| Pro | Success-only | Full |  **0.7029** |     51.9% |      87.1% | **8.12** | **26,709** |
| Pro | Success-only | Mid-only |      0.6860 | **52.2%** |      84.1% |     8.96 |     28,982 |
| Pro | Mixed | Full |  **0.6845** | **51.2%** |      85.8% | **8.22** | **26,518** |
| Pro | Mixed | Mid-only |      0.6545 |     49.4% |      80.6% |     9.33 |     29,270 |

## High 层增量

下表报告 `Full - Mid-only`。正值表示加入 High 后指标提高。

| 模型 | Evidence | Reward 差（95% CI） | 满分率差（95% CI） |
|---|---|---:|---:|
| Flash | Success-only | -0.0005 `[-0.0237, +0.0230]` | +0.4% `[-2.4%, +3.4%]` |
| Flash | Mixed | +0.0216 `[-0.0073, +0.0512]` | +2.1% `[-1.4%, +5.7%]` |
| Pro | Success-only | +0.0169 `[-0.0098, +0.0441]` | -0.3% `[-3.2%, +2.7%]` |
| Pro | Mixed | +0.0301 `[-0.0019, +0.0617]` | +1.8% `[-1.8%, +5.3%]` |

Mixed 的完整补偿模式如下，差值均为 `Full - Mid-only`：

| 模型 | Reward 差 | Completion 差 | 平均步数差 | 输入 token 差 |
|---|---:|---:|---:|---:|
| Flash | +0.0216 | +4.3% | -0.89 | -2,126 |
| Pro | +0.0301 | +5.2% | -1.11 | -2,752 |

四个指标在两个模型上的方向完全一致：High 为较弱的 mixed Mid 提供补偿，同时带来更高 reward、更高 completion、更少步骤和更少 token。执行效率上的一致模式是清楚的；reward 与满分率的单项置信区间仍跨零，因此不把它写成“High 的 reward 增益已经显著”。

## Mixed 差异定位

下表报告同一模型和层级下的 `Mixed - Success-only`。

| 模型 | 层级 | Reward 差（95% CI） | 满分率差（95% CI） |
|---|---|---:|---:|
| Flash | Full Tower | -0.0033 `[-0.0271, +0.0207]` | +0.3% `[-2.9%, +3.6%]` |
| Flash | Mid-only | **-0.0253** `[-0.0480, -0.0028]` | -1.3% `[-4.3%, +1.7%]` |
| Pro | Full Tower | -0.0183 `[-0.0469, +0.0100]` | -0.7% `[-4.1%, +2.8%]` |
| Pro | Mid-only | **-0.0315** `[-0.0610, -0.0014]` | -2.8% `[-5.8%, +0.2%]` |

稳定信号出现在 Mid-only：Mixed 的 reward 在 Flash 和 Pro 上都显著低于 Success-only。恢复 High 后，两模型的 mixed-success reward 区间都跨零。High×evidence 的 reward 交互在 Flash 为 `+0.0221`、CI `[-0.0091, +0.0549]`，在 Pro 为 `+0.0132`、CI `[-0.0270, +0.0535]`，均未显著。

因此，当前证据指向 mixed 的弱点主要出现在 Mid 层，而 High 具有补偿 Mid 的功能；它不支持“High 导致 mixed 失效”。由于 High×evidence 交互项尚未显著，这仍是机制定位而非完整因果证明。

## High 交叉实验

为进一步区分“High 本身提供补偿”与“High 只对同源 Mid 有效”，在同一 Random-300 上交换 success-only 与 mixed 的 High 卡，Mid 库保持不变。四个新增条件（2 个 Mid 来源 × 2 个模型）均完成 900/900 episode，零重复、零缺失、零 error。High 的结构证据相同，只交换渲染内容及对应检索向量；该实验是主实验之后的机制分析，不参与配置选择。

下表报告交叉 High 相对 Mid-only 的增量：

| 模型 | 固定 Mid | 交叉接入 High | Reward 差（95% CI） | 满分率差（95% CI） | 步数差 | 输入 token 差 |
|---|---|---|---:|---:|---:|---:|
| Flash | Success-only | Mixed High | -0.0112 `[-0.0382, +0.0153]` | -0.9% `[-4.2%, +2.4%]` | -0.48 | -802 |
| Flash | Mixed | Success-only High | +0.0098 `[-0.0163, +0.0364]` | +1.7% `[-1.3%, +4.7%]` | -0.84 | -1,685 |
| Pro | Success-only | Mixed High | -0.0038 `[-0.0307, +0.0227]` | -1.3% `[-4.6%, +2.0%]` | -0.71 | -1,664 |
| Pro | Mixed | Success-only High | **+0.0363** `[+0.0064, +0.0668]` | +0.9% `[-2.1%, +4.0%]` | -1.39 | -4,335 |

交叉结果强化了补偿机制：success-only High 接到 mixed Mid 后，两模型都提高 reward、减少步数和输入 token；其中 Pro 的 reward 增益置信区间完全高于零。反向把 mixed High 接到 success-only Mid 时，reward 略降但区间均跨零，执行仍更短、更省 token。

同一 Mid 上比较原同源 High 与交叉 High，四个 reward 差异的置信区间均跨零。因此目前不能声称 High 必须与 Mid 同源，也不能声称某一种 High 普遍更优；能够稳定落下的是，High 的补偿可以跨 evidence 库迁移，且对较弱的 mixed Mid 更有价值。完整机器可读统计位于 `artifacts/evaluations/webshop-cross-high-random300-v1/report.json`。

本消融不参与配置选择。统一机制解释和 WebShop 封版边界见 [Trace2Tower WebShop 完整实验报告](complete-experiment-report.md)。
