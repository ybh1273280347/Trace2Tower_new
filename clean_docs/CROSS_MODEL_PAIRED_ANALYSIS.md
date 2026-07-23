# Trace2Tower 跨模型执行与翻转配对分析

日期：2026-07-22

## 1. 结论摘要

第一，Trace2Tower 在三个执行模型上均能提高主指标。ALFWorld 中，DeepSeek V4 Flash、GPT-5.4 和 DeepSeek V4 Pro 接入 Tower 后分别提升 35.07、33.58 和 19.40 个百分点；WebShop 中，三者的 mean reward 分别提高 0.06242、0.04658 和 0.01642。

第二，模型差异主要表现为逐任务成功集合的重新分配，而非严格的能力支配。ALFWorld 上，GPT-5.4 与 Flash 的 No-Skill 成功率仅差 1.49 个百分点，但有 48 个任务发生相反结果；Tower 条件下仍有 18 个任务发生翻转。DeepSeek Pro 的 No-Skill 显著高于 Flash，但使用同一个 GPT-5.4 编写的 Tower 后，两者收敛到 85.82% 和 88.06%。因此，Flash 的高分不能简化为“弱模型偶然超过强模型”，更合理的表述是：不同执行模型与任务、技能表达和环境交互协议之间存在明显适配差异。

第三，补齐的 ALFWorld 2×2 控制矩阵表明，图结构价值与文本渲染质量应分开讨论。DeepSeek V4 Flash 编写的 Tower 在 Flash 和 Pro 上分别达到 65.67% 和 79.10%，相对各自 No-Skill 都提高 12.69 个百分点；GPT-5.4 编写的同构 Tower 则分别达到 88.06% 和 85.82%。因此，较弱渲染器生成的技能仍具有跨执行器正增益，但 GPT-5.4 的技能表达提供了不可忽略的额外收益。

WebShop 的现象更直接：Flash 在 No-Skill 和 Tower 两个条件下的 mean reward 都高于 GPT-5.4 与 DeepSeek Pro。由于这一优势在无技能条件下已经出现，不能将差距全部归因于 Tower；同时 Tower 对三个执行器的 mean reward 都有正向点估计，说明图技能仍具有跨模型可用性。

## 2. 实验协议

所有条件使用相同冻结评估集：ALFWorld 134 个任务，WebShop 100 个任务。采用统一的 Trace2Tower 运行时合同：Top-3 High 检索、plan rewrite、按 rewritten step 检索 Mid，并联合注入 High 与 Mid。WebShop 没有关闭检索或使用独立的简化路径。

本报告区分两类交叉实验：

| 条件                | Tower 技能作者        | 执行模型              | 可解释边界                                      |
|-------------------|-------------------|-------------------|--------------------------------------------|
| Flash 主实验         | GPT-5.4           | DeepSeek V4 Flash | 正式主实验                                      |
| GPT-5.4 交叉实验      | DeepSeek V4 Flash | GPT-5.4           | 同时改变技能文本作者与执行模型，用于检验跨模型互操作                 |
| DeepSeek Pro 泛化实验 | GPT-5.4           | DeepSeek V4 Pro   | 与 Flash 主实验复用同一 GPT-5.4 Tower，较干净地检验执行模型迁移 |

GPT-5.4 交叉实验保留相同的图构建、层级发现与路径挖掘，只重新渲染技能卡及其索引。因此该条件可以说明“Flash 编写的 Tower 能否由 GPT-5.4 使用”，但不应把它与 Flash 主实验的差值全部归因于执行模型。

### 2.1 ALFWorld 作者×执行器控制矩阵

四个条件使用同一 P310 证据、同一图算法、同一 134 题测试清单和同一 `plan_rewrite + budgeted_v2` 运行合同。行只改变技能卡作者，列只改变执行模型；测试时 rewrite 始终由统一 renderer 端点执行，不属于技能作者变量。WebShop 因跨分区退化尚需进一步考虑，本轮不新增对应实验。

| 技能作者 | DeepSeek V4 Flash 执行 | DeepSeek V4 Pro 执行 |
|---|---:|---:|
| GPT-5.4 | **118/134（88.06%）** | **115/134（85.82%）** |
| DeepSeek V4 Flash | 88/134（65.67%） | 106/134（79.10%） |

| 技能作者 | Flash 执行：相对 No-Skill | Pro 执行：相对 No-Skill |
|---|---:|---:|
| GPT-5.4 | +35.07 pp | +19.40 pp |
| DeepSeek V4 Flash | +12.69 pp | +12.69 pp |

固定 Flash 执行器时，GPT-5.4 作者相对 Flash 作者有 34 个独赢、4 个反向独赢，净增 22.39 个百分点，McNemar exact `p=6.04e-7`。固定 Pro 执行器时对应翻转为 20/11，净增 6.72 个百分点，`p=0.150`。这说明作者效应与执行器存在交互：GPT-5.4 渲染对 Flash 执行器帮助更大，而 Pro 对较弱技能文本更稳健。后一个差值在本轮配对样本上未达到显著水平，不作强归因。

## 3. 汇总结果

### 3.1 ALFWorld

| 执行模型              | Tower 技能作者        | No-Skill | Trace2Tower |      Tower 增益 | Tower 平均步骤 |
|-------------------|-------------------|---------:|------------:|--------------:|-----------:|
| DeepSeek V4 Flash | GPT-5.4           |   52.99% |  **88.06%** | **+35.07 pp** |      10.06 |
| GPT-5.4           | DeepSeek V4 Flash |   51.49% |  **85.07%** | **+33.58 pp** |      10.22 |
| DeepSeek V4 Pro   | GPT-5.4           |   66.42% |  **85.82%** | **+19.40 pp** |   **9.90** |

GPT-5.4 的 No-Skill 与 Flash 基本相当，Tower 增益也相近。DeepSeek Pro 的基础能力更高，因此 Tower 的绝对边际增益较小，但最终成功率仍从 66.42% 提高到 85.82%。三个既有交叉条件的 Tower 成功率落在 85.07%–88.06% 的窄区间内，说明图技能可被多种执行模型使用；补齐作者控制后同时表明，绝对性能仍受技能作者与执行器适配影响。

### 3.2 WebShop

| 执行模型              | Tower 技能作者        | No-Skill mean reward | Trace2Tower mean reward |     Tower 增益 | No-Skill / Tower 满分率 |
|-------------------|-------------------|---------------------:|------------------------:|-------------:|---------------------:|
| DeepSeek V4 Flash | GPT-5.4           |              0.65235 |             **0.71477** | **+0.06242** |            47% / 52% |
| GPT-5.4           | DeepSeek V4 Flash |              0.60097 |             **0.64755** | **+0.04658** |            37% / 41% |
| DeepSeek V4 Pro   | GPT-5.4           |              0.61125 |             **0.62767** | **+0.01642** |            49% / 46% |

WebShop 的主指标是连续 reward。DeepSeek Pro 接入 Tower 后 mean reward 提高 0.01642，但满分率下降 3 个百分点，说明其收益来自部分完成质量，而不是更多任务达到满分。该结果应按 mean reward 报告，不能用满分率替代主指标，也不应掩盖两个指标方向不同这一事实。

## 4. Tower 与 No-Skill 的逐样本翻转

### 4.1 ALFWorld 成功翻转

“Tower 独赢”表示同一任务上 Tower 成功而 No-Skill 失败；“No-Skill 独赢”含义相反。

| 执行模型              | Tower 独赢 | No-Skill 独赢 | 同成功 | 同失败 | 净翻转 |
|-------------------|---------:|------------:|----:|----:|----:|
| DeepSeek V4 Flash |       51 |           4 |  67 |  12 | +47 |
| GPT-5.4           |       53 |           8 |  61 |  12 | +45 |
| DeepSeek V4 Pro   |       33 |           7 |  82 |  12 | +26 |

三种模型的正向翻转均远多于负向翻转。尤其 GPT-5.4 虽然最终 Tower 分数略低于 Flash，但它在自身 No-Skill 基线上产生了最多的正向翻转（53 个）。这说明应分别讨论“某模型最终分数”和“Tower 对该模型的因果配对增益”，两者不是同一个问题。

### 4.2 WebShop reward 翻转

| 执行模型              | Tower reward 更高 | No-Skill reward 更高 | 相等 | 平均 reward 差 |
|-------------------|----------------:|-------------------:|---:|------------:|
| DeepSeek V4 Flash |              16 |                  9 | 75 |    +0.06242 |
| GPT-5.4           |              26 |                 17 | 57 |    +0.04658 |
| DeepSeek V4 Pro   |              19 |                 16 | 65 |    +0.01642 |

WebShop 中大量任务 reward 相等，而少数任务上的改善幅度决定了总体均值。GPT-5.4 的正向任务数最多，但 Flash 的平均收益最大，说明只比较“赢了多少题”也不足以描述连续奖励；需要同时报告胜负数量和平均差值。

## 5. 执行模型之间的翻转

### 5.1 ALFWorld

下表均以目标模型与 Flash 配对比较。“目标独赢”表示目标模型成功、Flash 失败。

| 条件                             | 目标独赢 | Flash 独赢 | 同成功 | 同失败 |       总分差 |
|--------------------------------|-----:|---------:|----:|----:|----------:|
| GPT-5.4 vs Flash，No-Skill      |   23 |       25 |  46 |  40 |  -1.49 pp |
| GPT-5.4 vs Flash，Tower         |    7 |       11 | 107 |   9 |  -2.99 pp |
| DeepSeek Pro vs Flash，No-Skill |   33 |       15 |  56 |  30 | +13.43 pp |
| DeepSeek Pro vs Flash，Tower    |    3 |        6 | 112 |  13 |  -2.24 pp |

GPT-5.4 与 Flash 的 No-Skill 翻转几乎完全对称（23 对 25），不构成稳定的强弱排序。DeepSeek Pro 在 No-Skill 下有明确优势（33 对 15），但 Tower 条件下两者绝大多数任务共同成功（112 个），最终只差 3 题。一个谨慎且数据一致的解释是：Tower 将多种执行模型推向相近的任务上限，因而压缩了原始模型差距；本轮单次执行不足以进一步区分上限、随机性和模型—提示适配各自贡献。

### 5.2 WebShop

| 条件                             | 目标 reward 更高 | Flash reward 更高 | 相等 |    目标均值差 |
|--------------------------------|-------------:|----------------:|---:|---------:|
| GPT-5.4 vs Flash，No-Skill      |           19 |              34 | 47 | -0.05138 |
| GPT-5.4 vs Flash，Tower         |           12 |              27 | 61 | -0.06722 |
| DeepSeek Pro vs Flash，No-Skill |           13 |              19 | 68 | -0.04110 |
| DeepSeek Pro vs Flash，Tower    |           10 |              21 | 69 | -0.08710 |

WebShop 上 Flash 的领先在 No-Skill 条件下已经存在，并在 Tower 条件下延续。这更像是执行模型与 WebShop 的网页状态表达、候选比较和动作协议之间的整体适配差异，而不是 Tower 单一模块造成的反转。现有数据可以确认现象和范围，但不能仅凭结果对内部原因作唯一归因。

## 6. 图形化数据挖掘

### 6.1 模型与方法的交互增益

![Cross-model Tower gains](figures/cross-model-tower-gains.png)

斜率图同时保留各模型的起点、终点和 Tower 增益。ALFWorld 中，DeepSeek Pro 的 No-Skill 起点最高，但三种模型在 Tower 条件下收敛到 85.07%–88.06%；WebShop 中三条线均向上，但 Flash 的起点、终点和增益均最高。这个视图比只比较最终柱高更清楚地区分了基础模型能力与方法增益。

### 6.2 ALFWorld 技能作者×执行器矩阵

![ALFWorld author-executor matrix](figures/alfworld-author-executor-matrix.png)

左图给出四个受控条件的成功率，右图按执行器各自的 No-Skill 基线计算增益。Flash 作者一行在两个执行器上均为 +12.69 pp，证明正向价值并不依赖 GPT-5.4 编写技能；GPT-5.4 作者一行进一步取得更高点估计，说明高质量文本渲染会放大图结构的可执行性。两行的列间模式不同，直接显示作者—执行器交互，而不是简单的模型强弱排序。

### 6.3 ALFWorld 任务族收益谱

![ALFWorld cross-model family spectrum](figures/alfworld-cross-model-family-spectrum.png)

上半图给出 Tower 在六个任务族上的绝对成功率，下半图给出相对各模型自身 No-Skill 的配对增益。收益并非集中在同一任务族：Flash 在 cool 与 heat 上分别提高 52.4 和 52.2 个百分点；GPT-5.4 在 cool 上提高 57.1 个百分点；DeepSeek Pro 在 pick-two 上提高 64.7 个百分点，但在本就达到 100% 的 look-in-light 上下降 5.6 个百分点。这说明总体均值背后存在真实的模型—任务族交互，不宜用单一“强弱模型”顺序概括。

### 6.4 WebShop 配对 reward 瀑布图

![WebShop paired reward waterfalls](figures/webshop-paired-reward-waterfalls.png)

瀑布图将每个任务的 `Trace2Tower - No-Skill` reward 从低到高排列。Flash 的正向样本较少但改善幅度较大，因此取得最高均值增益；GPT-5.4 改善的任务更多，但同时有更多负向翻转；DeepSeek Pro 的正负任务数接近，因而均值增益最小。它说明 WebShop 的总体提升由少数非零变化任务驱动，而不是所有任务获得均匀的小幅改善。

### 6.5 模型成功集合一致性

![Cross-model success-set agreement](figures/cross-model-success-set-agreement.png)

Jaccard 相似度衡量两种模型满分任务集合的交并比。ALFWorld 的 No-Skill 模型间相似度仅为 0.49–0.56，Tower 后提高到 0.82–0.93；其中复用同一 GPT-5.4 Tower 的 Flash 与 Pro 达到 0.93。这表明 Tower 不仅提高平均成功率，还显著提高不同执行模型成功任务集合的一致性。WebShop 的一致性改善较弱且不完全单调，符合该环境中模型适配差异更强的结果。

GPT-5.4 交叉条件同时改变了技能作者，因此涉及 GPT-5.4 的 Tower 一致性只作为互操作现象；Flash 与 DeepSeek Pro 复用相同 Tower，其比较具有更清楚的执行模型控制条件。

## 7. 可用于论文的表述

建议正文使用如下结论：

> Trace2Tower 在三种异构执行模型上均提高了主评估指标。ALFWorld 的 2×2 控制矩阵进一步表明，DeepSeek V4 Flash 编写的 Tower 在 Flash 与 Pro 执行器上均相对各自 No-Skill 提高 12.69 个百分点，证明图技能的正向价值可跨执行器复用；GPT-5.4 编写的 Tower 则分别达到 88.06% 和 85.82%，说明技能文本质量会进一步放大效果。逐任务配对结果不支持简单的模型强弱排序，也不支持把所有差值归因于图算法或执行模型中的单一因素。WebShop 仅报告已有跨执行模型结果，本轮不新增作者矩阵实验。

不建议写成“Flash 比强模型更强”或“Tower 消除了模型差异”。前者忽略了 ALFWorld 的对称翻转和 DeepSeek Pro 的高 No-Skill 基线，后者超出了单轮实验能够支持的证据范围。

## 8. 结果来源

正式结果来自以下运行目录：

- Flash：`alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0`、`alfworld-test-v1-flash-noskill-r0`、`webshop-alfworld-v17-replication-p100-validation-r0`、`webshop-original-concept-v1-validation-flash-noskill-r1`；
- GPT-5.4：`cross-dsflash-render-gpt54-agent-alfworld-tower-r0`、`cross-gpt54-agent-alfworld-noskill-r0`、`cross-dsflash-render-gpt54-agent-webshop-tower-r0`、`cross-gpt54-agent-webshop-noskill-r0`；
- DeepSeek Pro：`generalize-gpt54-render-dspro-agent-alfworld-tower-r0`、`generalize-dspro-agent-alfworld-noskill-r0`、`generalize-gpt54-render-dspro-agent-webshop-tower-r0`、`generalize-dspro-agent-webshop-noskill-r0`。
- ALFWorld 作者矩阵新增格：`alfworld-author-matrix-dsflash-author-dsflash-user-r0`、`alfworld-author-matrix-dsflash-author-dspro-user-r0`。

所有汇总均按 `sample_id` 配对；最终覆盖为 ALFWorld 134/134、WebShop 100/100，各正式结果集无缺失任务。图表由 `scripts/experiments/analyze/plot_cross_model_analysis.py` 从这些运行目录直接生成；配套聚合数据位于 `clean_docs/figures/cross-model-analysis-data.json`。每张图同时提供 PNG 与 PDF。
