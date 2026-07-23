# Trace2Tower 主实验报告

## 摘要

Trace2Tower 定位为一个**动态长程任务治理（Dynamic Long-Horizon Task Governance）**框架。它从成功、部分成功和失败轨迹中抽取领域事件，以**语义关系、时序转移、结果一致性和成败对比**构建 EigenTrace 图，自动发现 Mid 技能与 High 执行路径，并在运行时按任务与阶段选择性暴露经验。

这里的“治理”不是生成更多技能文本，而是统一管理经验的归纳、分层、选择、暴露和反馈修订：High 负责长程任务结构，Mid 负责阶段前置条件与局部执行，动态检索负责让当前任务只接收相关经验，部署优化负责根据反馈调整图结构与路径暴露。

本文回答三个问题：

1. 层次事件图能否治理长程任务中的阶段依赖、前置条件和失败恢复？
2. 动态经验暴露相较静态技能或扁平记忆是否更适配状态变化的交互任务？
3. 同一治理框架能否迁移到任务性质不同的环境？

| 环境                    |          指标 | Trace2Tower |                     最强对照 |       差异 |
|-----------------------|------------:|------------:|-------------------------:|---------:|
| ALFWorld valid_unseen |         成功率 |  **88.06%** | SkillX no-rewrite 81.34% | +6.72 pp |
| ALFWorld 部署任务         |         成功率 |  **84.58%** |             冻结 v0 80.42% | +4.17 pp |
| WebShop P100          | Mean reward | **0.71477** |   Expert-Crafted 0.70085 | +0.01392 |

在 ALFWorld 上，Trace2Tower 相较 No-Skill 提高 35.07 个百分点，并同时降低平均步骤、无效动作和输入 token。相较 Expert-Crafted Skills，成功率提高 11.94 个百分点，95% CI 为 `[+4.48,+19.40]` pp，McNemar exact `p=0.00522`。
作为公开 baseline，ExpeL 在同一 ALFWorld 测试集达到 80.60%，低于 Trace2Tower 7.46 个百分点。

构建消融中，移除时序转移、结果一致性和成败对比分解分别导致 17.91、14.18 和 14.18 个百分点的成功率下降；纯语义聚类无法形成合规的端到端 High。部署消融表明，rewritten High 已释放 Full 相对 No-Skill 总增益的 93.6%，Mid 进一步提供 2.24 个百分点的点估计增益。

在 240 个 ALFWorld 部署任务上，反馈图优化在 TF-IDF 和 embedding 两种相关性表示下均优于冻结图。在 WebShop 上，相同图构建算法和层次运行时合同取得最高 mean reward 和满分率点估计；同一 P100 预算下 ExpeL 的 mean reward 为 0.63348，低于 No-Skill、SkillX 和 Trace2Tower。Trace2Skill 的开源代码复现进一步表明，静态全局技能在 ALFWorld 上仍有正向价值，但在 WebShop 上受到动态状态与任务相关性变化的限制；这一结果用于界定动态治理的适用场景，不用于宣称普遍显著优于 Trace2Skill。

---

## 1. 实验设置

### 1.1 数据与模型

| 环境       | 构建池                                                | 评估集                           | 主指标         | 执行模型                |
|----------|----------------------------------------------------|-------------------------------|-------------|---------------------|
| ALFWorld | P310：310 个训练任务，每题 repeat 0–3，共 1,240 条 No-Skill 轨迹 | AgentBench valid_unseen 134 题 | 成功率         | `deepseek-v4-flash` |
| WebShop  | P100：100 个训练任务，每题 repeat 0–3，共 400 条 No-Skill 轨迹   | 冻结 validation 100 题           | mean reward | `deepseek-v4-flash` |

两套实验均使用 `gpt-5.4` 完成技能渲染和结构化 plan rewrite。

ALFWorld 与 WebShop 使用相同流程：

```text
领域事件抽取
→ quotient node 折叠
→ EigenTrace 图构建
→ 谱分解发现 Mid
→ 成功路径发现 High
→ High rewrite
→ 按 rewritten step 检索 Mid
→ High 与 Mid 联合注入
```

环境适配仅定义领域事件语义，不按任务类别人工分桶，也不将物体类型或商品类别直接作为图社区。

主评估采用固定任务集合、`temperature=0` 和每题一次正式执行。配对 bootstrap 以任务为重采样单位，仅衡量任务组成不确定性。

### 1.2 对照方法

* **No-Skill**：不使用外部经验。
* **Expert-Crafted Skills**：依据环境规则和领域经验人工编写的冻结技能。
* **SkillX**：从成功轨迹中归纳 Plan 与 Function。
* **Trace2Skill**：从成功与失败轨迹提出局部 patch，并层次合并为一个测试时无检索的静态技能；论文预定义的 `+Combined` 与 `+Error` 各构建一次并分别报告，不进行测试集选优。
* **Trace2Tower**：从成功、部分成功和失败轨迹中构建层次事件图。
* **ExpeL**：按公开仓库的规则归纳与 episodic memory 路径完成 P310/P100 全量复现，作为公开 baseline 纳入主表。

Expert-Crafted Skills 作为人工领域工程强基线，用于衡量自动轨迹归纳与专家规则之间的差距。

为避免人工先验、额外构建次数和开发集选择造成不对称优势，所有自动归纳方法均从无人工技能初始化的相同信息条件出发。本文比较的是各方法在统一预算下将轨迹转化为技能的能力，而非通过方法专属调优逼近各自可能达到的经验上限。

完整图表及其叙事分工见 [FIGURES.md](FIGURES.md)。ExpeL 全量复现协议与结果见
`experiments/baselines/expel/RESULTS.json`。
Trace2Skill 的开源复现协议、两个原生变体和跨环境分析见
`experiments/baselines/trace2skill/REPORT.md`。

---

# 2. ALFWorld

## 2.1 主结果

| 方法                    |        成功率 |      平均步骤 |   平均无效动作 | 平均输入 token | 平均 context 字符 |
|-----------------------|-----------:|----------:|---------:|-----------:|--------------:|
| No-Skill              |     52.99% |     14.84 |     0.54 |     45,677 |             0 |
| Expert-Crafted Skills |     76.12% |     11.49 |     0.37 | **38,889** |     **3,286** |
| Trace2Skill +Combined |     58.96% |     14.11 |     0.27 |     68,449 |         9,508 |
| Trace2Skill +Error    |     61.94% |     14.14 |     0.39 |     73,131 |        11,447 |
| SkillX no-rewrite     |     81.34% |     11.69 |     0.49 |     64,651 |        13,092 |
| ExpeL                 |     80.60% |     11.57 |     0.31 |     52,831 |         7,469 |
| **Trace2Tower**       | **88.06%** | **10.06** | **0.22** |     43,758 |         5,422 |

Trace2Skill 按官方静态技能合同复现，+Combined 与 +Error 相较 No-Skill 分别提高 5.97 和 8.96 个百分点。两者为并列的预定义变体，不是测试集候选。Trace2Tower 相较 No-Skill 提高 35.07 个百分点，相较 SkillX 提高 6.72 个百分点，相较 ExpeL 提高 7.46 个百分点。相较 SkillX，其平均输入 token 降低 32.3%，context 字符降低 58.6%；相较 No-Skill，输入 token 仍降低 4.2%。

相较 Expert-Crafted Skills，Trace2Tower 提高 11.94 个百分点，配对 bootstrap 95% CI 为 `[+4.48,+19.40]` pp，McNemar exact `p=0.00522`。自动图归纳因此不仅超过人工统一规则，也同时改善步骤、无效动作和执行成本。

### 2.1.1 任务族泛化

| 方法 | Macro 成功率 | 最弱任务族成功率 | 最强任务族成功率 |
|---|---:|---:|---:|
| No-Skill | 51.51% | 26.09% | 70.97% |
| Expert-Crafted Skills | 74.20% | 52.94% | 90.32% |
| Trace2Skill +Error | 60.76% | 41.18% | 83.33% |
| SkillX | 79.77% | 47.06% | 100.00% |
| ExpeL | 80.67% | 56.52% | 95.24% |
| **Trace2Tower** | **87.22%** | **76.47%** | 96.77% |

Macro 成功率对六个任务族等权，避免样本较多的 `pick_clean_then_place` 主导总体值。Trace2Tower 不仅取得最高 Macro 成功率，最弱任务族仍达到 76.47%，说明总体增益并非来自单一任务族。

![ALFWorld task-family heatmap](figures/alfworld-family-heatmap.png)

---

## 2.2 构建机制消融

13,724 个 segment instance 经 duplicate-embedding collapse 后形成 3,764 个 quotient node。所有消融均从同一预处理输入重新执行图构建、聚类、High 发现、渲染和索引。

![ALFWorld Tower structure](figures/tower-structure.png)

![ALFWorld High/Mid embedding map](figures/tower-embedding-map.png)

图中可以直接看到 Mid 聚类覆盖、High 路径长度以及 High 路径诱导的 Mid 共现关系；向量投影用于展示检索空间的层级分布，不作为额外调参依据。

| 结构指标 | 数值 |
|---|---:|
| Segment instances | 13,724 |
| Quotient nodes | 3,764 |
| Duplicate collapse rate | 72.57% |
| Mid / High | 39 / 118 |
| Mid 规模归一化熵 | 0.9741 |
| Mid 规模变异系数 | 0.435 |
| High 平均路径长度 | 2.924 |
| High 覆盖的 Mid | 28 / 39（71.79%） |
| 测试集实际使用 High | 53 / 118（44.92%） |
| 测试集实际使用 Mid | 30 / 39（76.92%） |

高归一化熵表明证据没有坍缩到少数巨型 Mid；测试时实际暴露了超过四成 High 和超过四分之三 Mid，说明层级目录具有真实运行时利用率，而非仅作为离线可视化结构。

![Tower compression and utilization](figures/tower-compression-utilization.png)

| 配置                | 保留信号                              | Mid / High |        成功率 |   相对 Full |
|-------------------|-----------------------------------|-----------:|-----------:|----------:|
| G0 Full           | 语义 + 时序 + 结果 + signed contrastive |   39 / 118 | **88.06%** |         — |
| G1 Semantic-Only  | 仅语义，固定 K=39                       |     39 / 0 |        不执行 |  无合规 High |
| G2 No Transition  | 语义 + 结果 + signed contrastive      |    19 / 76 |     70.15% | −17.91 pp |
| G3 No Outcome     | 语义 + 时序 + signed contrastive      |   39 / 106 |     73.88% | −14.18 pp |
| G4 No Contrastive | 语义 + 时序 + 结果                      |    10 / 44 |     73.88% | −14.18 pp |

| 配对比较    |   Full 优势 |  Full 胜/负/平 |     95% bootstrap CI | McNemar exact |
|---------|----------:|------------:|---------------------:|--------------:|
| G0 − G2 | +17.91 pp | 28 / 4 / 102 | `[+10.45,+25.37]` pp |    `1.93e-5` |
| G0 − G3 | +14.18 pp | 22 / 3 / 109 | `[+7.46,+20.91]` pp |    `1.57e-4` |
| G0 − G4 | +14.18 pp | 24 / 5 / 105 | `[+6.72,+21.64]` pp |     `5.46e-4` |

Semantic-Only 在相同 quotient node 和固定 K 下无法形成至少包含两个不同 Mid 的合规端到端 High，因此不使用 fallback 生成执行结果。

三个关系信号均具有显著贡献。移除时序转移造成最大下降；移除结果一致性改变了 Mid 成员和 High 路径；取消成败对比分解则使 Mid 从 39 个合并为 10 个。所有 G2-G4 均使用同一正式 rewrite 运行合同。

结果不能由上下文长度解释。G4 的平均 context 为 2,116 字符，低于 Full 的 5,422 字符，但成功率仍低 14.18 个百分点。

---

## 2.3 部署机制消融

部署消融复用同一个 Full Tower 和 rewrite 合同，仅改变是否注入 Mid。

| 配置           | Mid |        成功率 |      平均步骤 |   平均无效动作 | 平均输入 token | 平均 context 字符 |
|--------------|-----|-----------:|----------:|---------:|-----------:|--------------:|
| D0 Full      | 在场  | **88.06%** | **10.06** | **0.22** |     43,758 |         5,422 |
| D1 High-only | 缺席  |     85.82% |     10.50 | **0.22** | **37,410** |     **1,960** |

High-only 相较 No-Skill 提高 32.84 个百分点，释放 Full 总增益的 93.6%。Mid 进一步提高 2.24 个百分点，其 95% CI 为 `[-2.24,+7.46]` pp，McNemar exact `p=0.549`。

主要收益来自 rewritten High；Mid 在其基础上提供局部前置条件和阶段内数据流补充，但当前边际差异未达到显著水平。

---

# 3. ALFWorld 部署优化

## 3.1 数据边界

| 集合                    |  数量 | 用途            |
|-----------------------|----:|---------------|
| `deployment_feedback` | 450 | 生成优化动作并估计反馈效用 |
| Test-1                | 120 | 第一组部署评估       |
| Test-2                | 120 | 第二组部署评估       |

所有动作仅使用 feedback 数据。Test-1 和 Test-2 不参与图动作生成、阈值选择或技能修改。

## 3.2 图优化动作

| 动作         | 决策                              | 作用             |
|------------|---------------------------------|----------------|
| Split      | 拆分 `mid_0028`，增加两个 shadow child | 分离局部行为证据       |
| Merge      | no-op                           | 避免强行合并不相容社区    |
| Promote    | 增加 8 条 High motif               | 提升具有稳定正对比支持的路径 |
| Downweight | 对一个局部有害 High 施加 0.03 penalty    | 降低其在相似失败目标中的暴露 |

Promote 候选依据跨轨迹支持、映射纯度、正负对比得分和 child-Mid 多样性筛选，全程不使用人工任务族标签。

## 3.3 图感知 Pareto 路由

候选沿 Mid→High 和 Split child→parent→High 等关系扩展，并保留三个目标：

* `semantic_relevance`
* `child_relevance`
* `feedback_evidence`

系统先执行非支配排序，再从 Pareto front 中选择 Top-3，避免单一相似度主导检索。

## 3.4 部署结果

| 路由                      |     Test-1 |     Test-2 |         总体 |
|-------------------------|-----------:|-----------:|-----------:|
| 冻结 v0                   |     80.00% |     80.83% |     80.42% |
| 四动作图 + TF-IDF Pareto    | **86.67%** |     82.50% | **84.58%** |
| 四动作图 + Embedding Pareto |     81.67% | **85.83%** |     83.75% |

TF-IDF 路由相较冻结图提高 4.17 个百分点，并将平均步骤从 10.77 降至 10.28。Embedding 路由保持相同 Tower、动作、图关系、Pareto 目标和 Top-3 预算，仅替换相关性表示，仍提高 3.33 个百分点。

两种表示的总体差距为 0.83 个百分点，且分别在两个测试集上取得更高点估计。共同增益来自优化后的图结构、关系扩展和多目标选择，而非特定相似度实现。

---

# 4. WebShop

## 4.1 P100 构建

Trace2Tower 与 SkillX 使用相同的 P100 训练任务预算。400 条 No-Skill 轨迹共产生：

* 2,780 个事件 segment；
* 729 个 quotient node；
* 8 个 Mid；
* 38 条正对比 High path。

WebShop 使用与 ALFWorld 同构的运行时合同：

1. High Top-3 检索；
2. plan rewrite；
3. 按 rewritten step 检索 Mid；
4. 筛选最多 8 个 Mid；
5. 联合注入 rewritten High 与 Mid。

事件定义覆盖查询、结果导航、候选选择、属性检查、选项绑定、价格检查、回退和购买。商品名称与具体事实不直接作为社区标签。

## 4.2 主结果

| 方法                    | Mean reward |     满分率 |     平均步骤 |   平均无效动作 | 平均输入 token |
|-----------------------|------------:|--------:|---------:|---------:|-----------:|
| No-Skill              |     0.65235 |     47% |     7.69 |     0.30 |     19,572 |
| SkillX P100           |     0.68427 |     48% |     7.95 |     0.29 |     34,044 |
| ExpeL P100            |     0.63348 |     42% |     9.86 |     0.31 |     54,092 |
| Trace2Skill +Combined |     0.59685 |     42% |    11.50 |     0.54 |     47,043 |
| Trace2Skill +Error    |     0.62833 |     44% |    11.54 |     0.74 |     42,828 |
| Expert-Crafted Skills |     0.70085 |     50% | **6.22** | **0.06** | **15,526** |
| **Trace2Tower P100**  | **0.71477** | **52%** |     9.90 |     0.16 |     41,977 |

Trace2Tower 相较 No-Skill 提高 `0.06242` mean reward，95% CI 为 `[+0.00500,+0.12450]`；相较 SkillX 提高 `0.03050`，相较 ExpeL 提高 `0.08129`。

相较 Expert-Crafted Skills，Trace2Tower 的 reward 点估计提高 `0.01392`，95% CI 为 `[-0.02809,+0.05758]`；满分率提高 2 个百分点，区间为 `[-3,+7]` pp。当前结果支持最高点估计，但不支持显著超过人工强基线。

Trace2Tower 的平均步骤和输入 token 高于 No-Skill、SkillX 与人工技能，但低于 Trace2Skill +Error 和 ExpeL 的输入 token，且步骤少于两个轨迹归纳 baseline。其 reward 优势仍伴随较充分的候选检查与属性核验，但并非来自注入最长的上下文。

Trace2Skill 复现遵循其无测试时检索的静态全文注入合同。+Combined 与 +Error 各自只构建一次并并列报告；详细协议、完整性审计和适用边界见 `experiments/baselines/trace2skill/REPORT.md` 与 `IMPLEMENTATION_AUDIT.md`。当前点估计说明全局静态 SoP 与运行时状态主导的购物任务存在适配边界，不用于声称 Trace2Tower 在一般意义上显著强于 Trace2Skill。

## 4.3 Expert-Crafted Skills

Expert-Crafted Skills 是一份冻结的人类购物策略，覆盖搜索压缩、候选筛选、硬约束核验、选项确认、回退和购买门，不包含具体商品事实。

其相较 No-Skill 提高 `0.04850` reward，并取得最低的平均步骤、无效动作和输入 token，是主表中执行效率最高的方法。

Trace2Tower 在任务质量上达到人工强基线的同等量级并取得更高点估计，但当前在线成本明显更高。

## 4.4 跨环境差异

ALFWorld 的目标在任务开始时已暴露主要程序类型，动作前置条件和阶段顺序相对稳定。WebShop 的商品事实则在搜索结果、详情页和选项页逐步显露，是否回退、重搜或继续核验依赖当前页面状态。

因此，相同图算法和层次运行时合同在两个环境中均产生正向结果，但收益形态不同：

* ALFWorld 同时改善成功率、步骤、无效动作和 token；
* WebShop 获得更高 reward 点估计，但增加执行步骤和输入 token。

同一冻结 P100 Tower 在另一个 WebShop 分区上未获得同样稳定的增益。因此，WebShop 结论限定为：统一 Trace2Tower 算法具有明确跨域潜力，但跨分区稳定性弱于 ALFWorld。

### 4.5 为什么 WebShop 上难以形成显著优势

SkillX 与 Trace2Tower 在 WebShop 上均获得正向点估计，但未形成 ALFWorld 上同等稳定的显著优势。主要原因不是经验归纳无效，而是两类环境中**可复用经验的结构不同**。ALFWorld 的任务类型、动作前置条件和长程阶段在初始目标中基本可观测，不同轨迹能够对齐为稳定程序；WebShop 的决定性信息则分散在搜索结果、商品详情、选项和价格状态中，正确分支往往只有在交互后才能确定。相同初始需求可能对应完全不同的候选质量、属性缺失和恢复路径，从而削弱基于任务文本进行跨任务检索与提前注入的稳定性。

这一差异同时限制了两种方法。SkillX 主要从成功轨迹中抽取并检索完整 Plan，但 WebShop 中表面相似的购物需求未必共享相同的页面证据和执行分支；Trace2Tower 能进一步利用时序、结果和失败对比恢复检查与回退结构，但这些结构是否适用同样依赖当前页面状态。因此，两种方法都可能在困难任务上改善核验和恢复，同时在简单任务上引入额外步骤，最终表现为平均 reward 提升但任务间方差较大。

这一现象与 ExpeL 的跨环境观察一致。ExpeL 将 ALFWorld 描述为依赖特定动作集合、较适合从历史轨迹中复用执行经验的环境；WebShop 则同时要求查询改写、价格比较等网站推理，以及点击、搜索和选项选择等精确动作执行。其原始实验中，ExpeL 相对 ReAct 在 ALFWorld 上由 40% 提升至 59%，而 WebShop mean reward 仅由 0.665 提升至 0.701，并明确指出 WebShop 仍存在较大改进空间。([AAAI Publications][1]) 因此，WebShop 上较小且不稳定的优势更可能反映**交互后信息主导的任务性质**，而非某一种经验归纳方法的孤立缺陷。

[1]: https://ojs.aaai.org/index.php/AAAI/article/download/29936/31635 "ExpeL: LLM Agents Are Experiential Learners"

---

# 5. 结论

Trace2Tower 在 ALFWorld valid_unseen 上达到 88.06% 成功率，相较 No-Skill、SkillX、ExpeL 和 Expert-Crafted Skills 分别提高 35.07、6.72、7.46 和 11.94 个百分点，并同时降低步骤、无效动作和相对 SkillX 的输入成本。

构建消融表明，时序转移、结果一致性和 signed 成败对比分解均显著影响最终性能；纯语义聚类无法恢复合规的端到端 High。部署消融表明，主要收益来自 rewritten High，Mid 提供额外局部精化。

在 240 个 ALFWorld 部署任务上，反馈图优化在 TF-IDF 和 embedding 两种相关性表示下均优于冻结图，说明部署增益不依赖单一检索实现。

在 WebShop 上，相同图构建算法和层次运行时合同获得最高 mean reward 和满分率点估计，并显著优于 No-Skill 的 reward；其更高执行成本和较弱跨分区稳定性构成当前适用边界。

综上，Trace2Tower 的核心贡献不是另一种静态 skill writer，而是面向动态长程智能体的任务治理机制：

> 通过语义、时序、结果一致性和成败对比构建事件图，将异质轨迹组织为层次化执行知识；再依据当前任务、执行阶段和反馈，动态治理经验的选择、暴露与修订。

该定位限定于动态、长程、交互式任务：任务需要多阶段依赖与失败恢复，决定性状态在执行过程中逐步显露，且一次性注入全部经验会产生明显无关上下文。对于接口稳定、程序高度重复、静态 SoP 已足够覆盖的任务，Trace2Skill 式全局技能仍可能是更简单有效的选择。
