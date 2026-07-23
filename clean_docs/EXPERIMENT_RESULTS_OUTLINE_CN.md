# Trace2Tower 实验与结果章节写作大纲

本文档参考 SkillX 论文实验部分的组织方式，为 Trace2Tower 的实验与结果章节提供中文写作大纲。整体采用“集中声明实验协议、以结论组织结果、将实现细节下沉附录”的结构。正文措辞与最终数字仍可在论文定稿阶段调整。

参考材料：`MinerU_markdown_09_skillx_2604.04804_2076179248605401088.md`

## 需要明确声明的实验事项

- 数据集、固定划分、训练轨迹数量及评价指标。
- 技能构建模型、技能使用模型和计划 rewrite 模型分别是谁。
- No Skill、SkillX、ExpeL、Trace2Skill 和 Expert-Crafted Skills 的具体定义。
- 自动方法采用相同信息条件和统一构建预算，只构建一次且不在测试集上选优。
- 各方法遵循其开源实现的原生推理协议，不声称所有方法使用完全相同的检索方式。
- 主结果采用哪一次预先固定实验，重复实验如何单独报告。
- 构建和推理的关键参数、检索数量、注入内容及最大交互步数。
- 效率指标、构建成本统计口径，以及 embedding 成本不纳入 GPT token 成本。
- 消融实验、跨模型实验、案例分析和方法局限性。

# 5 实验

## 5.1 实验设置

### Benchmarks and Metrics

介绍 ALFWorld 和 WebShop 的任务性质、训练轨迹池、固定测试划分与指标。

ALFWorld 以任务成功率为主指标，同时报告平均交互步数、无效动作和输入 token。WebShop 同时报告平均奖励和完美完成率。部署反馈实验中的两个冻结评测集统一称为 Test-1 和 Test-2，具体来源放入附录。

### Models and Roles

明确区分三个模型角色：

- Skill Author：将训练轨迹转化为技能的模型。
- Skill User：在测试环境中使用技能执行任务的模型。
- Plan Rewriter：根据检索技能生成当前任务计划的模型。

正文主要报告 GPT-5.4 与 DeepSeek V4 Flash 构建、DeepSeek V4 Flash/Pro 执行的设置，并通过受控矩阵分析 author-user 交互。

### Baselines

主表包含以下方法：

- No Skill
- Expert-Crafted Skills
- SkillX
- ExpeL
- Trace2Skill +Combined
- Trace2Skill +Error
- Trace2Tower

正文用一至两句话说明各方法的经验表示形式；详细提示词、开源版本和执行协议放入附录。

### Fair Comparison Protocol

建议在正文中明确声明：

> 为避免人工先验、额外构建次数和开发集选择造成不对称优势，所有自动归纳方法均从无人工技能初始化的相同信息条件出发。本文比较的是各方法在统一预算下将轨迹转化为技能的能力，而非通过方法专属调优逼近各自可能达到的经验上限。

在此基础上补充：所有自动方法均使用相同训练信息与一次性构建预算，测试阶段不进行方法专属选优。基线遵循其公开实现的原生技能使用方式，因此比较的是统一信息条件下不同轨迹归纳机制的有效性，而非人为统一其内部推理流程。

### Implementation Details

声明 Trace2Tower 的事件轨迹抽取、重复 embedding 折叠、图构建与 Mid/High 归纳，以及测试时 Top-3 High 与计划 rewrite 的 High-only 主合同。Mid 是图中的局部证据层，但不在当前主运行中注入。

正文只保留决定实验语义的参数，其余实现细节放入附录。

## 5.2 主要结果

### Trace2Tower Improves Long-Horizon Task Completion

表 1 报告 ALFWorld 主实验。

建议论述顺序：首先报告 Trace2Tower High-only 的 85.82% 成功率，随后依次比较 No Skill、Expert-Crafted Skills、SkillX、ExpeL 和 Trace2Skill。说明自动归纳方法显著改善无技能执行，并在当前单次完整运行中超过人工先验技能。

### Hierarchical Experience Organization Outperforms Flat Representations

重点比较 Trace2Tower 与 SkillX、ExpeL、Trace2Skill。

核心表述方向：实验结果支持“先在事件图中组织跨轨迹证据，再归纳可执行技能”的设计。这一结果说明经验组织方式具有决定性作用，但不应被扩大解释为所有层次化文本格式必然优于所有扁平表示。

### Improvements Are Not Explained by Additional Construction Cost

将 Trace2Tower 与 SkillX 的 GPT 构建 token 放在同一张成本表中。

Trace2Tower 的总 chat token 比 SkillX 低约 16.85%，输出 token 低约 58.09%。ExpeL 的历史运行未保存完整 token，因此只报告可验证的调用次数，不将缺失值记为零。

### Runtime Context and Token Efficiency

单独报告执行时效率，不与构建阶段 GPT 成本混写。明确区分：平均 agent 输入 token 衡量整个 episode 的实际输入量；注入 context 字符衡量一次性经验文本长度。两项均不含 embedding，也不把 renderer 的测试时 plan rewrite token 记入 agent 输入。

重点对照 SkillX、ExpeL、Trace2Skill 和 Expert-Crafted Skills：Trace2Tower High-only 相对 SkillX 的平均输入 token 降低 42.1%，经验 context 字符降低 85.0%；相对 ExpeL 分别降低 29.2% 和 73.8%。同时说明相对人工技能的 token 差距较小，但人工技能仍拥有步骤效率优势。用性能-token-步骤散点图展示该关系，避免把短 context 误写成唯一目标。

### Cross-Domain Benefits Depend on Task Structure

表 2 报告 WebShop 验证集结果。

正文说明 Trace2Tower 在主验证划分上取得最高的自动方法结果，但跨划分收益不如 ALFWorld 稳定。将其解释为任务结构差异：ALFWorld 的长程状态转换与先决条件更适合层次事件图；WebShop 的搜索、筛选和局部商品判断使全局技能的边际作用更受实例分布影响。

该部分应作为适用边界分析，而不是为退化结果进行事后辩护。

## 5.3 结构与行为分析

### Graph Compression and Skill Utilization

展示 13,724 个 segment instance 折叠为 3,764 个图节点，以及 39 个 Mid 和 118 个 High 的层次结构。

报告重复折叠率、High 对 Mid 的覆盖、测试期 High 使用覆盖率和 Mid 的分布熵，以证明图不是仅用于生成最终文本的临时中间产物。

### Generalization Across Task Families

按 ALFWorld 任务族报告成功率、宏平均和最弱任务族表现，避免总体成功率掩盖任务差异。

### Paired Behavioral Analysis

分析 Trace2Tower 与 No Skill 的逐任务翻转、失败类别、交互步数和无效动作。重点呈现搜索耗尽、多目标未完成和状态转换偏离，不强行为所有失败建立单一归因。

# 6 进一步分析

## 6.1 图构建组件 case study

报告 Full、无 transition、无 outcome、无 contrastive，以及 semantic-only 无法形成合规 High 的结果。

semantic-only 应标为“构建失败”或“无可执行结果”，不能记成 0% 成功率。该组结果以单次 Full Mid 运行作为锚点，只用于展示结构变化，不进入主效果或显著性结论。

## 6.2 Full Mid Case Study

保留 88.06% Full Mid 运行，记录其与 High-only 的 7 胜 / 4 负任务翻转及 Mid 选择分布。后续重复显示 Mid 自筛选发生漂移，因此不得将该结果描述为可复现的部署增益。

## 6.3 Construction Cost

比较 Trace2Tower、SkillX 和其他能够获得可靠统计的自动方法所使用的 GPT 构建调用与 token。

embedding 只承担图构建和检索中的向量计算，不计入 GPT token 成本。正文应同时展示质量与构建成本，避免仅报告 Trace2Tower 自身消耗。

## 6.4 Case Study

选择两至三个完整案例：

- 图技能帮助补全先决条件或动作顺序。
- Full Mid 单次运行中，Mid 在 High 计划基础上修正局部执行。
- 检索技能完整，但智能体仍因搜索或步数限制失败。

案例应展示检索内容、计划变化和最终动作，避免只展示经过筛选的成功文本。

# 7 局限性

明确说明以下边界：

- ALFWorld 上的收益稳定且显著，但 WebShop 上的收益更依赖具体数据划分。
- Mid 的边际增益尚未达到统计显著。
- 技能文本质量受 skill author 能力影响，图结构不能完全消除语言模型生成质量差异。
- 当前实验主要验证固定环境内的动态长程任务治理，尚未证明跨环境或跨工具模式直接迁移。
- 部署反馈优化只在 ALFWorld 上验证。

# 附录建议

## A. 数据集与评价协议

列出训练池、固定 manifest、测试集编号、任务族和指标计算方式。

## B. Baseline 实现

记录各 baseline 的开源版本、技能构建输入、实际推理协议和公平性约束。

## C. Trace2Tower 实现细节

记录图构建、节点折叠、技能归纳、检索、rewrite 和注入参数。

## D. 完整实验结果

放置完整主结果、重复实验、任务族结果、配对翻转和置信区间。

## E. 构建成本

列出 GPT 调用次数、输入 token、输出 token、总 token 及统计缺失项。

## F. 结构 Case Study

放置完整构建结构变体、Full Mid 单次运行及其不稳定性说明。

## G. 案例

放置完整技能文本、检索结果、计划 rewrite、动作轨迹和失败案例。

# 整体叙事

Trace2Tower 面向动态长程任务治理：它通过事件图保存跨轨迹结构，并通过可检索的 High 技能支持任务执行。其优势不应表述为“在所有数据集和模型上全面领先”，而应表述为“在具有显式状态转换和长程依赖的任务上表现突出；当前 High-only 主运行、结构分析与跨域边界共同构成证据链”。
