# Trace2Tower WebShop 完整实验报告

## 摘要

本报告封存 Trace2Tower 在 WebShop 上的全部正式证据。配置选择只使用 `webshop:50-149` 验证任务；最终性能使用冻结的 Random-300 测试选择；Mid-only 与 High 交叉实验只解释机制，不反向修改配置。封版证据包含 29 个完整条件、19,500 个 episode，所有条件均无缺失 key、重复 key或未解决 error。

最稳定的结果有四点：

1. Success-only Tower 在 Flash 和 Pro 上都取得全部 baseline 中最高的 mean reward，分别为 `0.7186` 和 `0.7029`。
2. 在 Pro 上，Success-only Tower 相对 NoSkill 的 reward 提高 `0.0486`、满分率提高 `5.1` 个百分点，两项 95% CI 均不跨零。
3. Tower 相对 Flat 的 reward 差尚未显著，但两个模型上都提高 completion、减少步骤，并降低 `12.4%-18.4%` 的输入 token。
4. Mixed 的主要弱点位于 Mid，而不是 High。High 能补偿较弱的 mixed Mid；将 success-only High 交叉接到 mixed Mid 后，Pro reward 显著提高 `0.0363`。

因此，WebShop 证据支持的核心价值不是“Tower 在每个指标上大幅击败所有方法”，而是：在保持当前最高平均奖励的同时，分层检索比 Flat 更高效，比 SkillX 更能保留满分所需的细粒度约束，并且 High 层对 Mid 噪声具有可迁移的补偿作用。

## 研究问题

实验集中回答五个问题：

1. Skill 是否优于 NoSkill，还是只会增加上下文噪声？
2. Tower 是否比 Flat skill library 更有效或更高效？
3. 只使用成功轨迹与混入错误/部分成功轨迹有什么差别？
4. Mid 与 High 分别贡献了什么，mixed 失效发生在哪一层？
5. 技能收益是否依赖执行模型？

## 数据、模型与统计口径

### 训练证据

NoSkill 训练池由 Flash 在 train split 上的 50 个任务、4 次重复构成，共 200 条轨迹。正式 artifact 使用两种证据策略：

| Evidence | 轨迹数 | Segment 数 | Mid cluster | High path | 图中负邻接项 |
|---|---:|---:|---:|---:|---:|
| Success-only | 94 条满分轨迹 | 466 | 16 | 26 | 0 |
| Mixed | 173 条有信息轨迹 | 947 | 19 | 17 | 2,114 |

Success-only 是标准控制；mixed 保留满分成功轨迹，同时加入有信息价值的错误和部分成功轨迹。SkillX 严格只使用同一批 94 条满分轨迹，与 mixed 消融明确分离。

所有 Flat、Mid、High 和 SkillX 抽取/渲染均使用 GPT-5.4；agent rollout 使用 `deepseek-v4-flash` 或 `deepseek-v4-pro`。凭据只从忽略的 `.env` 读取。

### 验证集

配置选择使用 `webshop:50-149` 的 100 个任务，每任务 3 次重复，每条件 300 episode，agent 为 Flash。11 个验证条件包括 NoSkill、Flat cap3/cap8，以及 success-only/mixed Tower 的 cap3/5/8/12，共 3,300 episode。

选择规则预先固定为：满分成功率优先，其次 mean reward，最后比较 token、步骤和无效动作成本。该规则选择 Flat cap3、Tower cap3、High top-k 1。验证过程与完整性审计见 [WebShop 验证与正式测试口径](webshop-validation-and-final-test-protocol.md)。

### 正式测试与机制消融

最终测试使用 selection `selection_32248afcaee8da76`：从此前未出现在 result/error 中的 WebShop held-out 候选中，以三个固定随机种子各抽 100 个任务，共 300 个互不重复任务。每任务 3 次重复，每条件 900 episode。

NoSkill、Flat、Success Tower 和 Mixed Tower 的配置与 artifact 在首次 Random-300 rollout 前冻结。SkillX 是 selection 冻结后追加的外部 baseline；它直接使用官方抽取/检索参数，没有根据 Random-300 结果调参，因此可以比较性能，但不应被描述为原始预注册方法矩阵的一部分。

| 证据块 | 条件数 | Episode 数 | 用途 |
|---|---:|---:|---|
| 验证矩阵 | 11 | 3,300 | 只选择配置 |
| 正式 baseline 矩阵 | 10 | 9,000 | 最终泛化表现 |
| Mid-only | 4 | 3,600 | 定位 High 增量 |
| High 交叉 | 4 | 3,600 | 检验 High 补偿能否跨库迁移 |
| 合计 | 29 | 19,500 | WebShop 封版证据 |

满分定义为 `primary_score >= 0.999`。置信区间先在任务内平均 3 次重复，再对独立任务做 10,000 次 task-cluster bootstrap。Completion 表示正常完成流程，不等同于满分成功。输入 token 是一个 episode 所有 agent 请求的累计输入量，不包括技能构建成本。

### 开发历史与封版证据

在上述证据形成之前，项目还运行过 1/2/4/20/25/50-task smoke、旧 holdout、legacy retrieval、self-filter、event-stratified、compact High 和局部 cap8 等开发实验。这些结果用于发现依赖、修复检索、扩大轨迹池和形成验证假设，原始 artifacts 均保留；但其样本小、用途不同，部分样本后来承担过配置探索，因此不与 Random-300 合并，也不进入最终显著性结论。

冻结清单锁定的是最终可解释证据链，而不是选择性删除开发失败：验证矩阵说明配置如何产生，Random-300 回答泛化表现，Mid-only/High 交叉回答机制。早期结果只作为实验演进记录，可在 `decisions.md` 和 `flash-pool-expansion.md` 中追溯。

## 配置选择结果

### Flat cap

| Cap | Mean reward | 满分率 | Completion | 步数 | 输入 token |
|---:|---:|---:|---:|---:|---:|
| 3 | **0.7276** | **47.3%** | **94.7%** | **8.18** | **25,082** |
| 8 | 0.6871 | 44.3% | 91.0% | 8.42 | 35,119 |

cap3 相对 cap8 的 reward 差为 `+0.0405`，CI `[-0.0014,+0.0843]`；满分率差为 `+3.0%`，CI `[-1.0%,+7.0%]`。显著性不足以宣称“更多卡必然有害”，但 cap3 在两个选择指标和全部成本指标上都占优，因此冻结 cap3。

### Tower cap 与 evidence

| Evidence | Cap3 reward / 满分率 | Cap5 | Cap8 | Cap12 |
|---|---:|---:|---:|---:|
| Success-only | `0.6966 / 48.7%` | `0.7043 / 45.0%` | `0.6760 / 44.3%` | `0.6756 / 44.0%` |
| Mixed | `0.6748 / 44.0%` | `0.6798 / 45.0%` | `0.6862 / 45.7%` | `0.6748 / 43.7%` |

Success-only cap3 具有该控制下最高满分率。它相对 cap8 的满分率高 `4.3%`，CI `[+0.7%,+9.0%]`，且每 episode 少 7,053 输入 token；因此正式 Tower 冻结 cap3。

mixed 的局部最优出现在 cap8，但这不能改变全局 cap3 选择。它揭示的是一个机制：mixed 库更大、更多样，也含负关系；cap3 的有限槽位更容易被较弱或冲突的 Mid 占据，cap8 增加覆盖后可找回部分有效信息，而 cap12 又因稀释和上下文负担下降。该解释与 cap 曲线和后续 Mid-only 消融一致，但“槽位挤占”本身没有被逐卡干预测量，属于证据支持的机制推断。

## 正式 baseline 结果

### 任务表现

| 模型 | 方法 | Mean reward | 满分率 | 相对 NoSkill reward 差（95% CI） | 相对 NoSkill 满分率差（95% CI） |
|---|---|---:|---:|---:|---:|
| Flash | NoSkill | 0.7037 | 49.7% | - | - |
| Flash | Flat cap3 | 0.7050 | 51.2% | +0.0013 `[-0.0242,+0.0276]` | +1.6% `[-1.7%,+4.9%]` |
| Flash | SkillX | 0.6981 | 46.8% | -0.0057 `[-0.0306,+0.0194]` | -2.9% `[-6.4%,+0.6%]` |
| Flash | **Success-only Tower** | **0.7186** | 51.3% | +0.0148 `[-0.0106,+0.0410]` | +1.7% `[-1.9%,+5.3%]` |
| Flash | Mixed Tower | 0.7153 | **51.7%** | +0.0116 `[-0.0137,+0.0379]` | +2.0% `[-1.7%,+5.8%]` |
| Pro | NoSkill | 0.6543 | 46.8% | - | - |
| Pro | Flat cap3 | 0.6930 | **52.9%** | +0.0387 `[+0.0069,+0.0715]` | +6.1% `[+2.6%,+9.8%]` |
| Pro | SkillX | 0.6893 | 47.6% | +0.0351 `[+0.0058,+0.0654]` | +0.8% `[-2.7%,+4.2%]` |
| Pro | **Success-only Tower** | **0.7029** | 51.9% | **+0.0486** `[+0.0176,+0.0801]` | **+5.1%** `[+1.3%,+8.9%]` |
| Pro | Mixed Tower | 0.6845 | 51.2% | +0.0303 `[-0.0021,+0.0634]` | +4.4% `[+0.7%,+8.2%]` |

Success-only Tower 在两个模型上都取得最高 mean reward。Flash 的差异仍是点估计；Pro 上相对 NoSkill 的 reward 与满分率提升都有 95% CI 支持。

### 完成度与执行成本

| 模型 | 方法 | Completion | 步数 | 输入 token | 无效动作率 |
|---|---|---:|---:|---:|---:|
| Flash | NoSkill | 90.1% | 7.45 | 18,886 | 4.46% |
| Flash | Flat | 88.9% | 8.54 | 27,892 | 3.32% |
| Flash | SkillX | 92.7% | **6.77** | 23,532 | 4.56% |
| Flash | Success Tower | 92.2% | 7.11 | 22,764 | 3.37% |
| Flash | Mixed Tower | **92.9%** | 7.19 | **22,140** | **2.81%** |
| Pro | NoSkill | 81.4% | 8.69 | 23,664 | 9.58% |
| Pro | Flat | 84.7% | 9.12 | 30,500 | 8.16% |
| Pro | SkillX | **88.0%** | **7.56** | 27,766 | **6.15%** |
| Pro | Success Tower | 87.1% | 8.12 | 26,709 | 7.55% |
| Pro | Mixed Tower | 85.8% | 8.22 | **26,518** | 6.99% |

Tower 相对 Flat 的优势主要体现在执行形态：Success Tower 在 Flash/Pro 上分别少 `1.42/1.00` 步、少 `18.4%/12.4%` 输入 token，同时 completion 高 `3.3/2.4` 个百分点。reward 差为正但 CI 跨零，因此效率优势应与性能显著性分开表述。

### 跨 seed 稳健性

Random-300 由三个互不重叠的 100-task seed 组成。Success Tower 相对 NoSkill 的 reward 在 `2 模型 × 3 seed` 六个单元格中全部为正；相对 Flat 在六个单元格中五个为正。相对 Flat 的平均步数和输入 token 则六个单元格全部更低。

这不是额外显著性检验，正式推断仍以合并后的 300-task cluster bootstrap 为准；它说明总体方向并非由某一个 seed 单独驱动。唯一的 Success Tower reward 低于 Flat 单元格是 Pro seed `20260714`，差约 `-0.0059`，幅度很小，也与总体“Tower 尚未显著击败 Flat”的谨慎结论一致。

## 与 SkillX 的比较

正式 SkillX 使用 94 条成功轨迹，经官方 hybrid pipeline 从 109 个候选中保留 2 个 atomic skill 和 26 个 planning skill。其最终上下文平均为 5,369 字符，高于 Success Tower 的 3,830 字符和 Mixed Tower 的 3,376 字符。

SkillX 的鲜明特点是“更快结束，但满分约束更弱”：

- 相对 Flat，它在 Flash/Pro 上平均少 `1.77/1.57` 步，completion 高 `3.8/3.3` 个百分点；
- 但满分率分别低 `4.4/5.3` 个百分点，两个 CI 都完全低于零；
- Success Tower 相对 SkillX 的满分率高 `4.6/4.3` 个百分点，两个 CI 都完全高于零；
- Success Tower 虽多约 `0.35/0.57` 步，累计输入 token 反而少 `768/1,057`，说明较短且相关的分层上下文抵消了额外步骤。

最可能的成因是 SkillX 最终只保留了通用 `search_action` 和 `click_action` atomic guidance。它能规范搜索、点击和购买流程，所以 completion 高、步骤少；但 WebShop 满分依赖品牌、属性、尺寸、颜色、价格等多约束同时满足，过于通用的流程技能容易推动模型过早完成购买。Tower 的 Mid/High 卡保留了更多状态与路径条件，因此完成稍慢，却更能维持满分约束。

## Mixed、Mid 与 High 的机制

### Mixed 的问题位于 Mid

Mid-only 下，Mixed 相对 Success-only 的 reward 差在 Flash/Pro 上分别为 `-0.0253`、`-0.0315`，95% CI 都完全低于零。恢复 High 后，完整 Tower 的差异缩小到 `-0.0033`、`-0.0183`，两个 CI 都跨零。

这排除了“High 导致 mixed 失效”的解释。mixed 的负关系和部分成功片段扩大了图覆盖，但其 Mid 摘要更容易混合正确动作、恢复动作和失败边界；当这些信息直接作为当前状态建议注入时，会降低决策纯度。

### High 是补偿层

完整 Mixed Tower 相对 Mixed Mid-only：

| 模型 | Reward 差 | Completion 差 | 步数差 | 输入 token 差 |
|---|---:|---:|---:|---:|
| Flash | +0.0216 | +4.3% | -0.89 | -2,126 |
| Pro | +0.0301 | +5.2% | -1.11 | -2,752 |

四项指标在两个模型上的方向一致。单项 reward CI 略跨零，但 High 交叉实验提供了更强证据：把 success-only High 接到 mixed Mid，Flash reward 提高 `0.0098`，Pro 提高 `0.0363`，后者 CI `[+0.0064,+0.0668]`；同时两模型都减少步骤和 token。

因此 High 的优势不只是“多一段提示”。它将局部 Mid 行为放回任务路径中，能抑制较弱 Mid 的短视或冲突建议，而且这种补偿可以跨 evidence 库迁移。反向把 mixed High 接到 success-only Mid 时 reward 略降但 CI 跨零，说明 High 来源可能影响质量，但当前证据不足以声称必须同源。

## 模型依赖与行为约束

NoSkill 下 Pro 的 reward `0.6543` 低于 Flash 的 `0.7037`，所以不能把模型标签简单解释成 WebShop 上的固定强弱排序。更有解释力的是行为数据：Pro NoSkill 的无效动作率为 `9.58%`，约为 Flash 的两倍；Success Tower 将其降到 `7.55%`，SkillX 降到 `6.15%`。

技能对 Pro 的 reward 增益也更大。SkillX 相对 NoSkill 在 Pro 上为 `+0.0351`，Flash 上为 `-0.0057`；模型间差中差为 `+0.0407`，CI `[+0.0030,+0.0778]`。这说明结构化技能可能主要约束了 Pro 更易出现的工具调用偏离、重复探索或过度推理，而 Flash 本身路径较短，额外技能的边际价值较小，甚至可能引入干扰。

该结论限定于当前 WebShop agent endpoint 与两个模型，不应外推为“越强的模型越需要技能”的普遍定律。

## Trace2Tower 的潜在优势

### 1. 奖励与完成质量兼顾

Tower 没有追求最短路径而牺牲任务约束。它在两个模型上取得最高 mean reward；相对 SkillX，满分率稳定更高。这说明层级技能的价值可能更多体现在“保持目标条件”而不是单纯“推动流程结束”。

### 2. 上下文效率

相对 Flat，Tower 用更少上下文、步骤和 token 达到更高的 reward 点估计。检索不是把更多经验塞进 prompt，而是在当前状态选择少量 Mid，并用一个 High 路径约束其使用方式。这个优势对长轨迹或昂贵模型可能比 WebShop 上的绝对 reward 差更重要。

### 3. 对噪声的层级容错

Mixed Mid 明显较弱，但完整 Mixed Tower 接近 Success-only，且交叉 High 能补偿 mixed Mid。High 因而表现为一个可迁移的纠偏层。错误轨迹没有在本轮直接带来更高最终分数，但它构成了有价值的消融：错误信息的价值取决于是否被正确组织，而不是只取决于是否被收集。

### 4. 给易偏离模型提供行为先验

Pro 的高无效动作率和更大的技能收益表明，Tower 可能在充当行为正则项：减少不符合工具协议或当前页面状态的动作。这是一个适合在其他交互 benchmark 上检验的优势。

### 5. 完成度较高且成本可控

Success/Mixed Tower 的 completion 在 Flash 上为 `92.2%/92.9%`，在 Pro 上为 `87.1%/85.8%`，均高于对应 NoSkill。与 Flat 相比，Tower 并非通过更长轨迹换取分数；它反而更短、更省 token。

## 证据边界与不足

1. Tower 相对 Flat、SkillX 的 reward 直接差异仍未显著，不能写成全面统计胜出。
2. 效率指标目前是描述性比较，未单独给出 task-cluster CI。
3. Mid-only 与 High 交叉复用了 Random-300 测试任务，只能解释冻结方法，不能再用于选择新配置。
4. mixed 的“槽位挤占/上下文稀释”解释符合 cap 曲线，但没有逐条替换检索卡的完整因果干预。
5. 训练池只覆盖 26 个成功任务，虽然有 94 条满分轨迹；更大且更多样的成功池可能改变 SkillX、Flat 与 Tower 的相对结果。
6. 结论只覆盖 WebShop 和两个 DeepSeek agent，不代表 ALFWorld 或其他模型家族。
7. 在线输入 token 不含离线技能构建成本。Tower、Flat 与 SkillX 的 renderer/embedding 调用结构不同，因此本文的效率优势只针对 rollout 阶段，不能直接扩展为端到端货币成本结论。

## 冻结结论

WebShop 实验线至此冻结。正式配置为 Success-only Tower、diverse direct-Mid retrieval、cap3、High top-k 1；Mixed 保留为错误轨迹消融，SkillX 保留为成功轨迹官方 baseline。

冻结后允许的操作只有：从现有原始结果复算统计、审计哈希、修正文档错误，以及不改变结论的可复现性维护。不得再根据 WebShop 验证或 Random-300 结果选择新的 cap、prompt、evidence、检索策略或 artifact；任何新方法开发应转移到其他 benchmark 或建立全新、预注册且与本封版证据隔离的数据协议。

机器可读冻结清单见 `webshop-freeze-manifest.json`，可在仓库根目录运行 `.venv\Scripts\python.exe scripts/experiments/freeze_webshop.py --verify` 复核。正式数字的详细配对统计见 [WebShop Random-300 正式测试报告](webshop-final-random300-report.md)，机制细节见 [WebShop Mid-only 机制消融](webshop-mid-only-ablation.md)，SkillX 复现边界见 [SkillX Baseline](skillx-baseline.md)。
