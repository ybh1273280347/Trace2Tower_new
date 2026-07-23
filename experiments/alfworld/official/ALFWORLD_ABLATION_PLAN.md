# ALFWorld Trace2Tower 消融实验结论

## 实验口径

- 训练池：P310 train 任务，repeat 0/1/2/3，共 1,240 条 No-Skill 轨迹。
- 执行集：AgentBench `valid_unseen` 134 题，repeat 0；执行模型为
  `deepseek-v4-flash`，temperature 0，最多 20 步；renderer/rewrite 使用 `gpt-5.4`。
- Full 参照：118/134（88.06%），对应
  `original-concept-v17/p310/tower.json`。
- 构建机制 G0-G4 改变图结构，需要从同一份预处理输入重新执行图构建、High 路径、
  renderer、索引和 snapshot 流程。
- G0-G4 统一使用 `collapse_duplicate_embeddings=true`：13,724 个 segment instance
  先按 `(event_type, embedding)` 折叠为 3,764 个 quotient node，再执行图构建或聚类。
- 部署机制的正式可比实验为 D0 与 D1：二者复用同一个 Full Tower 和
  `plan_rewrite/budgeted_v2` 合同，只改变是否注入 Mid。

## 核心结论

构建层结论保持不变：移除时序转移、结果一致性或 signed 成败对比分解，分别造成
36.57、31.34 和 27.61 个百分点的下降。这三类关系信号各自提供独立有效的信息，
效果不能用语义聚类或上下文长度解释。

部署层的旧结论需要修正。旧 D1 虽声明开启 rewrite，实际走的是 SkillX-native
`high_to_mid` provider，而不是 Full 使用的正式 `plan_rewrite/budgeted_v2` provider，
因此旧 D1 的 78/134（58.21%）作废。修正后 D1 达到 115/134（85.82%）：正式
rewrite 在没有 Mid 时已经保留了大部分效果；Mid 在 rewrite 在场时带来 +2.24pp，
但当前配对差异不显著。以 No-Skill 的 52.99% 为 Baseline，D1 已取得 +32.84pp，
释放了 Full 相对 Baseline 总增益的 93.6%；Mid 将成功率进一步推高到 88.06%。因此，
现有证据支持“整体架构贡献主要增益，Mid 提供极限性能边际收益”，不支持
“rewrite 与 Mid 必须同时存在”的交互叙事。

## 构建机制消融

| ID | 配置 | 保留信号 | Mid / High | 成功率 | 相对 Full |
|---|---|---|---:|---:|---:|
| G0 | Full | S + T + O + signed contrastive | 39 / 118 | **88.06%** | - |
| G1 | Semantic-Only | S | 39 / 0 | 不执行 | 无合规 High |
| G2 | No Transition | S + O + signed contrastive | 19 / 76 | 51.49% | -36.57 pp |
| G3 | No Outcome | S + T + signed contrastive | 39 / 106 | 56.72% | -31.34 pp |
| G4 | No Contrastive | S + T + O | 10 / 44 | 60.45% | -27.61 pp |

G1 在正确执行 duplicate-embedding collapse 后无法形成合规端到端 High，因此它给出
结构性消融结论，不产生执行分数。G2-G4 的输入 token 均高于 Full，G4 的 context
字符数也高于 Full，但成功率显著更低，说明决定效果的是关系图质量而不是文本长度。

## 部署机制消融

### 正式可比结果

| ID | 运行时合同 | Mid | 成功 | 成功率 | 平均步数 | 平均无效动作 | 平均输入 token | 平均 context 字符 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D0 Full | `plan_rewrite/budgeted_v2` | 在场 | **118/134** | **88.06%** | **10.06** | **0.22** | 43,758 | 5,422 |
| D1 High-only | `plan_rewrite/budgeted_v2` | 缺席 | 115/134 | 85.82% | 10.50 | 0.22 | **37,410** | **1,960** |

D0 相对 D1 的成功率差为 +2.24pp，task bootstrap 95% CI 为
[-2.24pp, +7.46pp]；逐题配对为 7 胜、4 负、123 平，McNemar exact
`p=0.549`。因此，Mid 在当前设置下表现为小幅增量，而不是恢复 Full 所必需的条件。
D1 同时显著缩短 context，并减少约 6,348 个平均输入 token，说明正式 rewrite 已能
把图中的 High 阶段有效绑定到当前任务；Mid 的作用更适合表述为局部执行支持。

从系统增益角度看，无 Mid 的基础改进版已经释放整体系统 93.6% 的可观测潜力：
相对 No-Skill 提升 32.84pp。加入 Mid 后再获得 2.24pp 的边际收益，并达到当前最高的
88.06%。两者都远超 No-Skill，说明决定性优势来自层次图与正式 rewrite 构成的整体
架构；Mid 的价值在于继续压榨上限，而不是独自承担主要提升。

### 旧诊断结果的边界

D2（no-rewrite + High + Mid）和 D3（no-rewrite + High-only）仍使用 legacy
`high_to_mid` provider。它们可以保留为历史 no-rewrite 诊断，但与 D0/D1 的 provider
合同不同，不能再拼成正式的 rewrite x Mid 2x2 因果消融，也不能用 D1-D3 估计
rewrite 的独立贡献。本轮按要求只重跑错误的 D1，不扩展重跑 D2/D3。

## D1 重跑审计

- 作废运行：`alfworld-ablation-v1-high-only-rewrite-flash-r0`。
- 作废原因：配置名声称 rewrite，实际使用 SkillX-native
  `skillx-native:plan-rewrite:36747f4` 路径。
- 正式运行：`alfworld-ablation-v1-plan-rewrite-high-only-flash-r0`。
- Tower：`tower_d2c2d0090ed9b6b4`；manifest：`alfworld_test.jsonl`。
- 完整性：134 条结果、134 个唯一 `(sample_id, repeat_id)`、134 条轨迹，无缺失、
  无重复、无越界任务。
- 正式配置：`retrieval_strategy=plan_rewrite`、
  `rewrite_contract_version=budgeted_v2`、`reference_high_top_k=3`、
  `max_mid_skills=0`。

## 解释边界

- 构建消融与部署消融分表报告，不合并为跨层级排名。
- 消融差值只说明当前 P310/ALFWorld 条件下的组件贡献，不证明普遍必要性。
- `valid_unseen` 已参与最终链路诊断，因此结果作为配对机制证据，不包装成新的未见集
  泛化结论。
- 历史 no-event、task-conditioned、三信号运行时检索和错误 provider 结果只作为诊断，
  不并入正式结论。
