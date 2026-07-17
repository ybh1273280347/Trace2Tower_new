# Trace2Tower 实验结论

本文只保留能够支撑当前方法叙事的结果。旧 WebShop `same_event` 邻域、continuous
outcome、product-entity、decision-state-v8、goal-conditioned High 和多 renderer 分支已
隔离为历史消融；它们增加了复杂度，却没有形成稳定收益。新的约束分支图不再要求
WebShop 在节点表示上机械照搬 ALFWorld，而是在共同的事件图、成败对比和层级发现契约
下，为不同任务性质提供不同的领域决策表示。

## 当前方法

Trace2Tower 的跨数据集共同部分是：从领域事件构成的成败图中发现 Mid 行为社区和 High
组合主干。领域事件与图节点必须保留该任务真正决定成败的状态，不能强迫不同环境使用
同一种实体签名。

```text
domain trajectories
-> deterministic domain events
-> evidence-bearing decision nodes
-> semantic S + temporal T + outcome O graph
-> contrastive Mid communities
-> successful Mid-composition High communities
-> one-time High + optional retrieved Mid
```

ALFWorld 是低分支、长程、显式实体前置条件、二值完成率任务。图的价值主要表现为从
线性事件链中归纳“定位、取物、状态变换、放置/检查”等典型成功模式。WebShop 是高
分支、商品高度异质、属性逐页显露、允许部分 reward 的检索决策任务。图的价值主要
表现为从大量异质商品轨迹中发现“约束搜索、候选核验、选项绑定、购买”主干以及有证据
支持的恢复分支，而不是记忆商品事实。运行时 rewrite 无法凭空知道当前商品状态，因此
WebShop 正式配置关闭 rewrite。

## 有效主结果

### ALFWorld valid_unseen

ALFWorld 使用官方/AgentBench 动作口径，从 `take_action.action` 的原始字符串确定性
解析 primitive action，并以 reset observation 的真实任务为目标。修复后的实体/事件
签名将图节点从旧实现的 56 个不同文本恢复到 3,764 个 quotient node，谱社区从 8 个
恢复到 39 个，High 从少量粗模板恢复到 118 条真实成功组合。

| 方法                  |          成功 |        成功率 |      平均步数 |   平均无效动作 | 平均输入 token |
|---------------------|------------:|-----------:|----------:|---------:|-----------:|
| No-Skill            |      71/134 |     52.99% |     14.84 |     0.54 |     45,677 |
| Manual event policy |     102/134 |     76.12% |     11.49 |     0.37 | **38,889** |
| **Trace2Tower v18** | **118/134** | **88.06%** | **10.06** | **0.22** |     43,758 |，

Trace2Tower 相对 No-Skill 提高 35.07 个百分点，逐任务为 51 胜、4 负，95% 配对
bootstrap 区间为 `[+26.12,+44.03]`，McNemar 精确双侧 `p=2.05e-11`。相对
Manual 提高 11.94 个百分点，23 胜、7 负，区间 `[+4.48,+19.40]`，McNemar
`p=0.00522`。两项优势均显著。

| 任务族 | 任务数 | 成功 | 成功率 |
|---|---:|---:|---:|
| Clean | 31 | 30 | 96.8% |
| Cool | 21 | 18 | 85.7% |
| Heat | 23 | 18 | 78.3% |
| 灯下检查 | 12 | 11 | 91.7% |
| 单物体搬运 | 30 | 28 | 93.3% |
| 双物体搬运 | 17 | 13 | 76.5% |

改善覆盖全部任务族，不是单一操作类别造成的偶然优势。该结果是当前最强的跨域有效性
证据。

### WebShop 历史部署证据

以下结果来自早期 WebShop Tower 和 graph-cap3 部署算法。它们是有效实验观测，但其
旧图输入不再作为当前统一方法的概念实现，因此只能支持“关系结构和图感知检索具有
执行价值”，不能代替新同构图的正式结果。

| Flash Test-A repeat3 | reward | 完全成功率 | 步数 | 无效动作 | 输入 token |
|---|---:|---:|---:|---:|---:|
| **Final Tower graph-cap3** | **0.71119** | 53.67% | 6.83 | 0.16 | 20,461 |
| Legacy Tower cap8 | 0.70958 | **54.67%** | 7.14 | 0.17 | 32,334 |
| Manual | 0.69158 | 50.67% | **6.53** | **0.06** | **19,259** |
| No-Skill | 0.68492 | 50.33% | 7.89 | 0.38 | 20,610 |

Final Tower 相对 No-Skill 的 reward 差为 `+0.02628`，95% 区间
`[-0.02936,+0.08356]`，属于方向性证据。

| Pro Test-A repeat3 | reward | 完全成功率 | 步数 | 无效动作 | 输入 token |
|---|---:|---:|---:|---:|---:|
| **Final Tower graph-cap3** | **0.69919** | **51.33%** | 7.56 | 0.38 | 25,376 |
| Manual | 0.69194 | 51.00% | **6.59** | **0.04** | **18,700** |
| No-Skill | 0.59217 | 42.67% | 9.23 | 0.88 | 25,672 |

Final Tower 相对 No-Skill 的 reward 提升为 `+0.10703`，区间
`[+0.06255,+0.15334]`；完全成功率提高 8.67 个百分点，区间
`[+3.33,+14.33]`。两项均显著。

| Flash Test-B repeat0 | reward | 完全成功率 | 步数 | 无效动作 | 输入 token |
|---|---:|---:|---:|---:|---:|
| **Final Tower** | **0.75123** | **53%** | **6.87** | **0.08** | 19,637 |
| No-Skill | 0.73323 | 48% | 7.32 | 0.32 | **16,571** |
| Legacy P100 Tower | 0.71465 | 49% | 7.13 | 0.18 | 31,236 |

Final Tower 相对 No-Skill 为 `+0.01800`；No-Skill repeat1 为 0.72798，说明 Test-B
高基线不是一次偶然异常。该优势仍是方向性、split-sensitive 证据。

### 图感知检索隔离

在同一 T1 Tower 上只替换检索方式：

| Test-A repeat0 | reward | 完全成功率 | 步数 | 无效动作 | 输入 token |
|---|---:|---:|---:|---:|---:|
| Legacy cap8 | 0.67983 | 52% | 7.66 | 0.25 | 35,273 |
| Legacy cap3 | 0.69283 | 53% | 7.16 | 0.21 | 30,242 |
| Graph cap8 | 0.71442 | 54% | 7.56 | 0.21 | 27,808 |
| **Graph cap3** | **0.71925** | **54%** | **6.81** | **0.16** | **20,059** |

Graph cap3 比 legacy cap8 高 `+0.03942`，区间 `[-0.00500,+0.09025]`，同时少用
43.1% 输入 token。扩大到 cap8 没有带来效果，说明收益来自结构化选择，而非注入更多
文本。

## 新 WebShop 同构图

从 P200 原始轨迹重新构建，不复用任何被删除版本的中间产物：

| 指标 | 数值 |
|---|---:|
| 训练轨迹 | 800 |
| 事件片段 | 5,419 |
| quotient nodes | 1,855 |
| 折叠重复片段 | 3,564 |
| 全局图边 | 13,803 |
| 跨事件边 | 9,997 |
| Mid communities | 50 |
| 真实成功 High paths | 17 |

snapshot 为 `tower_f45724ac610f0035`，50 个 Mid 和 17 个 High 均由唯一 Tower
renderer 使用 GPT-5.4 重新生成，覆盖完整。

### 同构图 + SkillX 官方后处理

任务期使用冻结官方代码的实际后处理顺序：High Top-3 后只取首个、GPT-5.4 rewrite、
按 rewritten High 的步骤逐步召回 Mid、候选超过 8 条才调用 selector；最终只注入
rewritten High 与选中的 Mid。Tower 额外保证 rewrite 失败时回退原始 High，因此每个
任务始终注入一个端到端 High。

| 集合 | 方法 | mean reward | 满分率 | 平均步数 | 平均无效动作 | 平均输入 token |
|---|---|---:|---:|---:|---:|---:|
| 验证集 | No-Skill | 0.65235 | 47% | 7.69 | 0.30 | 19,572 |
| 验证集 | SkillX no-rewrite r1 | 0.64393 | 45% | 8.01 | 0.37 | 30,666 |
| 验证集 | SkillX no-rewrite r2 | 0.66285 | 47% | 8.07 | 0.39 | 31,366 |
| 验证集 | SkillX 官方实际入口 | **0.68427** | 48% | 7.95 | 0.29 | 34,044 |
| 验证集 | Trace2Tower P200 no-rewrite | 0.64477 | 45% | 9.35 | 0.28 | 39,651 |
| 验证集 | Trace2Tower P200 | 0.68035 | **49%** | 10.56 | **0.20** | 52,912 |
| TestA | No-Skill | **0.68075** | **51%** | 7.98 | 0.36 | **21,388** |
| TestA | SkillX no-rewrite | **0.72050** | **53%** | 7.43 | 0.29 | 28,072 |
| TestA | SkillX 官方实际入口 | 0.66490 | 49% | 8.54 | 0.31 | 38,834 |
| TestA | Trace2Tower P200 no-rewrite | 0.64000 | 50% | 11.16 | 0.29 | 52,055 |
| TestA | Trace2Tower P200 | 0.65950 | 49% | 10.35 | **0.22** | 54,557 |

两种图/技能方法都在验证集小幅超过 No-Skill，却同时在 TestA 下降。Trace2Tower 相对
SkillX 在验证集为 `-0.00392`，TestA 为 `-0.00540`，新图没有形成可归因的增益。
因此链路已经跑通，但新同构图的跨 split 效果失败，不能写成 WebShop 主结果。

离线召回还显示 17 个 High 在 TestA 只使用了 8 个，其中服装变体 High 覆盖 64%，
海苔零食 High 覆盖 24%。这说明当前主要问题不是缺少 rewrite，而是少数商品族 High
承担了绝大多数异类任务；rewrite 又把执行过程拉长到 10.35 步，增加了核验和试错。

no-rewrite 消融进一步定位了差异：Tower 原始 High 在验证集/TestA 只有
`0.64477/0.64000`，rewrite 分别带来 `+0.03558/+0.01950`，说明当前 Tower High
确实依赖任务适配；SkillX 原始具体 Plan 则在 TestA 达到 `0.72050`，rewrite 后下降
`0.05560`。因此后续不是统一关闭 rewrite，而是提高 Tower High 的任务覆盖与具体性，
使其在不依赖强模型重写时也能直接指导执行。

## WebShop 约束分支图

### 图与社区发现

P200 的 800 条训练轨迹首先按官方 WebShop 事件切分，再把商品实例泛化为可观测决策
状态：约束数量、候选匹配证据、选项完成度、价格证据、重复试错和当前事件。商品名、
ASIN 和未观测属性不进入节点主键。新图仍显式保留语义、时序、结果三类关系。

| 图结构 | 数量 |
|---|---:|
| 训练轨迹 | 800 |
| 决策节点 | 184 |
| 总边 | 2,046 |
| 真实观测转移边 | 472 |
| 自动谱 Mid 社区 | 3 |
| 正对比 High 路径 | 3 |
| 自动 High 社区 | 2 |
| High 社区模块度 | 0.06461 |

3 个 Mid 分别对应候选/选项/购买主链、查询/详情证据、回退/重搜恢复。High 社区数不预设，
而是在成功轨迹的 Mid 与有向 Mid 转移图上递归执行正模块度分裂。真实数据自动得到两个
社区：101 条成功轨迹组成“含恢复分支”社区，277 条组成“直接完成”社区。因此最终实现
没有人为把所有 Mid 合成一个全局社区，也没有按商品族或选项数手工分桶。

### 图主干 High 主结果

将图中三个策略子结构的共同成功主干确定性压成一张 1,193 字符 High，只注入一次，
不 rewrite、不注入 Mid。这是 WebShop 的主部署形式：技能直接来自图结构中的稳定
高层组合关系，商品事实仍由运行时页面提供。

| split | No-Skill reward | 图主干 High reward | 差值 | 满分率 | 步数 | 输入 token |
|---|---:|---:|---:|---:|---:|---:|
| 验证集 | 0.65235 | **0.70402** | **+0.05167** | 49% | **6.82** | **17,298** |
| TestA | 0.68075 | **0.71432** | **+0.03357** | 49% | **6.39** | **16,290** |

验证集相对 No-Skill 为 10 胜、84 平、6 负，reward 配对区间
`[+0.00117,+0.10750]`；TestA 为 11 胜、79 平、10 负，区间
`[-0.02068,+0.09099]`。它没有提高二值满分率，却同时提高平均匹配 reward、减少 1.59
步并减少输入 token。这说明 WebShop 的稳定价值首先来自紧凑图主干，而不是运行时补写
具体商品知识。

图发现的两个 High 社区分别对应“直接完成”和“运行中需要恢复”。恢复需求只有看到搜索
结果或商品页后才可观测，任务开始时无法可靠检索。因此它不应被提升为另一张初始 High，
而应作为条件恢复分支写入同一张全局主干卡。这里合并的是不可初始判定的执行分支，不是
把所有社区假装成一个社区。

### 层级接入消融

消融 snapshot `tower_ef59dccffa7228ac` 保留自动发现的 2 High、3 Mid 和 3 条 High
路径。High 一次性 Top-1 注入；Mid 按 High 步骤召回，最多 3 张；rewrite 关闭。所有卡
都由图角色压成短执行单元，避免早期 LLM renderer 把社区重新展开为冗长核验清单。

| split | 方法 | reward | 满分率 | 步数 | 无效动作 | 输入 token |
|---|---|---:|---:|---:|---:|---:|
| 验证集 | No-Skill | 0.65235 | 47% | 7.69 | 0.30 | 19,572 |
| 验证集 | SkillX no-rewrite r1 | 0.64393 | 45% | 8.01 | 0.37 | 30,666 |
| 验证集 | SkillX no-rewrite r2 | 0.66285 | 47% | 8.07 | 0.39 | 31,366 |
| 验证集 | SkillX 官方实际入口 | 0.68427 | 48% | 7.95 | 0.29 | 34,044 |
| 验证集 | **Trace2Tower compact** | **0.68785** | **50%** | **7.05** | **0.14** | **17,908** |
| TestA | No-Skill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| TestA | SkillX no-rewrite | **0.72050** | **53%** | 7.43 | 0.29 | 28,072 |
| TestA | SkillX 官方实际入口 | 0.66490 | 49% | 8.54 | 0.31 | 38,834 |
| TestA | **Trace2Tower compact** | 0.69092 | 52% | 7.82 | **0.13** | 23,240 |

层级版本相对 No-Skill 在验证集为 `+0.03550`，9 胜、87 平、4 负，95% 配对区间
`[-0.00817,+0.08450]`；TestA 为 `+0.01017`，9 胜、81 平、10 负，区间
`[-0.05200,+0.07375]`。两者均是方向性而非显著优势，但它是当前表中唯一在两个 split
都高于 No-Skill 的多卡技能方法。相对 SkillX 官方入口，验证集/TestA 分别为
`+0.00358/+0.02601`。

同一 compact snapshot 的 2-High-only 在 TestA 只有 `0.66475`，加入 3 个紧凑 Mid 后
提高到 `0.69092`，说明在保留两个真实 High 社区时，Mid 对局部阶段仍有补偿作用；但
单张全局主干 High 的 `0.71432` 更高。原因不是全局 High 更“忠实”，而是两个社区的
边界依赖运行中状态，初始检索无法判定；把它们拆成两张任务级卡反而破坏了条件分支。

### 与 SkillX 的共同退化

WebShop 对冗长、重复核验的技能上下文高度敏感。验证集 no-rewrite 独立复跑从
`0.64393` 变为 `0.66285`，两轮均值 `0.65339`，与 No-Skill `0.65235` 基本持平，
因此不能用较差的 r1 单轮概括 SkillX，也不能用较好的 r2 单轮宣称稳定优势。旧
LLM-rendered Trace2Tower 在 TestA
为 `0.65950 / 10.35 步 / 54,557 token`，SkillX 官方 rewrite 为
`0.66490 / 8.54 步 / 38,834 token`，都低于 No-Skill `0.68075`。SkillX 关闭 rewrite
后 TestA 上升到 `0.72050`，但验证集两轮均值只有 `0.65339`；其优势无法跨 split
保持。
Trace2Tower 的修正不是让 rewrite 猜商品，而是从图结构直接归纳紧凑 High，并将不可
初始判定的恢复模式保留为卡内条件分支。

## SkillX 官方代码 baseline

baseline 现只按冻结官方代码 `36747f4` 的实际主入口处理，不采用论文叙事。主入口为：
Plan Top-3/0.45、只取第一条、rewrite 覆盖原 Plan、逐步骤 Top-4 Function、按名称去重、
候选超过 8 条才 selector，并注入 rewritten Plan 与 Function。官方 commit 的参数名错误
只在项目适配层修复，vendor 不修改。详细证据见
`../../SKILLX_INFERENCE_CONTRACT_AUDIT.md`。

旧 P100 SkillX 没有 rewrite，TestA 为 `0.71224 / 49%`；官方实际入口版本为
`0.66490 / 49%`。rewrite 没有改变满分率，却使 reward 下降 `0.04733`、平均步数从
6.92 增至 8.54。因此旧结果只保留为 rewrite 消融，新的两个 100-task run 是冻结
baseline。

## 限制

- ALFWorld v18 是 repeat0，且运行时契约经过同一 split 的失败样本诊断，需要独立
  repeat 或冻结 split 验证。
- ALFWorld v18 使用 GPT-5.4 运行时改写/过滤、DeepSeek v4 Flash 执行，成本和模型
  能力不对称必须报告。
- WebShop 历史高分图不是当前统一实现；新同构图在验证集和 TestA 分别为 0.68035 与
  0.65950，尚未复现历史优势。
- WebShop 图主干 High 只有 repeat0；验证集 reward 区间刚好高于 0，TestA 区间仍包含
  0，因此不能宣称跨 split 显著优势。当前价值是两个 split 的一致正方向、短执行链和
  低 token 成本。
- WebShop 优化平均 reward，ALFWorld 优化二值完成率；领域执行 prompt 可以反映评分
  目标。两者共享事件图和层级发现契约，但节点表示不应机械同构。

## 权威文件

- WebShop 同构配置：`configs/experiments/webshop_trace2tower_alfworld_isomorphic.yaml`
- WebShop snapshot：`artifacts/trace2tower/webshop/alfworld-isomorphic/p200/tower.json`
- ALFWorld snapshot：`artifacts/trace2tower/alfworld/original-concept-v17/p310/tower.json`
- ALFWorld 主运行：`artifacts/runs/alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0/`
- SkillX 审计：`experiments/SKILLX_INFERENCE_CONTRACT_AUDIT.md`
- SkillX TestA：`artifacts/runs/webshop-skillx-native-inference-p100-testa-flash-r1/`
- SkillX 验证集：`artifacts/runs/webshop-skillx-native-inference-p100-validation-flash-r1/`
- 新同构图 TestA：`artifacts/runs/webshop-native-postprocess-p200-testa-flash-clean-r1/`
- 新同构图验证集：`artifacts/runs/webshop-native-postprocess-p200-validation-flash-r1/`
- SkillX no-rewrite TestA：`artifacts/runs/webshop-skillx-no-rewrite-p100-testa-flash-r1/`
- SkillX no-rewrite 验证集：`artifacts/runs/webshop-skillx-no-rewrite-p100-validation-flash-r1/`
- SkillX no-rewrite 验证集复跑：`artifacts/runs/webshop-skillx-no-rewrite-p100-validation-flash-r2/`
- Tower no-rewrite TestA：`artifacts/runs/webshop-tower-no-rewrite-p200-testa-flash-r1/`
- Tower no-rewrite 验证集：`artifacts/runs/webshop-tower-no-rewrite-p200-validation-flash-r1/`
- WebShop 约束分支图：`artifacts/trace2tower/webshop/constraint-branch-v1/p200/branch-graph.json`
- WebShop compact snapshot：`artifacts/trace2tower/webshop/constraint-branch-v1/p200/tower-compact.json`
- WebShop compact TestA：`artifacts/runs/webshop-constraint-branch-v1-compact-full-testa-flash-r0/`
- WebShop compact 验证集：`artifacts/runs/webshop-constraint-branch-v1-compact-full-validation-flash-r0/`
- WebShop 图主干 TestA：`artifacts/runs/webshop-constraint-branch-v1-generalized-high-testa-flash-r0/`
- WebShop 图主干验证集：`artifacts/runs/webshop-constraint-branch-v1-generalized-high-validation-flash-r0/`
