# ALFWorld 官方事件 Graph + High 验证

## 结论

在同一组 139 个 `valid_seen` 样本、同一 `deepseek-v4-flash`、仅
`repeat_id=0` 的配对实验中，修复后的官方事件 Trace2Tower 与 No-Skill 都成功
85 个，成功率同为 61.15%。配对翻转为 8 胜 8 负，McNemar 双侧精确检验
`p=1.0`；10,000 次配对 bootstrap 的 95% CI 为 [-5.76, +5.76] 个百分点。

这轮结果证明修复消除了旧无事件实现的整体负收益：旧 P310 Mid-only cap3 为
55.40%，当前修复版提高 5.76 个百分点并回到 No-Skill 水平。但它仍没有超过
No-Skill，也明显低于同 cohort 的 Global SkillX 81.29%。

## 实验合同

| 项目       | 值                                                         |
|----------|-----------------------------------------------------------|
| Cohort   | 与既有 Flash No-Skill 完全相同的 139 个 `valid_seen` 样本            |
| Repeat   | `repeat_id=0`，不重复实验                                       |
| Agent    | `deepseek-v4-flash`                                       |
| 评估       | AgentBench 二值成功，最多 20 agent turns                         |
| Tower    | `tower_4ff0dbf6b2420d21`                                  |
| 训练池      | P310，全局 310 tasks × 4 repeats，共 1,240 条轨迹                 |
| Runtime  | `include_high=true`、graph retrieval、cap3                  |
| High     | 2% 原阈值自然产生 12 条，未使用 fallback                              |
| Run      | `alfworld-dev-v1-flash-official-event-graph-high-cap3-r0` |
| No-Skill | 复用 `alfworld-dev-v1-flash-noskill-r1`，未新增调用               |

## 总体结果

| 条件                     |     成功数 |    成功率 | 相对 No-Skill |   平均步数 | invalid rate |
|------------------------|--------:|-------:|------------:|-------:|-------------:|
| No-Skill               |  85/139 | 61.15% |           - | 13.820 |        3.54% |
| 旧无事件 Mid-only cap3     |  77/139 | 55.40% |    -5.76 pp |      - |            - |
| 官方事件 Graph + High cap3 |  85/139 | 61.15% |     0.00 pp | 14.000 |        3.80% |
| Global SkillX          | 113/139 | 81.29% |   +20.14 pp |      - |            - |

官方事件 Tower 相对 No-Skill 的平均输入 tokens 增加 8,937/episode，平均输出
tokens 增加 8.13/episode，平均延迟增加约 14.16 秒。`unresolved_failure_count=0`。
运行初期 20 并发产生的 97 条 RPM 429 均由同 run ID 的低并发 checkpoint 续跑
补齐，不计为最终失败。

## 配对翻转

| 类型 | 数量 |
|---|---:|
| No-Skill 失败 → Tower 成功 | 8 |
| No-Skill 成功 → Tower 失败 | 8 |
| 结果相同 | 123 |
| McNemar 双侧精确 `p` | 1.0 |

## 按目标事件分层

| 目标事件         |  N |      No-Skill |         Tower | 翻转（胜/负） | High episode |
|--------------|---:|--------------:|--------------:|--------:|-------------:|
| CleanObject  | 26 | 12/26（46.15%） | 14/26（53.85%） |     4/2 |           26 |
| ToggleObject | 13 | 12/13（92.31%） | 11/13（84.62%） |     1/2 |           13 |
| CoolObject   | 22 |  3/22（13.64%） |  4/22（18.18%） |     2/1 |            0 |
| HeatObject   | 14 |  9/14（64.29%） |  8/14（57.14%） |     0/1 |            0 |
| Cool + Heat  |  1 |           0/1 |           0/1 |     0/0 |            0 |
| 普通搬运         | 63 | 49/63（77.78%） | 48/63（76.19%） |     1/2 |            0 |

Clean 子集呈现 +7.69 pp 的正方向，但只有 26 个样本，4 胜 2 负的 McNemar
`p=0.6875`，不能宣称显著改善。Toggle 子集为 -7.69 pp。其余 100 个任务因
图中没有事件完全匹配的可信 High，按合同自动回退到 graph Mid-only。

## High 实际使用

High 仅出现在 39/139 个目标事件匹配的 episode 中；由于 Tower 每步刷新检索，
共记录 527 次 High context 注入。

| High                | 覆盖 episode | 注入 steps | 主要语义         |
|---------------------|-----------:|---------:|--------------|
| `high_9eeb22dc6f5e` |         13 |      192 | 取物、清洁、前往下一位置 |
| `high_797cbcdd9245` |          8 |      151 | 清洁后前往并放置     |
| `high_3e5aeae29257` |          4 |       49 | 前往 sink 并清洁  |
| `high_b921c9fef095` |          1 |       14 | 清洁后搬运        |
| `high_fac31ad72925` |         13 |      121 | 取物、检查、移动与照明  |

## 解释边界

- 有利结论：官方事件接入、角色化签名、自然 High 和事件门控共同把旧无事件实现
  的 -5.76 pp 修复到与 No-Skill 持平；Clean High 出现正向局部信号。
- 不利结论：总体没有优势，置信区间仍宽；Toggle High 有负方向；运行成本增加。
- 不能把本轮写成 Trace2Tower 已在 ALFWorld 胜出。它证明的是正确概念接入避免了
  错误实现导致的全面溃败，并产生了可继续验证的 High 局部信号。

机器评估位于
`artifacts/evaluations/alfworld-dev-v1-flash-official-event-graph-high-cap3-r0/`。

## 失败集三信号检索诊断（v9）

原始资料中的 $S_{uv}$、$T_{uv}$、$O_{uv}$ 已用于训练期 EigenTrace 图构建和
Mid 谱聚类，但旧运行时所谓“graph 原生检索”并未显式联合计算三项：High 主要
依赖目标语义和事件签名，Mid 主要依赖状态语义和事件兼容。因此新增隔离的
`three_signal_retrieval.py`，不修改既有 `graph_retrieval.py`，验证部署时显式使用
语义相似性、Mid 时序转移强度和转移成败一致性是否能救回失败样本。

离线前 10 条、137 个执行 step 的审计结果如下：

| 检索条件 | Mid 注入覆盖率 | 与实际 primitive action 的事件对齐精度 |
|---|---:|---:|
| 严格语义 Mid，阈值 0.60 | 4.38% | 50.00% |
| 三信号，阈值 0.70 | 5.84% | 50.00% |
| 三信号，阈值 0.75 | 2.92% | 75.00% |
| 三信号，阈值 0.80 | 0.00% | - |

选择 `0.75` 进行在线失败集验证，运行契约为 High Top-1 开局注入一次、Mid
Top-2 动态检索、Low 关闭。完整结果为 24/61（39.34%），没有达到 31/61 的
救回门槛。与 v9 多社区 High-only 的 60 条可配对样本相比，两者成功集合完全
相同：共同成功 24 条、三信号独有成功 0 条、High-only 独有成功 0 条。

| 失败集条件 | 成功数 | 成功率 | 平均累计 skill context 字符数 |
|---|---:|---:|---:|
| v6 局部动态 Graph + High | 7/61 | 11.48% | 61,670 |
| v7 端到端路径 High-only | 26/61 | 42.62% | 2,027 |
| v8 单全局 High 诊断上界 | 31/61 | 50.82% | 2,276 |
| v9 多社区 High-only | 24/60 | 40.00% | 1,805 |
| v9 多社区 High + 三信号 Mid | 24/61 | 39.34% | 2,223 |
| 手写事件策略 | 42/61 | 68.85% | 3,286 |

三信号版本仅在 15/61 个 episode 中实际注入 Mid，其中成功 6 个（40.00%）；
未注入 Mid 的 46 个 episode 成功 18 个（39.13%）。这说明三信号将离线事件
对齐精度从 50% 提高到 75%，但没有转化为任务成功率或样本级翻转。

因此当前证据不支持用三信号检索替换主方法，也不能把它叙述为相对语义 Mid 的
在线优势。可以保留的结论是：三信号部署检索在概念上更忠实、能形成更稀疏且
更精确的 Mid 门控，但 ALFWorld 失败集的主要效果仍由 High 的全局信息覆盖决定；
v8 单全局 High 明显强于 v9 多社区 High 和任何动态 Mid 变体，说明下一步应优先
解决多社区 High 如何保留跨社区共享的全局恢复规则，而不是继续优化 Mid Top-K。

离线审计：
`experiments/alfworld/official/validation/three-signal-retrieval-first10-strict-audit.json`。
在线运行：
`artifacts/runs/alfworld-dev-v1-flash-high-mid-three-signal-v9-first10-r0/`（由前 10 条
checkpoint 扩展至完整 61 条，run ID 保留历史名称）。

## 人工阶段 Mid 因果诊断（v10）

三信号版本实际注入 Mid 但仍失败的样本共有 9 条。为区分“Mid 卡无效”和
“Mid 自动召回不准”，新增隔离的 `labeled_mid_diagnostic` 诊断入口：High 仍按
v9 多社区算法在开局 Top-1 注入一次；关闭自动 Mid 与 Low；人工为每条任务标注
动作词表中的规范目标对象、目的地和变换 Mid，并仅在确定的局部阶段注入已有
`mid_0000`（开容器/取物/放置）、`mid_0005`（清洗）或 `mid_0006`（冷却）。
没有新增手写策略文本，也没有修改 Mid 卡内容。

结果为 **0/9**，相对同一批样本的三信号自动 Mid **0/9** 没有救回任何样本。
人工 Mid 平均累计注入 8,821 字符，明显高于正式设置，因此该结果不是“提示信息
太少”造成的；相反，它说明增加低信息量局部卡只会提高上下文成本。

| 样本 | 人工注入 Mid | 与自动 Mid 轨迹对比 | 主要失败原因 |
|---|---|---|---|
| `T20190906_215856_374127` | `mid_0000` | 20 步动作完全相同 | 目标是 pan，但搜索后拿了 pot；未进入冷却阶段 |
| `T20190907_233056_022802` | `mid_0000`、`mid_0005` | 20 步动作完全相同 | 第 18 步才取到 mug，第 19 步才到 sink，步数耗尽 |
| `T20190908_233922_171295` | `mid_0000` | 20 步动作完全相同 | 顺序扫描容器，未找到 fork |
| `T20190909_004531_429065` | `mid_0000`、`mid_0006` | 20 步动作完全相同 | 第 17 步才取到 pot，只完成前往/打开 fridge |
| `T20190909_044933_815840` | `mid_0000`、`mid_0006` | 20 步动作完全相同 | 第 18 步才取到 apple，第 19 步才返回 fridge |
| `T20190909_113844_191747` | `mid_0000` | 仅增加一次无关导航 | 第 19 步才取到规范对象 ladle，来不及清洗和放置 |
| `T20190909_121908_219603` | `mid_0000` | 20 步动作完全相同 | 顺序扫描大量 cabinet，未找到 ladle |
| `T20190909_183724_205399` | `mid_0000` | 前 7 步同构，之后均为检查循环 | 把自然语言 spoon 绑定到错误对象，清洗并放置了 spoon 而非 ladle |
| `T20190911_131350_027076` | `mid_0000` | 前 7 步同构，之后均为检查循环 | 把自然语言 spoon 绑定到错误对象，清洗并放置了 spoon 而非 ladle |

其中 6/9 条轨迹的 20 个动作与自动 Mid 版本逐步完全一致；另外 3 条只发生无关
导航或失败后的检查顺序变化。`mid_0005/0006` 在到达变换阶段时确实提示了去
sink/fridge，但模型原本就采取了这些动作。`mid_0000` 在关闭容器、目标可取或
可放置时也只复述了 admissible actions 中显而易见的局部操作，不能决定搜索优先级、
目标对象绑定和全局停止条件。

因此当前失败不是单一召回问题，而是两层问题叠加：

1. 自动三信号检索受 `GOTO` 转移 Hub 支配，28 次动态注入全部选择 `mid_0001`，
   阶段识别确实错误。
2. 即使人工消除召回误差，现有 Mid 的信息增益仍接近零。难例所缺的是 High/全局
   对比图应提供的搜索优先级、对象词汇绑定、阶段完成检查和失败恢复规则，而不是
   更多“打开、拿取、清洗、冷却、放置”的局部操作说明。

### 官方目标文本噪声

诊断还确认 3 个 trial 的 AgentBench parquet `goal_text` 使用 `spoon` 或
`large metal spoon`，但同一游戏路径、规范任务和可执行动作词表使用 `ladle`。
这不是 manifest 索引错位；官方 `traj_data.json` 的多条人工释义本身同时混用了
spoon 与 ladle。所有方法承受相同输入，因此横向比较仍然成立，但这类样本会显著
考验自然语言目标到精确动作词表的绑定能力，必须在失败分析中单列，不能误归因于
Mid 检索。

人工标签：
`experiments/alfworld/official/validation/labeled-mid-failure-diagnostic.json`。
在线运行：
`artifacts/runs/alfworld-dev-v1-flash-high-labeled-mid-v10-nine-failures-r0/`。

## 对象条件化 High 诊断（v11）

v9 的五张 High 卡虽然覆盖了搬运、加热、照明、清洗和冷却的端到端事件链，
但渲染内容把训练社区中的对象、来源和目的地全部泛化为 `target item`、
`named destination` 和 `relevant fixture`。这不是社区发现本身要求的抽象，
而是渲染契约丢失了具体任务条件。离线审计据此增加对象条件化层：只消除
`ladle 2` 之类实例编号，保留 `ladle`、`sinkbasin`、`countertop`、变换事件和
数量关系。

P310 的 850 条成功轨迹包含 179 个 canonical goal；其中 160 个目标至少有 4 条
成功支持，共覆盖 813/850（95.65%）成功轨迹。若把事件、对象、来源和目的地
全部硬编码为社区 ID，会碎成 205 个原型；若以 canonical goal 形成具体任务
子社区，则训练池的 repeat 支持足以构建 160 张任务卡。因此正确实现不是回到
五张高度抽象的任务族卡，也不是把每个场景实例切成独立簇，而是保留事件图的
Mid 结构，并在 High 社区内保存可检索的具体任务原型。

检索键与注入内容也必须拆分。v11 只对 160 条 canonical goal 建 High 索引，
具体执行卡仍保留完整动作链和失败约束；索引总计仅消耗 1,533 embedding input
tokens。对 9 条人工 Mid 仍失败的难例，纯语义 Top-1 即使使用 canonical goal，
仍会把 `clean mug` 召回成 `cool mug`，或把 `countertop` 召回成
`diningtable/cabinet`。结构门控后，目标对象匹配 9/9，对象与官方事件同时匹配
8/9，目的地匹配 5/9，三者完全匹配 4/9。正式门控因此要求对象、事件和目的地
完全兼容，语义相似度只在兼容卡之间排序；无兼容卡时不得注入错误具体卡。

实验还定位并修复了 ALFWorld 适配器错误：环境 reset observation 已给出真实
`Your task is to:`，但旧实现仍把 parquet 的外部 `goal_text` 作为主任务提示。
三条噪声样本的外部标题写 `spoon/large metal spoon`，真实任务和动作词表使用
`ladle`。仅让检索使用 canonical goal 仍会被主提示中的错误 `spoon` 压制；
修复后，ALFWorld 统一使用 reset observation 的 canonical goal，解析失败才回退
外部标题。该修复对所有方法一致，不是 Tower 特判。

在线结果如下。九条样本此前在 v9 三信号 Mid 与人工 Mid 下均为 0/9：

| v11 条件 | 样本数 | 成功数 | 结果 |
|---|---:|---:|---:|
| 对象 + 事件 + 目的地有完全匹配具体社区 | 4 | 4 | 100.00% |
| 无完整三元组，按同对象/同事件链查询时绑定具体目标 | 5 | 1 | 20.00% |
| 合计 | 9 | 5 | 55.56% |

完全匹配的 4 条分别为 `cool pan -> stoveburner`、`cool pot -> diningtable`，以及
两条 `clean ladle -> diningtable`，全部成功。后两条原本受外部 `spoon` 噪声
影响，canonical 主任务修复后均正确拿取、清洗并放置 ladle。fallback 唯一成功
为 `clean mug -> coffeemachine`；其余 4 条都在 20 步内搜索耗尽，没有再次执行
错误变换或错误目的地。这说明任务绑定问题已经得到实质修复，剩余瓶颈是社区
没有提供足够可靠的对象来源先验，而不是需要重新把技能抽象成 functional role。

这组 5/9 是算法诊断，不直接替代旧 NoSkill、Manual、SkillX 的正式横向结果。
原因是 ALFWorld 主任务文本契约已经从脏外部标题修正为环境 canonical goal；最终
测试必须让所有 baseline 使用同一适配器后重跑，才能进行公平比较。

离线产物：

- `experiments/alfworld/official/validation/object-conditioned-community-audit.json`
- `experiments/alfworld/official/validation/object-conditioned-retrieval-failure-audit.json`

在线运行：

- `artifacts/runs/alfworld-dev-v1-flash-object-conditioned-high-v11-canonical-exact4-r0/`
- `artifacts/runs/alfworld-dev-v1-flash-object-conditioned-bound-high-v11-fallback5-r0/`

上述 v11 诊断随后已迁移到统一 `DomainTaskAdapter + TaskConditionProfile` 接口。
ALFWorld 对象、事件和目的地仅存在于领域适配器，核心 provider 和检索器不读取这些
字段；WebShop 已使用同一接口完成真实 P100 T1 快照加载和在线冒烟。统一契约与
跨域边界见 `experiments/TASK_CONDITIONING_INTERFACE.md`。
