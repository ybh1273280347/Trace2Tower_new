# ALFWorld Global SkillX Flash 单次验证

## 结论

在同一组 139 个 `valid_seen` 样本、同一 `deepseek-v4-flash`、仅 `repeat_id=0` 的配对实验中，Global SkillX 成功率为 81.29%，高于 NoSkill 的 61.15% 和 P310 Mid-only cap3 的 55.40%。

SkillX 相对 NoSkill 提高 20.14 个百分点：32 个 NoSkill 失败样本转为成功，4 个 NoSkill 成功样本转为失败，McNemar exact 双侧检验 `p=1.94e-6`。这是当前 ALFWorld 单次验证中明确的正向结果。

## 任务族边界审计

旧 artifact `artifacts/skillx/alfworld-success-family-v1/combined/library.json` 是六个任务族分别构建后合并的，并在 plan 文本中加入了 `Task family:` 前缀。该 artifact 不满足泛任务全局构建口径；本轮曾启动 22 个样本后即停止，结果不进入统计。

本轮使用的新 artifact 为 `skillxlib_9f23af185cf6fe0f`。构建过程如下：

1. 输入原 P310 训练池中的 850 条成功轨迹，不包含验证或测试轨迹；
2. 复用逐任务独立产生的 850 条摘要、200 个原始 plan 和 22 个已通过 SkillX 过滤的候选；这些中间结果的 prompt 不含任务族标签；
3. 对全池缺失的 24 个任务目标补做 plan 抽取，得到 224 个全局 plan；
4. 在 22 个候选上重新执行一次跨全池聚类，不保留原 family 分区；
5. 10 个全局簇中有 5 个 merge 未通过 SkillX 官方解析器，按照 SkillX 的事务规则回滚到原始候选，最终执行库包含 224 个 plan 和 19 个 skill；
6. 原始 plan、全局 recovery 输入和最终执行库中 `Task family:` 前缀数均为 0；运行时只按任务目标做向量检索，不读取 `task_family`。

因此，旧任务族目录在本轮只作为可恢复计算分片，任务族标签和分区边界没有进入新的全局聚类、最终库或运行时检索。

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | `deepseek-v4-flash` |
| Cohort | 与 Flash NoSkill、Mid-only cap3 完全相同的 139 个 `valid_seen` 样本 |
| Repeat | 仅 `repeat_id=0` |
| 训练来源 | 原 P310 训练池中的 850 条成功轨迹 |
| SkillX library | 224 plans，19 skills，全局索引 |
| 检索 | plan top3、每步 skill top4、总 skill 上限8 |
| NoSkill run | `alfworld-dev-v1-flash-noskill-r1` |
| Mid-only run | `alfworld-dev-v1-flash-p310-mid-only-cap3-r1` |
| SkillX run | `alfworld-dev-v1-flash-skillx-global-p310-r1` |

## 总体结果

| 指标                   |    NoSkill | Mid-only cap3 | Global SkillX |
|----------------------|-----------:|--------------:|--------------:|
| 成功数                  |   85 / 139 |      77 / 139 | **113 / 139** |
| 成功率                  |     61.15% |        55.40% |             1 |
| 相对 NoSkill           |          - |      -5.76 pp | **+20.14 pp** |
| 平均步数                 |      13.82 |         14.42 |     **10.48** |
| 成功样本平均步数             |       9.89 |          9.92 |      **8.29** |
| 平均 invalid action    |       0.49 |          0.63 |      **0.29** |
| 平均输入 token           | **44,631** |        56,058 |        59,434 |
| 平均输出 token           |        560 |           585 |       **432** |
| 平均 skill context 字符数 |          0 |        33,856 |    **13,268** |

SkillX 使用更多输入 token，但同时提高成功率、减少步数、减少无效动作并降低输出 token。收益不是通过更长 episode 获得。

## 配对结果

| 比较 | 正向翻转 | 反向翻转 | McNemar exact p |
|---|---:|---:|---:|
| SkillX 相对 NoSkill | 32 | 4 | 1.94e-6 |
| SkillX 相对 Mid-only cap3 | 38 | 2 | 1.49e-9 |

## 按任务族

任务族只用于结果分解，不参与新库构建或运行时检索。

| 任务族 | 样本数 | NoSkill | Mid-only cap3 | Global SkillX |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 13 | 12 | 12 | **13** |
| `pick_and_place` | 35 | 33 | 29 | **34** |
| `pick_clean_then_place` | 27 | 13 | 13 | **20** |
| `pick_cool_then_place` | 24 | 4 | 3 | **16** |
| `pick_heat_then_place` | 16 | 11 | 7 | **13** |
| `pick_two_obj_and_place` | 24 | 12 | 13 | **17** |

Global SkillX 在六个任务族上均不低于 NoSkill，最大增益出现在 `pick_cool_then_place`，从 4/24 提高到 16/24。

## 产物

- 原生全局库：`artifacts/skillx/alfworld-global-p310-recovered/library.json`
- 执行库：`artifacts/skillx/alfworld-global-p310-recovered/execution/library.json`
- 构建报告：`artifacts/skillx/alfworld-global-p310-recovered/report.json`
- 执行索引报告：`artifacts/skillx/alfworld-global-p310-recovered/execution/report.json`
- Flash run：`artifacts/runs/alfworld-dev-v1-flash-skillx-global-p310-r1`

## Valid-unseen 复现（Flash）

为排除 dev 单次结果偶然性，冻结同一全局执行库，在 AgentBench `valid_unseen` 的
134 个样本上做一次 `repeat_id=0` 验证。SkillX 首轮有 3 条 ALFWorld server
断连，使用同一 run ID 的 checkpoint 续跑后全部补齐；`errors.jsonl` 中保留的是
历史失败尝试，不计入最终结果。

| 条件 | 成功数 | 成功率 | 平均步数 | 平均输入 token |
|---|---:|---:|---:|---:|
| NoSkill | 71/134 | 52.99% | 14.84 | 45,677 |
| Manual event policy | 102/134 | 76.12% | 11.49 | 38,889 |
| Global SkillX | **109/134** | **81.34%** | 11.69 | 64,651 |

Global SkillX 相对 NoSkill 提高 **28.36 pp**，相对手写事件策略提高 **5.22 pp**。
因此 dev 上的 `113/139` 不是一次偶然命中；在未见任务集上仍保持高于 NoSkill 和
手写策略的优势。该结果也进一步说明，当前 Trace2Tower ALFWorld 的主要差距不应
归因于“外部 SkillX 只是任务族特化”或 API 偶然性。

测试运行：
`artifacts/runs/alfworld-test-v1-flash-skillx-global-p310-r0/`。

## v9 失败集逐样本审计

这里的“全局”要准确表述为：执行库是跨任务全局构建的（224 个 plan、19 个
skill），但运行时不是把全库注入。`SkillXProvider.retrieve()` 先在全局 224 个
plan 中以 `task_goal` 检索 Top-3，只注入得分最高且过阈值的 1 个 plan；随后把该
plan 拆成步骤，再从全局 19 个 skill 中按步骤语义检索，最多保留 8 个去重 skill。
这整个 `1 plan + skills` 上下文在 episode 开始时注入一次，不是每步重新召回。

在 Trace2Tower 三信号 v9 的 9 个 Mid-injected failure 上，SkillX 成功 3/9。
关键 plan/skill 与结果如下：

| 样本 | SkillX 结果 | 命中的 plan（原任务） | 实际起作用/缺失的 skill | 诊断 |
|---|---:|---|---|---|
| `T20190906_215856_374127` | 成功（需审慎） | `put a chilled frying pan onto a counter top` | 搜索目标、取物、`chill held object with fridge`、放置 | 7 步找到 pan 并冷却；但环境在 `cool pan` 后直接 `reward=1, done=true`，没有执行任务文本要求的放置，属于评估终止条件与任务文本不一致 |
| `T20190907_233056_022802` | 失败 | `Put a warm mug in the coffee maker.` | 搜索/取物/冷却 skill；缺少 clean skill | plan 把 clean mug 错配成 warm mug，检索到错误变换技能，20 步仍在搜索 |
| `T20190908_233922_171295` | 成功 | `Place a rinsed fork in a drawer.` | 搜索、取物、`clean held item at sinkbasin`、放置 | 目的地 drawer 与当前 counter 不同，但通用四阶段计划仍完成 10 步任务 |
| `T20190909_004531_429065` | 失败 | `put chilled plate on the table` | 冷却与多个放置 skill | 计划目标 plate/table 与实际 pot/diningtable 偏离，搜索顺序消耗全部步数 |
| `T20190909_044933_815840` | 失败 | `Put a chilled apple on the counter.` | 取物、冷却、放置 | 计划结构正确，但通用搜索未在 20 步内到达 apple 的真实位置 |
| `T20190909_113844_191747` | 失败 | `To rinse of a spoon and put it on the table.` | 搜索 spoon、clean-at-sink、放置 | 环境规范对象是 `ladle`，SkillX 按噪声 goal 选择并清洗 `spoon`，没有突破噪声 |
| `T20190909_121908_219603` | 成功 | `Put the washed ladle on the gray pot on the table.` | 搜索 ladle、clean-at-sink、放置 | plan 的灰锅/桌面细节与当前 counter 不同，但规范对象 ladle 一致，通用结构成功泛化 |
| `T20190909_183724_205399` | 失败 | `To rinse of a spoon and put it on the table.` | 同上 | 与上一条同一 plan/skill 组合；模型清洗并放置 spoon，环境要求 ladle，之后循环检查 |
| `T20190911_131350_027076` | 失败 | `To rinse of a spoon and put it on the table.` | 同上 | 同一 plan/skill 组合；模型完成 spoon 的清洗放置，但 canonical 目标为 ladle |

### 对“突破噪声”的结论

SkillX 的全局 plan 确实突破了部分**任务细节噪声**：成功的 pan、fork、ladle
样本中，plan 的原任务在目的地或表述上与当前游戏不完全一致，但 plan 抽取的是
“搜索目标 -> 取物 -> 变换 -> 放置”的跨任务结构，且共享 skill 明确要求从当前
observation 读取精确对象实例。这解释了它在普通泛任务样本上的强优势。

但它没有突破**目标对象词汇冲突**。三个 `spoon`/`ladle` 样本全部命中同一条
spoon plan 和同一组 clean/search skill，并实际输出 `take/clean/move spoon`；
SkillX 的“exact object names from observation”约束反而强化了错误的 spoon 绑定，
因为 provider 没有把环境 canonical task 与第一条人工 `goal_text` 做冲突消解。

更强的配对证据是：成功的规范 ladle 样本 `T20190909_121908_219603` 与三个
spoon/ladle 噪声样本注入的 8 个 skill ID **完全相同**，都包含搜索定位、精确
取物、导航和三张清洗 skill。成功样本命中的 plan 写 `ladle`，失败样本命中的
plan 写 `spoon`。因此这里的成败差异来自 plan 对目标对象的绑定，而不是 skill
召回覆盖率；共享 skill 本身没有解决词汇冲突。

其中 pan 样本不能作为完整端到端突破计入：从动作链看，9 条中只有 fork 和规范
ladle 两条完成了取物、变换和放置；pan 是环境提前终止造成的表面成功。

因此更准确的定位是：

1. SkillX plan 不是固定全局卡，而是全局库中的 Top-1 任务 plan；其强点是一次性
   暴露完整端到端阶段结构，避免 Trace2Tower 低信息 Mid 只描述局部动作。
2. 它对“任务描述细节不同但对象词一致”的泛化有效，9 条难例中有 2 条完整救回；
   第 3 条是环境提前终止，不能作为完整链路证据。
3. 它对 `spoon`/`ladle` 这类对象词冲突无能为力，9 条中最明确的 3 条噪声样本
   全部失败；因此不能说 SkillX 已解决数据集噪声。
4. Trace2Tower 若要利用全局 High 的泛化优势，重点不是复制 SkillX 的检索器，
   而是让全局 High 在端到端对比卡中表达阶段结构，同时把环境 observation 的
   canonical 可执行对象名置于自然语言 goal 之上。这样能保留图算法创新，又避免
   被第一条 noisy `goal_text` 绑定。

## SkillX Plan 与 v9 High 的信息差异

逐字对照后，SkillX plan 并不比 v9 High 包含更多全局策略。v9 的 clean/cool
High 已经覆盖“识别目标、搜索、取物、清洗/冷却、前往目的地、必要时打开、最终
放置”，并额外包含错误对象替换、重复扫描、前置条件修复和完成检查。SkillX plan
真正增加的是实例级条件，而不是更完整的阶段结构。

| 信息维度 | Trace2Tower v9 High | SkillX 命中 plan | 实际影响 |
|---|---|---|---|
| 对象绑定 | 使用 `target object/item` 占位，并要求与 goal 一致 | 直接写 `pan`、`fork`、`ladle`、`spoon` | 对象词正确时增强控制；goal 有噪声时放大错误绑定 |
| 搜索位置先验 | 只写“likely surfaces/containers”，不说明哪些位置优先 | 保留相似训练任务中的 countertop、table、drawer、appliance 等来源/目的地类别 | 成功样本优先搜索 countertop/table；v9 High 常从 cabinet 1 顺序扫描 |
| 阶段结构 | 5-7 个端到端步骤，包含恢复和完成条件 | 3-4 个压缩步骤 | 两者结构等价；SkillX 更短、更聚焦，但不是新增信息 |
| 动作落地 | High 只描述语义动作，依赖模型自己绑定对象实例 | plan 标注 `take_action`，配套 skill 提供精确字符串、参数和输出变量 | plan 本身增益有限，真正的可执行细节来自后续 skill |
| 状态数据流 | 用自然语言约束“保持同一目标”“完成后停止” | skill 显式维护 `found_object`、`found_location`、`item_cleaned`、`object_chilled` 等状态 | 减少发现目标后继续搜索或变换后丢失目标的概率 |
| 恢复规则 | 明确处理错误对象、重复扫描、缺前置条件和提前放置 | plan 恢复信息较少 | 这一项反而是 Trace2Tower High 更强 |

因此，SkillX 成功难例的主要新增信息可以归纳为：

1. **对象条件化的搜索先验**：不仅说“去可能的位置”，还通过相似任务 plan 暗示
   先查 countertop/table/drawer/appliance 中的哪一类。
2. **具体对象词和目的地词**：把抽象 `target item` 绑定为 pan/fork/ladle，降低
   普通样本的实体选择歧义，但也会在 spoon/ladle 噪声中产生反效果。
3. **显式执行数据流**：配套 skill 要求记录发现位置和精确实例名，找到后停止，
   再把同一实例传给取物、变换和放置阶段。
4. **每阶段的精确动作契约**：skill 给出 `go to`、`take ... from ...`、
   `clean/cool ... with ...`、`move ... to ...` 的构造和成功检查。

这意味着下一版 High 不需要增加更多泛化步骤，而应补充两类当前缺失的信息：
基于成功/失败图归纳的**对象条件化搜索优先级**，以及从环境 observation 绑定
canonical 对象实例并贯穿全链路的**显式状态槽位**。恢复规则和完成检查应继续
保留，它们已经是 v9 High 相对 SkillX plan 的信息优势。
