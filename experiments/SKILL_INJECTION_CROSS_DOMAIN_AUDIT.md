# SkillX 与 Trace2Tower 实际注入内容跨域审计

## 核心结论

跨域差异不是“ALFWorld 有任务族，所以通用 skill 不成立”，也不是事件抽取本身
过于领域化。真正的问题是两种方法在两个数据集上保留下来的信息层级不同：

| 数据集 | SkillX 实际保留的信息 | Trace2Tower 实际保留的信息 | 更有增量的一方 |
|---|---|---|---|
| ALFWorld | 目标级操作 plan、精确 action schema、前置条件、循环与状态变换 | 4 个宽泛混合 Mid；High 主要覆盖 clean/toggle | SkillX |
| WebShop | 39 类近似同构购物 plan、2 张通用工具说明 | 页面事件阶段、选项/详情/回退/购买路径、每步动态检索 | Trace2Tower |

ALFWorld 上，修复版 Tower 为 85/139（61.15%），与 No-Skill 持平，但显著低于
Global SkillX 的 113/139（81.29%）。SkillX 相对 Tower 有 32 个正向翻转、4 个
反向翻转。WebShop Test-A repeat3 上，最终 Tower 奖励为 0.71119，SkillX 为
0.70627；完全成功率分别为 53.67% 和 49.33%。

因此，不是某个方法在一个领域“天然有效”、换领域就神秘失效，而是当前
Trace2Tower 的 ALFWorld 表示把最有用的任务程序压掉了；SkillX 在 WebShop 则把
大量构建成本花在了高度同构的购物 plan 和两张工具说明上。

## ALFWorld 实际注入

### SkillX

全局 SkillX 库包含 224 个 plan 和 19 个 functional skill。139 个验证 episode
全部召回一个 plan，共使用 84 个不同 plan；120/139 个 episode 同时注入满额 8
个 skill，其余注入 5 至 7 个。平均一次性上下文为 13,268 字符。

SkillX 注入内容不是抽象口号。以“把冷却后的酒瓶放到桌上”为例，实际召回 plan
明确给出：

1. 去桌面找到并取得目标物体。
2. 去冰箱，必要时先打开。
3. 使用冰箱执行冷却。
4. 返回桌面并放置冷却后的物体。

同时注入的 functional skill 包含 `chill held object with fridge`、
`cool held item with appliance`、`take object from known receptacle` 和
`place carried object into openable destination`。正文直接给出
`go to <receptacle>`、`open <receptacle>`、`cool <object> with <fridge>`、
`move <object> to <destination>` 等 action schema，以及“必须已持有物体”“仅在
closed 时 open”“从观察中复制精确编号”等前置条件。

### Trace2Tower

当前官方事件 Tower 只有 4 个 Mid：

| Mid | 正文功能 | 主要事件组成 |
|---|---|---|
| `mid_0000` | 打开容器、取物、放置 | Pickup/Open/Put |
| `mid_0001` | 前往目标并检查 | Goto 84.6% |
| `mid_0002` | 检查表面或使用灯 | Scan/Toggle/Goto |
| `mid_0003` | 在 sink 清洁并关闭容器 | Clean/Close/Scan |

12 条自然 High 只形成了可信 clean 和 toggle 路径；cool、heat、普通搬运和双物体
任务没有匹配 High，只能回退到 Mid。运行时 cap3 面对总共 4 个 Mid，等价于每步
注入 75% 的库。

更严重的是 Mid 使用混合事件质量超过 0.1 即可进入候选。`mid_0003` 虽以 Clean
为主，但 Close/Scan 质量足以让它通过非 clean 任务的门控。实际 episode 级覆盖为：

| 非 clean 任务族 | 样本数 | 被注入 clean Mid |
|---|---:|---:|
| `pick_and_place` | 35 | 30 |
| `pick_cool_then_place` | 24 | 24 |
| `pick_heat_then_place` | 16 | 16 |
| `pick_two_obj_and_place` | 24 | 23 |
| 合计 | 99 | 93 |

High 的排他事件错配已经从 115/139 修到 0，但 Mid 仍存在 93/99 的跨事件污染。
这解释了为什么修复版不再低于 No-Skill，却仍无法接近 SkillX。

## ALFWorld 配对结果

| 任务族 | N | No-Skill | SkillX | 官方事件 Tower |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 13 | 12 | **13** | 11 |
| `pick_and_place` | 35 | 33 | **34** | 33 |
| `pick_clean_then_place` | 27 | 13 | **20** | 14 |
| `pick_cool_then_place` | 24 | 4 | **16** | 5 |
| `pick_heat_then_place` | 16 | 11 | **13** | 10 |
| `pick_two_obj_and_place` | 24 | 12 | **17** | 12 |
| 总计 | 139 | 85 | **113** | 85 |

SkillX 相对 No-Skill 的 32 个正向翻转分布在全部六个任务族，其中 cool 任务占
12 个、clean 占 8 个、双物体占 5 个。它不是只在一个任务族上取巧。

### 失败样本 1：冷却后放置

目标：`Set a chilled bottle of wine on the table.`

SkillX 注入了结构完全匹配的“取物→冰箱→cool→返回桌面→放置”plan，agent 实际
执行 7 步成功。Tower 前五步反复注入 `mid_0000/mid_0001/mid_0002/mid_0003`
的不同排列，其中包含错误 clean 指导；agent 用 20 步搜索并最终只把另一个瓶子
放进冰箱，未完成冷却和目标放置。

### 失败样本 2：清洁后放置

目标：`Place a rinsed fork on a counter.`

SkillX plan 明确包含“找到 fork→取物→sink 清洁→counter 放置”，10 步成功。
Tower 虽召回 clean High，但 High 从“先去 sink”开始，缺少先定位和取得目标物体的
可靠前置阶段；agent 先去空 sink，随后遍历容器，20 步耗尽。

### 失败样本 3：双物体循环

目标：`Put two books on a desk.`

SkillX plan 明确要求重复 fetch-and-place，agent 最终完成两个对象。Tower 没有
计数、循环或“交付后返回取第二个”的 High，仅注入导航、检查和单次取放常识；
agent 在第 20 步刚找到第二本书，未完成交付。

## SkillX 的有效部分是什么

实际注入表明，ALFWorld SkillX 的主要价值很可能来自目标级 Reference Plan，而不
完全来自 19 张 functional skill：

- 每个 episode 都召回一个 plan，84 个不同 plan 覆盖 139 个目标。
- plan 保留 clean/cool/heat/toggle、最终放置和双物体循环的顺序。
- 库中没有 heat functional skill；15/16 个 heat episode 甚至召回了 cool skill，
  但 SkillX 仍把 heat 成功数从 11 提高到 13，因为 plan 本身写出了 heat 步骤。

这是强证据，但不是因果证明。要确认贡献来源，仍需同一库上的 `plan-only`、
`skill-only` 和完整 SkillX 三条件消融。

## WebShop 实际注入

### SkillX

P100 WebShop SkillX 库只有 51 个 plan 和 2 张 skill：`search_action` 与
`click_action`。在 Test-A repeat1/2 的 200 个 episode 中，每次都注入这两张
工具说明，只召回 39 个不同 plan。两张说明主要重复 agent 已知的工具能力：构造
关键词、点击可见值、检查详情、选择选项、最后购买。

语义检索的 plan 还会带入无关商品属性。例如当前目标是“白色 engineered wood
twin bunk bed”，实际召回的是“黑色 king size eco-friendly bed frame”plan。
它们共享的有效内容仍只是搜索→打开→选项→验证→购买这一通用购物流程。

### Trace2Tower

WebShop Tower 的事件卡直接对应当前页面阶段：query formulation/refinement、结果页
候选选择、产品页选项、Attributes/Features/Description 检查、返回和购买。运行时
每步读取页面状态并重检索。

同一个 bunk-bed episode 的实际注入如下：

| 页面 | agent 动作 | Tower 提供的主要指导 |
|---|---|---|
| Search | 搜索 `twin size bunk bed white engineered wood` | 精炼查询、核心属性搜索 |
| Results | 打开候选 ASIN | 打开匹配结果、直接匹配购买路径 |
| Item | 选择 `white bunk bed` | 选择并核对所需选项 |
| Item | 选择 `bunk bed` | 继续完成选项配置 |
| Item | `Buy Now` | 配置完成后购买 |

200 个 repeat1/2 episode 的实际点估计为：

| 方法 | 平均 reward | 完全成功率 | 平均步数 | 平均注入字符 |
|---|---:|---:|---:|---:|
| P100 SkillX | 0.70328 | 49.5% | 7.105 | 6,679 |
| Final Tower graph cap3 | **0.70717** | **53.5%** | **6.845** | 24,018（跨步累计） |

WebShop Tower 也不是完美事件纯化：在 Search 页面仍可能提前带入 option card。但购物
任务最终都共享选项、验证和购买阶段，这类“未来阶段”提示通常只是冗余；ALFWorld
把 clean 提示注入 cool/heat 则是互斥操作，会直接改变执行目标。

## 为什么出现跨域反转

### 先纠正“错误 Tower”的对照口径

WebShop 上需要区分两种历史实现。纯 `Semantic-only` 只保留语义聚类，P50 Flash
奖励为 0.65025、Pro 为 0.54167，均低于对应 SkillX 的 0.70692 和 0.68500；
因此不能写成“纯语义聚类打败 SkillX”。与 SkillX 持平或略高的是
`Legacy Full Tower`：它已经包含关系 EigenTrace、Mid/High 分层表示，只是运行时
仍使用与结构不匹配的 cosine-only/cap8 检索。Flash Test-A repeat3 中 Legacy
Full 为 0.70958，SkillX 为 0.70627；Pro 中 Legacy Full 反而低于 SkillX
0.01994；Flash Test-B 则高 0.01892，三项奖励区间均跨零。

所以准确结论是：WebShop 上已经正确构建的 Tower 表示，即使部署检索仍有错误，
也与 SkillX 处于同一经验区间；不是早期退化为纯语义聚类的算法稳定胜过 SkillX。

### SkillX 提示词为何没有在 WebShop 复制 ALFWorld 增益

SkillX 的 plan 提示词对两个数据集完全相同：按 `user_task + 最短成功轨迹` 为每个
任务单独生成 plan，保留任务中的具体实体和成功动作顺序；随后 functional skill
提示词再把具体值参数化。这个机制在 ALFWorld 能保留“对象类型暗示来源位置、变换
设备和目的地”的实例先验，但在 WebShop 中没有同等增量：

1. WebShop P100 的 51 个 plan 近似同构，主要都是搜索、打开候选、验证属性、选择
   选项和购买。100 个 Test-A episode 只使用 39 种上下文。
2. 最终 skill 库只有 `search_action` 和 `click_action` 两张工具说明，平均每个
   episode 注入 `1 plan + 2 skills`。它们主要重复基础 agent 已知的工具 schema。
3. WebShop 的商品实体不携带稳定位置先验。ALFWorld 中 utensil/pan 常暗示
   countertop、table、drawer 或 appliance；WebShop 中“相似商品”不意味着当前
   查询应使用相同属性、候选或选项。任务级具体化反而会带入错误颜色、尺寸、材质
   或商品类别。
4. WebShop 的关键条件是当前页面状态和可见控件。一次性静态 plan 无法知道 agent
   正位于 Search、Results、Item、Attributes 还是返回页面；而 Tower 即使只用
   observation 语义，也会因这些页面文本差异自然形成近似状态机，并逐步刷新指导。
5. WebShop 的通用购物主程序本来就容易被基础模型掌握，SkillX 提供的信息增量小；
   ALFWorld 的隐藏物体搜索、持有状态、容器开闭、状态变换和多对象循环则需要更强
   的端到端程序提示。

因此跨域反转来自表示与领域结构的匹配：SkillX 的“单任务 plan + 参数化执行 skill”
适合 ALFWorld 的对象条件化长程程序；Tower 的“事件/页面状态 + 动态检索”适合
WebShop 的交互状态机。SkillX 不是没有能力，而是在 WebShop 上蒸馏出的内容与基础
模型已有能力高度重合，同时任务级具体属性还可能产生负迁移。

### 1. 任务程序熵不同

WebShop 的全体任务共享一个主程序，只改变商品约束；ALFWorld 至少有六种不同程序，
且 clean/cool/heat/toggle 与双物体循环不可互换。WebShop 的页面事件天然就是稳定
状态机，ALFWorld 的四个混合 Mid 不是。

### 2. 当前 Tower 在 ALFWorld 过度压缩

11 类官方事件被压成 4 个 Mid，导致 operator identity、前置条件和循环结构丢失。
WebShop P100 则形成 9 个 Mid、5 个 High，能够保留搜索、详情、选项和购买的阶段差异。

### 3. 奖励的信用分配强度不同

WebShop 有部分 reward。P100 mixed pool 中保留 161 条部分奖励轨迹和同任务对比，
负邻接可以定位低质量路径。ALFWorld 只有 episode 级 0/1 成功；同一失败轨迹中的
正确取物步骤和错误后续步骤被赋予相同结果标签，图上的 outcome 信号更粗。

### 4. ALFWorld 更依赖精确 grounding

ALFWorld 要求编号实体、容器开闭、持有状态、正确 appliance 和 exact command schema。
SkillX functional skill 显式编码这些条件。当前 Tower 卡片多是“去相关位置”“检查后
操作”的自然语言常识，对突破隐藏物体搜索、状态变换和双物体循环的信息增量有限。

### 5. 检索时机与信息类型匹配不同

SkillX 的静态目标 plan 很适合 ALFWorld 的长程程序；Tower 的动态状态检索很适合
WebShop 的页面状态机。当前 ALFWorld Tower 动态刷新的是四张过宽卡片，因而产生
上下文抖动和错误 Mid；WebShop 动态刷新则能随页面切换提供新信息。

## 不是哪些问题

- 不是因为 SkillX 使用了任务族运行时过滤：全局库和检索均不读取 `task_family`。
- 不是因为领域事件抽取“不通用”：两边都应先做各自领域事件抽取再进入同一建图步骤。
- 不是只要降低 High 阈值或扩训练池就能解决：当前缺的是正确 operator 结构和
  event-conditioned 信用分配，盲目增加路径会扩大噪声。
- 不是简单 renderer 文风问题：第二版 Tower 卡片已经合理，失败来自缺失程序和
  错误 Mid 投放，而不是语句不可读。

## 下一步最有信息量的实验

1. 在现有 ALFWorld SkillX 库上做 `plan-only / skill-only / full`，确认 20.14 pp
   收益到底由 plan 还是 functional skill 驱动。
2. 对 Tower 的 Mid 做排他事件门控或事件条件化渲染：混合簇可以保留，但注入正文
   必须与当前目标事件一致，先消除 93/99 的错误 clean Mid。
3. High 支持率仍保持 2%，但支持度分母按目标事件签名条件化，而不是按人工任务族
   分桶；这样 heat/cool 路径不会被全池异质任务稀释。
4. 为多对象目标在事件序列中保留计数和循环语义，避免压缩后只剩单次 fetch/place。
5. 完成上述结构修复后，再做一次同 cohort、repeat0 的 `graph Mid-only` 与
   `graph + High` 配对，不先扩池、不调阈值。

这些改进仍属于通用框架：领域适配器负责抽取事件和目标事件签名，通用 Tower
负责条件化建图、路径支持、层次生成和检索，不需要按数据集任务族维护多套 skill 库。
