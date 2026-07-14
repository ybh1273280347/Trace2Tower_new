# WebShop Event Tower V2 实验口径

## 1. 有效范围

本协议取代 2026-07-14 之前的 WebShop Flat、Static Tower、Random-300、旧规模实验和旧消融口径。历史结果不进入新主表，也不参与配置选择。

当前只承认八个方法身份：

| 方法 | 角色 | 训练证据 | 执行口径 |
|---|---|---|---|
| `no_skill` | 零信息 baseline | 无 | 不注入技能 |
| `manual_skill` | 精心设置的强 baseline | 固定人工知识 | 固定全文直接注入 |
| `global_e2e_gpt` | 轨迹全局归纳 baseline | 满分成功轨迹 | GPT-5.4 读取完整 corpus，归纳 3-6 张独立端到端卡，运行时 top-1 |
| `skillx` | 相似方法 baseline | 满分成功轨迹 | 保留 SkillX 原生构建与检索，最多注入 8 个技能 |
| `trace2tower` | 完整方法 | success-prioritized mixed pool | Event-aware signed Mid + contrastive High |
| `trace2tower_semantic_only` | Mid 构建消融 | 与完整方法相同 mixed pool | 关闭事件分层和图关系，Mid 退化为 segment embedding 语义聚类 |
| `trace2tower_mid_only` | High 消融 | 复用完整 Tower | 显式 `include_high=false`，只检索 Mid |
| `trace2tower_no_mixed` | 负证据消融 | 仅满分成功轨迹 | 保留事件分层与层级结构，移除失败/部分成功证据 |

不保留旧方法名别名，也不提供旧 artifact 的兼容执行路径。

## 2. Trace2Tower 到底是什么

Trace2Tower 不是“把整条轨迹聚类后再让 GPT 总结”。它是一条从事件、关系图到层级技能的构建链：

1. 将轨迹转成逐步 transition，并按 WebShop 页面状态和动作语义抽取事件类型；连续同类事件合并为 event segment。
2. 以 segment 为节点，并且只在同一事件类型内建图。语义相似度描述“做的事情是否相近”，transition、outcome 和对比证据描述“这些行为在执行顺序和结果上是否应当靠近或排斥”。
3. 对 signed graph 做谱聚类，得到 Mid cluster；GPT 只负责把每个有证据边界的 cluster 渲染成可执行 Mid skill。
4. 把每条轨迹映射为 Mid 序列，在成功与失败/部分成功之间做路径对比，诱导跨阶段的 High skill。High 表达可复用的完整策略顺序，不是另一次语义聚类。
5. 在线检索固定 High Top-1，同时检索若干直接 Mid，去重后注入上下文。直接 Mid cap 由 validation 在 `3/5/8` 中冻结。

因此，“聚类”只是完整算法中构建 Mid 的一步。完整 Tower 额外使用事件边界、带正负关系的图、结果证据、跨 Mid 的路径诱导以及 High/Mid 联合检索。

`trace2tower_semantic_only` 专门隔离这个区别：它对同一批 segment 仅按 embedding 做语义聚类，不使用事件分层、transition edge、outcome edge 或 contrastive graph。为了只检验 Mid 构建机制，它仍从所得 Mid 序列诱导 High。`global_e2e_gpt` 则完全没有 segment、图、Mid 或 High 中间结构，直接从整个轨迹 corpus 归纳端到端卡。

## 3. 数据边界

- 训练池 P50：50 个 train tasks，4 repeats，共 200 条 No-Skill 轨迹。
- 训练池 P100：100 个 train tasks，4 repeats，共 400 条 No-Skill 轨迹；P50 必须严格包含于 P100。
- Validation/Test 候选仅为 WebShop indices `0..999`，不按历史是否见过过滤。
- 使用 `random.Random(20260716).sample` 无放回抽取 200 个任务：前 100 个为 validation，后 100 个为 test，两者零重叠。
- 每个条件固定 repeat IDs `0/1/2`，即每个模型、方法、split 条件 300 episodes。
- Validation 只能选择 Tower 直接 Mid cap；test 不再调参。

样本清单和 selection hash 以 `configs/experiments/webshop_event_tower_v2.json` 为准。

## 4. 模型与执行

- Agent endpoints：`deepseek-v4-flash`、`deepseek-v4-pro`。
- GPT 构建/渲染：`gpt-5.4`，总并发不超过 4。
- Agent 温度 0，最多 20 步；两模型共享任务、repeat、方法 artifact 和执行配置。
- SkillX 固定原生 `max_skills=8`，不参加 cap sweep。
- Global E2E 固定 top-1；NoSkill 和 Manual 不存在 cap。
- Tower High 固定 Top-1。`cap3/cap5/cap8` 只指直接 Mid 检索预算，属于运行时配置，不进入 Tower 快照内容身份。

## 5. 指标与统计

主指标为 WebShop mean reward。满分率 `primary_score >= 0.999`、completion、步骤、无效动作率和 agent chat tokens 为次指标。

统计单位是 task：先在每个 task 内平均三个 repeats，再做配对 task bootstrap。方法比较必须使用完全相同的 task/repeat keys。报告均值差、95% 区间、胜/平/负和完整覆盖率，不把 episode 当作 300 个独立样本。

Cap 选择只使用 P50 Full Tower validation：对每个 task 先平均 repeats 和两个 agent endpoints，得到三档 cap 的配对 reward。先找经验均值最高的 cap，再选择与它的 95% 配对 task-bootstrap 差值区间包含 0 的最小 cap。该规则偏向更低上下文成本，并在看 validation 结果前固定。

Flash 门控比较 Manual、Global E2E、SkillX、Full Tower 各自相对 Flash NoSkill 的 reward 提升，四个比较使用 Holm family-wise `alpha=0.05`。若没有任何正向差异通过校正，Flash 不再运行消融和 P100 规模实验；这只是计算资源门控，不删除阶段四已经得到的 Flash 主结果。

## 6. Artifact 契约

P50/P100 原始训练轨迹保留。Manual 文本固定为本目录的 `manual-skill.md`。Global E2E 和 SkillX 只有在训练轨迹集合、prompt、构建配置和内容哈希完全匹配时才能复用。

所有 v2 Tower 都必须重新构建。WebShop `trace2tower` 和 `trace2tower_no_mixed` 快照若未开启 `event_type_stratification=true`，代码直接拒绝构建或加载为正式方法。`trace2tower_semantic_only` 必须同时关闭事件分层、transition、outcome 和 contrastive graph。Mid-only 不产生新快照，复用 Full snapshot 并只改变运行时 `include_high`。

每个阶段冻结输入轨迹哈希、配置哈希、prompt 哈希、artifact ID、artifact SHA-256、模型 endpoint、run IDs 和结果哈希，然后独立提交并推送。
