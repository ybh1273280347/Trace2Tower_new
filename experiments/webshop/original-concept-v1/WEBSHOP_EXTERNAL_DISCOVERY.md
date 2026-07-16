# WebShop 外部资料审计与表示启发

## 资料范围

本审计优先使用 WebShop 官方论文、官方仓库和官方环境代码，并将近期公开的
WebShop agent 工作作为设计参考，不把后者的结果当作本项目实验结果。

- [WebShop 论文](https://arxiv.org/abs/2207.01206)
- [WebShop 官方仓库](https://github.com/princeton-nlp/WebShop)
- [官方 goal 与 reward 实现](https://github.com/princeton-nlp/WebShop/blob/master/web_agent_site/engine/goal.py)
- [官方 baseline 说明](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/README.md)
- [OPID：episode-level 与关键步骤 skill](https://arxiv.org/abs/2606.26790)
- [CONVOLVE：从轨迹归纳 stopping rules](https://arxiv.org/abs/2606.28733)
- [Progress/feasibility 双记忆框架](https://arxiv.org/abs/2604.02734)

## 官方契约

官方 WebShop 将任务定义为商品身份、类别、属性、可选项和价格上限的组合约束，
而不是一段没有结构的自然语言目标。官方 reward 也分别计算商品类型、属性、选项
和价格，再组合成最终分数。论文同时将 query reformulation、网页噪声和 strategic
exploration 列为主要难点。

官方 baseline 还明确分离了两个决策：search imitation model 生成搜索查询，choice
model 在当前页面的合法动作中选择商品、属性选项或购买动作，并提供直接使用用户
instruction 作为搜索词的消融。这说明 WebShop 的关键能力不是记住一个固定的
`search -> click -> buy` 流程，而是：

1. 形成或改写能够暴露候选的查询；
2. 在候选集合中进行商品类别和属性判别；
3. 在商品页绑定精确选项；
4. 在证据充分时停止购买，冲突时回退。

## 对当前 Tower 的诊断

当前两种失败表示分别有相反的问题：

- 旧 event-context 表示把 WebShop 的大量片段压成同构的搜索、点击和检查流程，
  没有表达候选商品已经满足了哪些目标约束；
- v7 product-entity 表示把完整商品标题、价格、选项和结果集合直接作为 embedding
  核心，造成实体碎片化，破坏了 High 的跨样本路径复用。

因此下一版不采用“纯流程”或“完整商品文本”二选一，而采用双层状态表示：

```text
Goal slots:
  category / identity / hard attributes / required options / price ceiling

Candidate state:
  candidate identity
  observed fields
  slot status: supported / contradicted / uncertain
  next decision: open / reject / refine / select / buy
```

图算法仍然只使用事件片段、时间转移和成败一致性；槽位状态属于事件抽取之后的
领域表示，不改变 Trace2Tower 的通用图契约。

## 对 skill 层级的启发

High 应是一条端到端商品决策策略，覆盖查询、候选判别、选项绑定、回退和购买停止。
Mid 不应再是泛化的页面操作流程，而应是少量关键决策单元：查询重写、类别冲突拒绝、
属性验证、选项绑定、候选回退和停止门控。

这一点与近期 OPID 的 episode-level global skill 加关键步骤 local skill、以及
CONVOLVE 从完整交互中归纳 stopping rules 的方向一致。它们支持“全局指导 + 关键
决策路由”，但不证明任何具体 WebShop skill 在本项目上有效。

## 立即尝试的表示

第一版原型使用 `decision_state` 签名：

- 保留事件类型和前后事件，保证关系图仍然可复用；
- 从目标和页面中抽取紧凑的类别、价格、选项、已选状态和候选数量；
- 不把结果页的完整商品列表直接写入 embedding；
- 为每个事件附加“槽位变化”和“下一决策角色”；
- 候选 ID 只用于同一轨迹内绑定，不作为语义相似度的主要内容。

该版本先进行离线签名、聚类和 skill 内容审计，再决定是否 rollout；不与 v6/v7
目录混用。

## v8 初步实现结果

`decision_state` 已在 P200 的 800 条训练轨迹上完成实现。5,419 个 segment 折叠为
1,816 个图节点，得到 90 个 Mid 和 29 个 High；跨事件边为 8,164 条，连通分量为
3。它没有复现 v7 的 High 数量坍缩。

WebShop High renderer 已改为条件策略：允许显式输出 `IF condition -> action` 和
`OTHERWISE -> recovery`。29/29 条 High 都包含至少一个条件分支，不再把查询失败、
候选冲突和属性不确定性压成单条线性流程。

hierarchical runtime 使用 `High top-1 + Mid top-2 + Low 0`。Test-A repeat0 结果为：

| 方法 | 平均奖励 | 完全成功率 | 平均步数 | 平均无效动作 | 平均输入 token |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.68075 | 51% | 7.98 | 0.36 | 21,388 |
| decision-state v8 | 0.70067 | 50% | 8.18 | 0.38 | 31,435 |

v8 相对 NoSkill 的配对 reward 差为 +0.01992，95% 区间
[-0.04009, +0.08133]；14 胜、13 负、73 平。它救回 8 个 NoSkill 零分样本，同时
新增 4 个零分。当前结论是：条件式 High 修复了线性渲染坍缩，并产生正向奖励点估计，
但没有提升完全成功率且区间跨零，暂时只能作为有希望的表示证据。

在同一 v8 snapshot 上进行 validation repeat0 后，平均奖励为 0.68725，NoSkill 为
0.65235，配对差为 +0.03490，95% 区间 [-0.03190, +0.10717]；完全成功率均为
47%。因此 Test-A 和 validation 的奖励点估计方向一致，但两个区间都跨零，且 v8
平均步数和无效动作略高。当前应称为 split-sensitive 的正向奖励信号，不能称为稳定
泛化优势。

后续实际注入审计见 `V8_SKILL_INJECTION_AUDIT.md`。核心问题是 v8 将单个训练任务
渲染为具体 High，却以 `high_similarity_threshold=-1.0` 强制给所有测试任务分配一张
High；错误 High 的 child Mid 又会绕过 direct Mid 门槛优先注入。旧图卡片虽然来自
错误的流程聚类，但内容实体中立，错误召回的危害更低。这解释了具体 v8 skill 为何
反而弱于旧图 skill。
