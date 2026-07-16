# WebShop v8 与旧图 Skill 注入审计

## 结论

v8 的主要失败不是条件式 High 本身无效，而是 task-specific High 在覆盖不足时被强制
注入。旧图卡片虽然来自错误的流程聚类，却因内容泛化、实体中立和篇幅较短，对错误
召回具有更高容错性。

Test-A repeat0 上：

| 方法 | reward | 完全成功率 | 步数 | 无效动作 | 输入 token |
|---|---:|---:|---:|---:|---:|
| 旧 T1 graph-cap3 | 0.71925 | 54% | 6.81 | 0.16 | 20,059 |
| decision-state v8 | 0.70067 | 50% | 8.18 | 0.38 | 31,435 |

逐任务比较为 v8 8 胜、14 负、78 平，平均 reward 差 -0.01858。

## 强制错绑机制

v8 只有 29 张 High，每张 High 的 `retrieval_condition` 都是一个具体训练任务，例如
`fragrance free lotion 18 fl oz`、`French Cellar area rug` 或 `jalapeno jerky`。
runtime 配置却使用：

```yaml
high_top_k: 1
high_similarity_threshold: -1.0
```

因此每个测试任务都会得到一张 High，不存在“无可信匹配则不注入”的路径。

`retrieve_tower_graph` 的 High 分数为：

```text
0.35 * goal_similarity
+ 0.35 * active_child_mid_state_similarity
+ 0.20 * event_compatibility
+ 0.10 * path_quality
```

WebShop 任务共享查询、候选、检查、选项和购买事件，v8 Mid 又普遍包含多个事件类型，
因此 event compatibility 无法区分商品类别。错误 High 选中后，其 compatible child
Mid 会优先占用最多两个 Mid 预算；这一步不检查 `direct_mid_similarity_threshold`。
剩余 direct Mid 再按当前页面文本检索，容易因 `cookie/chocolate`、`size/color` 等
表面词重合召回另一个具体任务卡。

## 具体失败证据

旧图优于 v8 的最大回退样本均出现了具体实体错绑：

| 样本 | 真实目标 | v8 注入的 High | 旧图结果 | v8 结果 |
|---|---|---|---:|---:|
| webshop:981 | Levi straight-leg big-and-tall jeans | Ballard Blue slim-fit dress shirt | 1.00 | 0.00 |
| webshop:405 | 15 ml Miss Dior travel perfume | fragrance-free 18 fl oz lotion | 0.75 | 0.00 |
| webshop:977 | sea-teal machine-washable curtains | French Cellar area rug | 1.00 | 0.50 |
| webshop:7 | passion-fruit smoothie mix | low-fat jalapeno jerky | 0.75 | 0.25 |
| webshop:899 | 5x7 photography backdrop | 10-foot easy-clean rug | 1.00 | 0.75 |
| webshop:261 | gray heavy-duty spa chairs | gray women's sandals | 1.00 | 0.75 |
| webshop:174 | pink/blue iPhone XS Max case | noise-cancelling earbuds | 0.50 | 0.25 |
| webshop:922 | 8 oz freeze-dried tomatoes | jalapeno jerky | 0.0667 | 0.00 |

另一个可完整复查的样本是 `webshop:873`。真实目标是低于 $40、独立包装、写有
congratulations 的巧克力；v8 开局注入了 `family birthday cupcake topper` High，
进入商品页后又注入 `Princess Cake and Cookie 128 fl oz BPA-free` Mid 和特定香水
搜索 Mid。agent 虽然搜索了正确的巧克力词，但最终购买的 Oreo 只得到 0.0333。

## 为什么旧错误图反而更强

旧图只有 6 张 High 和 12 张 Mid。High 没有 task-specific retrieval condition，主要
内容是：

- Search, open, and configure the chosen product；
- Search and buy a direct match；
- Verify details, then purchase from the product page。

旧 Mid 同样以 search、option selection、detail inspection 和 backtracking 为主。
这些卡片缺少商品条件化，所以不能证明旧图实现正确；但即使召回不精确，也不会要求
agent 把 jeans 当 shirt、把 curtains 当 rug。它实际形成了一组低风险的操作 scaffold。

v8 则把 task-specific 内容写入每张卡。Mid 平均检索文本约 1,585 字符，旧 Mid 约
973 字符；v8 Test-A 平均输入比旧图多 11,375 token。更具体、更长的错误指导比泛化
操作提示更容易争夺模型控制权。

## 联网资料的共同启发

近期 WebShop 工作对这一问题给出了一致警告：

- [CLEAR](https://arxiv.org/abs/2604.07487) 指出，直接复用过去任务的 retrieved
  context 会把适配负担交给执行模型；它改为根据当前任务生成 task-specific context。
- [UCOB](https://arxiv.org/abs/2606.29502) 明确把 skill 视为可能误导的非 oracle
  上下文，并用同任务、同状态下 skill/no-skill 的 return-to-go 决定局部教师。
- [D2Skill](https://arxiv.org/abs/2603.28716) 使用配对的 baseline/skill-injected
  rollout 学习 skill utility，并进行 utility-aware retrieval 和 pruning。
- [OPID](https://arxiv.org/abs/2606.26790) 只在检测到 critical decision 时优先使用
  step skill，否则退回 episode-level skill，而不是每一步固定注入局部卡。
- [SkillAdaptor](https://arxiv.org/abs/2606.01311) 将失败归因到第一个 actionable
  fault step，再对责任 skill 做有验收条件的定向更新。
- [GraSP](https://arxiv.org/abs/2604.17870) 强调 precondition/effect 和节点级验证；
  更多 skill 并不单调改善性能，错误编排本身会导致负迁移。

## 下一步建议

不要继续改 High 文本，也不要继续扩大具体 skill 库。优先修复选择和适配契约：

1. **High 可以 abstain。** 删除 `-1.0` 强制匹配；具体 High 只有在类别、身份和关键
   槽位兼容时才能注入。
2. **提供安全 fallback。** 无可信 task-specific High 时，注入一张从当前图全局归纳的
   实体中立条件策略，或不注入 High；不能选“最不差”的错误商品任务。
3. **child Mid 也要过语义/槽位门槛。** 不允许错误 High 的 child Mid 仅凭事件兼容
   自动占满预算。
4. **Mid 改为 critical-only。** 只有查询失败、候选冲突、选项缺失或购买门槛未满足
   时才注入对应决策卡；普通步骤不动态注入。
5. **检索结果先适配当前任务。** Top-K 图证据只作为 LLM 的参考，由一次任务级生成
   输出当前目标的条件策略，并允许返回 `NO_SKILL`。这比原样注入过去任务卡更符合
   CLEAR 的发现，也保留图算法作为证据选择核心。
6. **用配对 utility 管理 skill。** 对同一训练任务比较 skill/no-skill return；负效用
   卡降权或淘汰，而不是把所有成功轨迹渲染出的卡默认视为有益。

