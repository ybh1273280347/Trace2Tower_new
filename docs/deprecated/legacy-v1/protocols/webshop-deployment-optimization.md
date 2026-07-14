# WebShop 部署优化实验

## 研究问题

本实验验证 Pareto-based skill lifecycle management 是否能在不降低官方 reward 的前提下减少步骤和 agent chat token，并重点回答：mixed Tower 中由错误和部分成功轨迹带来的覆盖，能否在部署反馈抑制弱技能后转化为净收益。

“mixed 优化后反超 success-only”是待检验假设，不是选择规则。若未反超，仍完整报告无效或负结果。

## 固定比较对象

正式 deployment feedback 使用 `deepseek-v4-pro`。此前 WebShop 正式实验表明固定技能对 Pro 有显著正向价值，而 Flash 上边际价值较弱；Flash exposure 不进入本实验决策。

三组训练期 rollout 使用同一批 50 个 WebShop train 任务、`repeat_id = 0, 1, 2`、20 步上限和 temperature 0：

1. Pro NoSkill；
2. Pro success-only Tower cap3；
3. Pro mixed Tower cap3。

每组应有 150 个正式 episode。结果按 `benchmark + sample_id + repeat_id` 严格配对。训练任务来自原 50-task 构建池，但 deployment rollout 是重新调用 Pro 得到的新轨迹。

## 成本口径

成本只统计 agent chat：

```text
chat_tokens = chat_input_tokens + chat_output_tokens
```

检索 embedding token 不进入成本目标。技能上下文导致的 prompt 增长会自然体现在每轮 chat input 中。历史 aggregate input token 可能包含 embedding，因此不用于反推本实验成本。

## Pareto 规则

每个实际注入技能计算四个越大越好的目标：

```text
performance_level
paired_reward_gain
guarded_step_saving
guarded_chat_token_saving
```

当技能 episode 的 reward 低于配对 NoSkill 时，正向步骤或 token 节省截断为 0，负向成本仍保留。Mid 与 High 在各自 scope 内独立进行非支配排序。

生命周期规则在 exposure 结果完成前固定：

- 至少 10 个配对 exposure 才有资格 downweight；
- F1 技能受保护，只允许 downweight `pareto_front_rank > 1` 的技能；
- 第一轮只改变 `active -> downweighted`，不删除卡片；
- 原始 cosine 保持不变；downweighted 技能必须比 active 技能高出超过 `0.01` 才能排在前面；
- success-only 与 mixed 共用完全相同的门槛和 epsilon。

## 实验阶段

### 1. Exposure 与审计

分别为 success-only 和 mixed 生成 refinement report。报告绑定 Tower snapshot、agent model、run ID、result hash、150 个配对 key、技能 exposure、四维向量、Pareto front 和 downweight 决策。

任何缺失 key、重复 key、非 train episode、未知 skill ID 或缺失 chat token 都会阻断排名。

### 2. 优化后验证

在既有验证任务 `webshop:50-149` 上，以 Pro、3 repeats 比较：

- success-only v0 / v1；
- mixed v0 / v1；
- Pro NoSkill 作为共同参照。

主要指标是 mean reward 与满分率；效率指标是平均步骤和 agent chat token。报告同任务配对、task-cluster bootstrap 95% CI。验证集只判断 lifecycle 是否值得冻结，不作为最终泛化证据。

### 3. 新 held-out 测试

原 Random-300 已经用于旧方法结论，不能为新 refinement 提供未见测试。优化规则冻结后，从从未出现在历史 result/error 的 WebShop 任务中预注册新的 selection，记录 manifest 与 SHA-256，再一次性运行 success-only v0/v1、mixed v0/v1 和 NoSkill。

只有新 held-out 结果用于判断部署优化是否泛化，以及 optimized mixed 是否真正超过 optimized success-only。

## 当前进度

- Chat-only 成本字段、Pareto 排名、最小 exposure、F1 保护和 `0.01` 状态带已经实现并通过针对性测试。
- Flash deployment pilot 已停止，不进入正式优化。
- Pro exposure 正在按检查点执行。
- Split/Merge/Promote 的 Pareto 排序已有实现；完整候选塔重建与 Tower v1 结构更新仍需接通。当前 lifecycle v1 首先验证可逆 Downweight，不将未接通的结构更新描述为已完成。
