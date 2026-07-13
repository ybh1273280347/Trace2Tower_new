# ALFWorld 正式实验协议

## 评估口径

ALFWorld 按 AgentBench 实现报告二值成功率。每个 episode 最多 20 次 agent turn；一次合法 `take_action` 等于一次环境动作。ALFWorld 配置中的 50 步属于其训练算法内部上限，不是 AgentBench 的 LLM 评估上限。

同时报告总体成功率、六类任务宏平均成功率、平均步骤、输入 token 和无效动作率。方法比较以同任务、同 repeat 的配对 bootstrap 95% CI 为主；成功率是论文主指标，其余效率指标用于解释行为，不代替成功率。

## 数据边界

- 训练轨迹只来自 `train`（3,553 题）。
- 配置选择只使用 `valid_seen`。历史 smoke 已接触的 1 题永久排除，正式验证为剩余 139 题，每题 3 repeats。
- 最终测试只使用 `valid_unseen`。历史 smoke 已接触的 1 题永久排除，冻结配置后一次性评估剩余 133 题，每题 3 repeats。
- 不允许根据 `valid_unseen` 反向选择模型、cap、阈值或 evidence strategy。

固定样本、数据哈希、任务类型覆盖和排除项见 `configs/experiments/alfworld_protocol_v1.json`。

## 轨迹池

先在独立的 30 个分层训练任务上配对比较 DeepSeek V4 Flash 和 Pro，只用于选择挖掘模型。比较成功率后，再比较成功轨迹的合法动作率、步骤和 token；若成功率接近，优先选择产生成功轨迹更高效且无效动作更少的模型。

正式候选池含 300 个训练任务，六类各 50 题，每题 4 repeats，共 1,200 条候选轨迹。池的停止条件同时要求：每类至少 30 个不同训练任务产生过满分轨迹，总满分轨迹不少于 300。未达标时只对不足类别每次增加 10 题并做 4 repeats。

Success-only 仅使用满分轨迹。Mixed 保留全部满分轨迹，并且只加入“同一任务至少已有一次满分”的失败轨迹作为对照；ALFWorld 没有部分奖励，因此不将普通零分失败包装成部分成功证据。两种 evidence 库独立构建，SkillX 仍严格只使用成功轨迹。

## 参数选择

从 WebShop 迁移的是算法级先验：High Top-1、diverse retrieval、success-only/mixed 分离，以及 GPT-5.4 renderer。WebShop 验证选出的 Mid cap3、Flat cap3 和相似度阈值不直接迁移。

ALFWorld 验证矩阵至少比较 NoSkill、Flat cap3/cap8、Tower cap3/cap8，以及 success-only/mixed。若 cap3 与 cap8 的成功率 CI 重叠，按预注册次序选择：成功率、平均步骤、输入 token、无效动作率。Tower 的 High Top-1 不在本矩阵重新选择。

最终测试冻结后运行 NoSkill、Flat、Success-only Tower、Mixed Tower、SkillX，并补齐 Mid-only 与必要的跨层机制消融。所有方法共享任务、repeat、agent model、20 步上限和底层技能构建池。
