# WebShop 当前 Baseline 口径

本文件冻结 2026-07-14 已确定的 baseline 身份和已完成实验数据。机器可读版本见 `baseline-freeze-v1.json`。

## Baseline 身份

当前同时保留两个不同用途的 baseline，不能混称：

- `NoSkill` 是零信息对照，用来估计注入 skill 的因果增量。
- `Manual V2` 是当前确定的强 skill baseline。它是针对 WebShop 精心设计、包含充分任务知识的成熟策略，用来回答“如果 skill 内容和注入都正确，Pro agent 能否获益”。

`Manual V2` 不是自动学习方法，也不是 Trace2Tower 的组成部分。它绕过 embedding 检索和层级展开，将 `diagnostics/manual-skill-v2.md` 原文完整注入 `# Retrieved Experience`。因此它是强上界参照，不可用于声称算法已经从轨迹学会了该策略。

P50 SkillX、P50 Success Tower 和旧 Flat 继续作为方法参照，但不替代上述两个 baseline。Direct Induction V1/V2 是负结果诊断；Clustered Induction 是待评估候选。

## 固定评估协议

- Benchmark：WebShop。
- Agent：`deepseek-v4-pro`。
- Cohort：`configs/experiments/webshop_scale_v1.json` 中 selection seed `20260715` 固定的 100 个 test tasks。
- Repeats：每任务 `0/1/2`，每条件 300 episodes，最大 20 步。
- 配对键：`sample_id + repeat_id`。
- Bootstrap：先对每个 task 的 repeats 取均值，再按 task 做 10,000 次 bootstrap。
- 主指标：mean reward。
- 同报指标：满分率、completion、步骤、无效动作率、agent chat 输入 token、skill 字符数。
- 满分定义：`primary_score >= 0.999`；completion 与满分分开报告。
- 不报告构建或运行用时，因为它依赖设备和并发，不是模型调用成本。

## 已完成数据

| 条件 | Mean reward | 满分率 | Completion | 平均步数 | 无效动作率 | Chat 输入 | Skill 字符 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| NoSkill | 0.6270 | 42.7% | 81.0% | 8.91 | 9.69% | 23,441 | 0 | completed |
| Manual V2 | **0.7216** | **49.0%** | **93.7%** | **5.97** | **0.61%** | **15,247** | 2,552 | frozen strong baseline |
| P50 SkillX | 0.7050 | 48.3% | 91.7% | 6.62 | 3.78% | 22,468 | 5,357 | completed reference |
| P50 Success Tower | 0.6780 | 47.7% | 88.0% | 7.58 | 6.38% | 23,947 | 3,818 | completed reference |
| Direct Induction V1 | 0.4991 | 39.0% | 59.0% | 12.85 | 9.21% | 49,953 | 3,669 | completed negative diagnostic |
| Direct Induction V2 top-1 | 0.5221 | 37.0% | 69.3% | 11.89 | 5.36% | 44,885 | 2,535 | completed negative diagnostic |
| Clustered Induction | - | - | - | - | - | - | - | build-only, not evaluated |

Manual V2 相对 NoSkill 的配对效果：reward `+0.0946`，95% CI `[+0.0442, +0.1471]`；满分率 `+6.3` 个百分点，95% CI `[+1.3, +12.0]`；平均步骤 `-2.94`，输入 token `-8,195`。这证明成熟 skill 的确能被当前注入链路和 Pro agent 有效利用。

Manual V2 相对 P50 SkillX 的 reward 为 `+0.0167`，相对 P50 Success Tower 为 `+0.0436`，两者 reward CI 都跨零。强 baseline 的作用是提供成熟策略参照，不是宣称其显著超过所有自动方法。

## Direct Induction 结论

V1 将整份成功轨迹 corpus 交给 GPT-5.4 后生成原子决策卡。运行时 top-k 经常只得到搜索、候选筛选和属性核验，购买闭环几乎没有被检索到，最终显著退化。

V2 强制生成 4 张独立、完整、端到端技能并固定 top-1。卡片文本闭环通过人工审查，但 `294/300` episodes 都被路由到“搜索精炼与恢复”，说明按初始 goal 无法判断“第一次搜索是否失败”这种未来执行状态。它否定的是当前直接归纳加静态路由方式，不是“任何轨迹归纳都不可能有效”。

## Clustered Induction 当前状态

Clustered Induction 使用相同 P50 训练池的 94 条满分轨迹。26 个至少成功一次的训练任务按 `sample_id` 聚合，重复轨迹不会跨簇。候选 `k=2..6` 只使用训练数据，以 cosine silhouette 最大且每簇至少 3 个任务为规则，最终选出 5 簇，任务数为 `4/6/7/5/4`；GPT-5.4 为每个固定簇生成一张端到端卡。

该方法目前只有 build artifact，没有 rollout 结果。它必须继续标记为 `build-only`，不能和上表已完成条件比较，也不能称为 Tower。

## 数据来源

- Manual 汇总：`artifacts/experiments/webshop-scale-v1/manual-skill-diagnostic-summary.json`
- Direct Induction 汇总：`artifacts/experiments/webshop-scale-v1/global-flat-gpt54-summary.json`
- Cluster membership：`artifacts/flat_skill_summary/webshop-p50-clustered-gpt54-v1/clusters.json`
- Clustered library：`artifacts/flat_skill_summary/webshop-p50-clustered-gpt54-v1/library.json`
- 全部路径与 SHA-256：`baseline-freeze-v1.json`
