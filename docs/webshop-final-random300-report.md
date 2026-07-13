# WebShop Random-300 正式实验报告

## 实验范围

本报告评估冻结后的 WebShop 配置，不承担调参功能。配置选择依据见 [WebShop 验证与正式测试口径](webshop-validation-and-final-test-protocol.md)。正式测试预注册 selection 为 `selection_32248afcaee8da76`。

正式测试从抽样前从未出现在任何实验 result 或 error 中的候选样本里，用三个固定随机 seed `20260713`、`20260714`、`20260715` 分别抽取 100 个任务。三个集合互不重叠，共 300 个独立任务。每个任务执行 `repeat_id = 0, 1, 2`，所以每个模型-方法条件包含 900 个 episode。

测试矩阵在首次 rollout 前冻结：

- 执行模型：`deepseek-v4-flash`、`deepseek-v4-pro`
- 方法：NoSkill、Flat cap3、Success-only Tower cap3、Mixed Tower cap3
- 满分成功：`primary_score >= 0.999`
- 置信区间：先在同一任务内平均三次重复，再对 300 个任务做 10,000 次 task-cluster bootstrap
- 技能方法只与同模型、相同 `(sample_id, repeat_id)` 的 NoSkill 配对

执行 manifest SHA-256 为 `b055ef5458374c0b8e34935dd59d83f1a90d023bf93fb6a7d2c27c61bcd8fc3e`。8 个条件均达到 900/900 覆盖，合计 7,200 个 episode；每组均无重复 key、缺失 key、越界 key或未解决错误。模型、repeat、cap 和 artifact SHA-256 均与预注册一致。

## 汇总结果

| 模型 | 方法 | Mean reward | 满分成功率 | Completion | 平均步数 | 每 episode 累计输入 token |
|---|---|---:|---:|---:|---:|---:|
| Flash | NoSkill | 0.7037 | 49.7% | 90.1% | 7.45 | 18,886 |
| Flash | Flat cap3 | 0.7050 | 51.2% | 88.9% | 8.54 | 27,892 |
| Flash | Success-only Tower cap3 | 0.7186 | 51.3% | 92.2% | 7.11 | 22,764 |
| Flash | Mixed Tower cap3 | 0.7153 | 51.7% | 92.9% | 7.19 | 22,140 |
| Pro | NoSkill | 0.6543 | 46.8% | 81.4% | 8.69 | 23,664 |
| Pro | Flat cap3 | 0.6930 | 52.9% | 84.7% | 9.12 | 30,500 |
| Pro | Success-only Tower cap3 | 0.7029 | 51.9% | 87.1% | 8.12 | 26,709 |
| Pro | Mixed Tower cap3 | 0.6845 | 51.2% | 85.8% | 8.22 | 26,518 |

输入 token 是 Agent 在一个 episode 中各步请求的累计输入量，包含提示、任务、交互历史和技能上下文，不是技能构建成本。

## Flash 结果

| 比较 | Reward 差 | Reward 95% CI | 满分率差 | 满分率 95% CI |
|---|---:|---:|---:|---:|
| Flat cap3 vs NoSkill | +0.0013 | [-0.0242, +0.0276] | +1.6% | [-1.7%, +4.9%] |
| Success-only Tower vs NoSkill | +0.0148 | [-0.0106, +0.0410] | +1.7% | [-1.9%, +5.3%] |
| Mixed Tower vs NoSkill | +0.0116 | [-0.0137, +0.0379] | +2.0% | [-1.7%, +5.8%] |
| Mixed vs Success-only Tower | -0.0033 | [-0.0271, +0.0207] | +0.3% | [-2.9%, +3.6%] |
| Success-only Tower vs Flat | +0.0136 | [-0.0130, +0.0403] | +0.1% | [-3.4%, +3.7%] |

Flash 的三种技能方法在点估计上都没有明显伤害，但 reward 和满分率区间均跨零。正式测试不能确认任何技能方法稳定优于 Flash NoSkill，也不能确认 Tower 优于 Flat，或 mixed 优于 success-only。

效率上，Flat 平均多 1.08 步、每 episode 多约 9,005 个输入 token，并且 completion 低 1.2 个百分点。两种 Tower 的平均步数反而比 NoSkill 少约 0.3，completion 高 2.1 至 2.8 个百分点，输入 token 增量约 3,254 至 3,878。即使 reward 增益未被确认，Tower 在 Flash 上比 Flat 更符合成本约束。

## Pro 结果

| 比较 | Reward 差 | Reward 95% CI | 满分率差 | 满分率 95% CI |
|---|---:|---:|---:|---:|
| Flat cap3 vs NoSkill | +0.0387 | [+0.0069, +0.0715] | +6.1% | [+2.6%, +9.8%] |
| Success-only Tower vs NoSkill | +0.0486 | [+0.0176, +0.0801] | +5.1% | [+1.3%, +8.9%] |
| Mixed Tower vs NoSkill | +0.0303 | [-0.0021, +0.0634] | +4.4% | [+0.7%, +8.2%] |
| Mixed vs Success-only Tower | -0.0183 | [-0.0469, +0.0100] | -0.7% | [-4.1%, +2.8%] |
| Success-only Tower vs Flat | +0.0099 | [-0.0175, +0.0377] | -1.0% | [-4.0%, +1.9%] |

Pro 上的经验学习效应得到正式确认：

- Flat cap3 同时显著提高 mean reward 和满分成功率。
- Success-only Tower cap3 同时显著提高 mean reward 和满分成功率，并将平均步数减少 0.57、completion 提高 5.7 个百分点。
- Mixed Tower cap3 显著提高满分成功率；reward 点估计为正，但 95% CI 略跨零，因此不能确认连续 reward 提升。

Success-only Tower 与 Flat 的直接差异不显著。Success-only Tower 的 reward 点估计更高、输入 token 少约 3,791、平均少 1.0 步、completion 高 2.4 个百分点；Flat 的满分率高 1.0 个百分点。正确结论是两者都对 Pro 有效，而 Tower 更有效率，不是 Tower 在 reward 上已被证明优于 Flat。

Mixed 相对 success-only 的两个主要区间均跨零。保留错误和部分成功轨迹的消融没有证明额外收益，也没有证明稳定伤害。默认使用 success-only 仍与验证阶段的冻结决策一致；mixed 应继续作为科学消融报告，而不是主方法。

## 三个随机样本集的一致性

Pro 的 skill uplift 在三个 100-task seed 上保持同方向：

| Seed | Flat reward uplift | Success Tower uplift | Mixed Tower uplift |
|---:|---:|---:|---:|
| 20260713 | +0.0187 | +0.0354 | +0.0224 |
| 20260714 | +0.0616 | +0.0557 | +0.0179 |
| 20260715 | +0.0358 | +0.0547 | +0.0504 |

Flash 的 uplift 较小并存在正负波动：Flat 为 `-0.0117 / +0.0057 / +0.0099`，Success Tower 为 `+0.0137 / +0.0063 / +0.0245`，Mixed Tower 为 `+0.0061 / -0.0050 / +0.0336`。这与合并后的区间结果一致：Pro 的经验收益稳定，Flash 的收益接近零。

## 模型差异

在纯 NoSkill 下，Pro 相对 Flash 的 reward 差为 `-0.0495`，95% CI `[-0.0846, -0.0148]`；满分率差为 `-2.9%`，CI `[-7.2%, +1.4%]`。因此，本实验确认的是 Pro 在该 WebShop Agent 设置下裸 reward 更低，而不是“名义更强的模型必然有更高基线”。Pro 还平均多 1.24 步，每 episode 多约 4,778 个输入 token。

技能对 Pro 的 uplift 均大于对 Flash 的 uplift，但预注册的模型×技能差分中的差分尚未达到区间不跨零：

- Flat：`+0.0374`，95% CI `[-0.0008, +0.0760]`
- Success-only Tower：`+0.0338`，CI `[-0.0063, +0.0741]`
- Mixed Tower：`+0.0187`，CI `[-0.0241, +0.0596]`

所以可以确认“技能对 Pro 有效、对 Flash 未建立有效性”，但不能进一步声称跨模型的 skill-uplift 交互已经显著。该方向接近边界，适合作为后续机制研究，而不是本轮主假设结论。

## 最终结论

本轮正式测试支持以下结论：

1. WebShop 技能注入的价值依赖执行模型。固定技能库对 Pro 有明确正向价值，对 Flash 没有建立稳定增益。
2. Pro 上 Flat cap3 与 Success-only Tower cap3 都显著优于 NoSkill；Success-only Tower 的执行效率更好，但没有显著击败 Flat。
3. Mixed Tower 没有显著优于 Success-only Tower。错误轨迹仍是必要消融，但当前不能作为性能改进主张。
4. Flash 上 Tower 的点估计和效率优于 Flat，但所有主要 reward/满分率差异区间跨零，不能宣称方法提升。
5. 名义模型强度不能替代任务内基线测量。Pro NoSkill reward 显著低于 Flash NoSkill，但 Pro 能从冻结技能中获得显著提升。

没有任何正式测试结果被用于修改 cap、证据策略或 artifact。后续若继续开发，应使用新的验证样本提出改进，再预注册另一批未见测试样本；不得在本 Random-300 上反向选择配置。
