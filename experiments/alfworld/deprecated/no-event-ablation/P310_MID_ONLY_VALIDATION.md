# ALFWorld No-Event Mid-only 单次消融

## 结论

这轮结果不支持“原池图结构在不使用 High 时已经优于 NoSkill”。在严格配对的 139 个 `valid_seen` 样本上，原 310-task 训练池构建的 Mid-only cap3 在 Pro 上比 NoSkill 低 3.60 个百分点，在 Flash 上低 5.76 个百分点。

但 cap3 相比 cap8 的 66.91% 恢复了 5.04 个百分点，平均输入 token 从 61,733 降到 47,378。High 路径、High 卡和 High 索引均为空，两组运行记录中的 High skill 使用次数均为 0。因此可以确认选择性检索缓解了上下文饱和，但当前证据仍不足以宣称纯 Mid 图结构总体优于 NoSkill。

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | `deepseek-v4-pro` |
| 验证集 | `valid_seen`，沿用既有 NoSkill 的 139-sample cohort |
| Repeat | 仅 `repeat_id=0` |
| 训练池 | 原始 310 个 train 任务，1240 条轨迹 |
| Tower | 9 个图聚类 Mid，0 个 High |
| 检索 | legacy Mid-only，比较 `direct_mid_top_k=8` 与 `3` |
| NoSkill | 复用 `alfworld-dev-v1-pro-noskill` 的 repeat 0，不新增调用 |
| Mid-only cap8 | `alfworld-dev-v1-pro-p310-mid-only-r1` |
| Mid-only cap3 | `alfworld-dev-v1-pro-p310-mid-only-cap3-r1` |

当前 `alfworld_dev.jsonl` 比既有正式 cohort 多一个先前暴露样本 `alfworld:valid_seen:trial_T20190906_173120_350651`。该样本虽被本轮运行一次，但不进入配对结论。

## 总体结果

| 指标 | NoSkill | Mid-only cap8 | Mid-only cap3 |
|---|---:|---:|---:|
| 成功数 | 105 / 139 | 93 / 139 | 100 / 139 |
| 成功率 | 75.54% | 66.91% | 71.94% |
| 相对 NoSkill | - | -8.63 pp | -3.60 pp |
| 平均步数 | 12.87 | 13.38 | 13.00 |
| 成功样本平均步数 | 10.56 | 10.11 | 10.27 |
| 平均 invalid action | 0.38 | 0.64 | 0.60 |
| 平均输入 token | 39,184 | 61,733 | 47,378 |
| 平均 skill context 字符数 | 0 | 88,193 | 30,635 |

逐样本配对结果：

| 配对比较 | 正向翻转 | 反向翻转 | McNemar exact p |
|---|---:|---:|---:|
| cap3 相对 NoSkill | 9 | 14 | 0.4049 |
| cap3 相对 cap8 | 13 | 6 | 0.1671 |

cap3 相对 cap8 净恢复 7 个成功，但相对 NoSkill 仍净损失 5 个成功。两项差异均未达到常用显著性门槛。

## 按任务族

| 任务族 | 样本数 | NoSkill | cap8 | cap3 |
|---|---:|---:|---:|---:|
| `look_at_obj_in_light` | 13 | 12 | 8 | 10 |
| `pick_and_place` | 35 | 34 | 31 | 33 |
| `pick_clean_then_place` | 27 | 14 | 15 | 17 |
| `pick_cool_then_place` | 24 | 14 | 12 | 13 |
| `pick_heat_then_place` | 16 | 13 | 10 | 10 |
| `pick_two_obj_and_place` | 24 | 18 | 17 | 17 |

`pick_clean_then_place` 是唯一超过 NoSkill 的任务族：cap3 为 17/27，NoSkill 为 14/27。这个局部信号说明图聚类 Mid 并非完全无效，但不能替代总体结果。

## Flash 单次复核

Flash 使用完全相同的 139 个样本、`repeat_id=0`、P310 Mid-only Tower 和 cap3 runtime。NoSkill 与 Mid-only 均重新运行一次。

| 指标 | Flash NoSkill | Flash Mid-only cap3 | 差值 |
|---|---:|---:|---:|
| 成功数 | 85 / 139 | 77 / 139 | -8 |
| 成功率 | 61.15% | 55.40% | -5.76 pp |
| 平均步数 | 13.82 | 14.42 | +0.60 |
| 成功样本平均步数 | 9.89 | 9.92 | +0.03 |
| 平均 invalid action | 0.49 | 0.63 | +0.14 |
| 平均输入 token | 44,631 | 56,058 | +11,427 |

逐样本配对中，6 个 NoSkill 失败样本被 cap3 救回，14 个 NoSkill 成功样本在 cap3 下失败，McNemar exact 双侧 `p=0.1153`。High skill 使用次数为 0。

按任务族看，`pick_two_obj_and_place` 从 12/24 提高到 13/24；`look_at_obj_in_light` 和 `pick_clean_then_place` 持平；其余三个任务族下降。Flash 与 Pro 的总体方向一致，说明当前 Mid-only cap3 的负差异不是单一模型特例。

## 解释边界

cap8 在只有 9 个 Mid 的 Tower 上几乎失去选择性：139 个配对样本全部注入 8 或 9 个 skill ID，平均输入 token 增加 57.5%。cap3 将上下文字符数减少 65.3%，并将成功率从 66.91% 恢复到 71.94%，支持“选择性检索是必要条件”。

但 cap3 在 Pro 和 Flash 上都低于 NoSkill，因此现有证据不能写成“即使没有可信 High，图结构依然带来总体收益”。可辩护的表述是：

> 泛任务训练池无法产生可信 High，因此我们验证了纯 Mid 图结构。将检索预算从 8 收紧到 3 后，成功率恢复 5.04 个百分点，并在 `pick_clean_then_place` 上超过 NoSkill，说明图聚类 Mid 包含局部可用信息；但总体仍低于 NoSkill，表明现有语义检索尚不能稳定地把这些信息分配给所有任务。

严格来说，这里运行的是“图构建得到的 Mid + legacy cap3”，不是 WebShop 的 `retrieval_strategy: graph`。当前 graph retrieval 依赖 High 路径和 WebShop 事件 profile，不能直接用于无 High 的 ALFWorld Tower。

## 产物

- Mid-only Tower：`artifacts/trace2tower/alfworld/deprecated/original-concept-v1/p310-mid-only/tower.json`
- cap8 runtime：`configs/experiments/deprecated/no-event-alfworld/alfworld_trace2tower_mid_only_runtime.yaml`
- cap3 runtime：`configs/experiments/deprecated/no-event-alfworld/alfworld_trace2tower_mid_only_cap3_runtime.yaml`
- cap8 run：`artifacts/runs/alfworld-dev-v1-pro-p310-mid-only-r1`
- cap3 run：`artifacts/runs/alfworld-dev-v1-pro-p310-mid-only-cap3-r1`
- NoSkill run：`artifacts/runs/alfworld-dev-v1-pro-noskill`
- Flash cap3 run：`artifacts/runs/alfworld-dev-v1-flash-p310-mid-only-cap3-r1`
- Flash NoSkill run：`artifacts/runs/alfworld-dev-v1-flash-noskill-r1`
