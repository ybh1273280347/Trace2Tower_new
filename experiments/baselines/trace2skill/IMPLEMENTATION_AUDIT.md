# Trace2Skill 实现完整性审计

审计结论：**通过**。机器可读明细见 `IMPLEMENTATION_AUDIT.json`。

本审计只判断复现链是否完整，不依据测试分数评价或修改方法。审计对象为 ALFWorld P310 与 WebShop P100 的 `+Combined`、`+Error` 四个一次性构建 artifact，以及对应四个正式评估运行。

## 1. 每条训练轨迹均进入 analyst

| 环境 | 原训练池 | 固定选择合同 | 选择轨迹 | 预期/完成 analyst batch | 覆盖唯一轨迹 | 重复 |
|---|---:|---|---:|---:|---:|---:|
| ALFWorld | 1,240 | 每题 `repeat_id=0` | 310 | 78 / 78 | 310 | 0 |
| WebShop | 400 | 每题 `repeat_id=0` | 100 | 25 / 25 | 100 | 0 |

审计按构建器的确定性排序和 batch-size 4 重新生成每个 analyst batch 的预期 `sample_id` 集合，并逐一与 checkpoint 中的记录集合比较。所有轨迹均有且仅有一条 analyst 记录，没有缺批、漏轨迹或重复轨迹。

## 2. Patch 没有因解析、截断或异常大量丢失

| 环境 | 局部 patch | 每轨迹范围 | 有 patch 的轨迹 | 零 patch 轨迹 | JSON/结构解析失败 | 非法 patch |
|---|---:|---:|---:|---:|---:|---:|
| ALFWorld | 844 | 1–3 | 310 | 0 | 0 | 0 |
| WebShop | 235 | 0–3 | 91 | 9 | 0 | 0 |

WebShop 的 9 条零 patch 轨迹仍有完整 analyst 记录和固定 outcome，只是 analyst 没有返回可泛化 lesson。它们不是丢失：所在 batch 均成功返回并通过 tool schema 解析，也没有异常或截断记录。官方方法允许无法形成可靠因果教训的轨迹不贡献 patch。

## 3. Parallel many-to-one consolidation 完整执行

| 环境/变体 | 输入 patch | 层次流 | Merge checkpoint | 最终 patch | Final skill |
|---|---:|---|---:|---:|---|
| ALFWorld +Combined | 844 | 844 → 230 → 81 → 27 → 11 | 39 / 39 | 11 | 已物化 |
| ALFWorld +Error | 281 | 281 → 74 → 22 → 8 | 13 / 13 | 8 | 已物化 |
| WebShop +Combined | 235 | 235 → 66 → 24 → 10 | 12 / 12 | 10 | 已物化 |
| WebShop +Error | 135 | 135 → 36 → 12 → 12 | 8 / 8 | 12 | 已物化 |

审计从原始局部 patch 数量出发，以 merge batch-size 32 重放归并拓扑。每层预期调用均存在合法 checkpoint，输出全部进入下一层，最终 checkpoint 数量与 build report 一致。

## 4. +Combined 和 +Error 按官方定义构建

- `+Combined`：合并所有 outcome 为 success 或 error 的 trajectory-local patches。
- `+Error`：只合并冻结 outcome 为 error 的 trajectory-local patches。

ALFWorld 分别使用 844 条全部 patch 和其中 281 条 error patch；WebShop 分别使用 235 条全部 patch 和其中 135 条 error patch。两个变体共享同一次逐轨迹 GPT-5.4 analyst 输出，`+Error` 没有重新解释 outcome，也没有混入 success patch。四个 artifact 的 `evolution_signal`、轨迹计数和训练池 SHA-256 均与该定义一致。

## 5. 测试时全文注入且没有检索

| 环境/变体 | 正式结果覆盖 | 精确全文哈希匹配 | Error attempts | 检索调用 |
|---|---:|---:|---:|---:|
| ALFWorld +Combined | 134 / 134 | 134 / 134 | 0 | 0 |
| ALFWorld +Error | 134 / 134 | 134 / 134 | 0 | 0 |
| WebShop +Combined | 100 / 100 | 100 / 100 | 0 | 0 |
| WebShop +Error | 100 / 100 | 100 / 100 | 0 | 0 |

每条结果只记录一个 Trace2Skill artifact ID；`skill_context_chars` 与 `skill_context_sha256` 均精确等于对应 artifact 的完整 `skill_markdown`。运行时 provider 直接返回全文，不持有 embedding、index、Top-k 或 retrieval 参数。

## 6. 每个自动变体只构建一次，不做测试集选优

正式边界内恰好有四个 artifact：两个环境 × 两个论文预定义变体；对应恰好四个正式 run。构建入口没有 seed 候选、训练侧 validation、Expert Skill 初始化或测试 manifest 输入，也没有候选接受门。

`+Combined` 和 `+Error` 作为两个独立预定义变体并列报告，不在变体内部生成多个 artifact，也不依据测试集选择 seed、checkpoint、patch 子集或技能版本。因此 61.94% 是 ALFWorld `+Error` 一次构建、一次正式评估的结果。

## 公平性边界

未向 Trace2Skill 单独增加以下特权：

- 三个 seed 后选择最优；
- 人工 Expert Skill 初始化；
- 额外训练侧 validation 或 acceptance gate；
- 测试时 retrieval；
- 超出冻结 P310/P100 合同的训练轨迹或构建预算。

在上述实现完整性成立后，结果应按原值报告。进一步为该 baseline 增加这些机制会改变比较预算或方法身份，而不是修复实现缺陷。
