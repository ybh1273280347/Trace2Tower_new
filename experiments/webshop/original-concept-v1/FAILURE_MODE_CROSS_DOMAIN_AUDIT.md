# 跨数据集失败模式审计

本审计固定使用 DeepSeek v4 Flash、repeat0，并直接读取逐任务结果与轨迹。WebShop 使用
当前概念一致的 v6 P200 Tower；ALFWorld 使用 v18 Tower。目的不是重新选择检索阈值，
而是解释为什么同一套 High+Mid 机制在 ALFWorld 有明显增益，在 WebShop 不能稳定复现。

## 总体翻转

| 数据集/方法 | reward | 完全成功率 | 相对 NoSkill | 胜/负/平 |
|---|---:|---:|---:|---:|
| ALFWorld NoSkill | 0.52985 | 52.99% | - | - |
| ALFWorld SkillX | 0.81343 | 81.34% | +0.28358 | 16/7/111 |
| ALFWorld Tower | 0.88060 | 88.06% | **+0.35075** | **51/4/79** |
| WebShop NoSkill | 0.68075 | 51% | - | - |
| WebShop SkillX | 0.71224 | 49% | +0.03149 | 参考基线 |
| WebShop v6 Tower | 0.67417 | 50% | -0.00658 | 8/13/79 |

WebShop SkillX 的点估计是正的，但完全成功率没有提高，且当前证据不足以称为稳定
优势；Tower 则在 Test-A 负向。相较之下，ALFWorld 两种技能方法都取得大幅点估计
提升，且 Tower 的零分救回结构非常清晰。

从零分翻转看：

| 数据集/对照 | 救回基线零分 | 新增零分 |
|---|---:|---:|
| ALFWorld Tower vs NoSkill | 51 | 4 |
| WebShop v6 Tower vs NoSkill | 5 | 8 |

ALFWorld 的 51 个救回样本主要是 `put` 组合（18）、`cool`（11）、`clean`（9）和
`heat`（8），这些任务需要跨多个状态和前置条件执行。Tower 将失败率从 47.01% 降到
11.94%，平均无效动作从 0.54 降到 0.22。

WebShop 则相反：Tower 只救回 5 个原本零分任务，却让 8 个原本有正 reward 的任务变成
零分。Tower 平均上下文约 3,897 字符、平均步数 9.15；NoSkill 平均步数 7.98。
这更像错误指导造成的回退，而不是缺少长程程序。

## WebShop 退化样本

以下 8 个样本全部是 Tower 零分、NoSkill 非零。括号内是实际注入的 Top-3 High 中
最具代表性的错误类别：

| 样本 | 外部目标 | High 参考类别 | 终止状态 |
|---|---|---|---|
| `webshop:126` | 高性能 laptop，1TB/16GB/i7/Nvidia | tower PC | 产品页 Description，20 步耗尽 |
| `webshop:201` | 30x30x46 蓝色实木 ottoman | 白色 faux-fur ottoman bench | 产品页颜色选择，20 步耗尽 |
| `webshop:232` | 红色 queen heavy-duty bed frame | 黑色 king bed frame | Back to Search 后搜索不可用，20 步耗尽 |
| `webshop:46` | 黑白猫图案 iPhone case | Oribox screen protector | 产品页回退，20 步耗尽 |
| `webshop:920` | 非毒性 hair-loss treatment | hair extensions | 产品页 Description，20 步耗尽 |
| `webshop:969` | cheese platter | cupcake topper | 产品页 Buy Now 前耗尽 |
| `webshop:977` | 2 件 teal curtains，可机洗 | hair-drying towel | 产品页颜色选择，20 步耗尽 |
| `webshop:990` | olive camo high-waist yoga pants | jeans | 产品页 Attributes，20 步耗尽 |

这不是单纯的属性遗漏，而是一级商品类别错配。对应地，5 个 Tower 救回样本的 High
参考类别均与目标一致（apron、boots、nut bars、lemonade、slippers），且都走到
`Buy Now` 或得到正 reward。

## 为什么 ALFWorld 不同

ALFWorld 的任务目标和事件程序具有较强的离散结构：`heat/cool/clean -> pick -> put`
以及容器、持有状态、目标位置等前置条件。一个 High 即使不是同一对象，也能提供可迁移
的程序骨架；对象条件化改写再负责绑定具体实体。

WebShop 的核心难点不是抽象程序，而是检索阶段必须先选对商品类别，再逐项核验属性、
价格和选项。当前 Top-3 High 的 `retrieval_condition` 在商品类别上不够硬，LLM 改写
虽然能改写约束，却仍会把错误类别的搜索/回退/购买轨迹作为参考。结果是：

1. 正确类别的 High 能带来明显收益；
2. 错误类别的 High 会把执行模型带到错误商品或无效回退；
3. WebShop NoSkill 本身已有 0.68 左右平均 reward，错误 High 的负迁移很容易抵消少量正收益；
4. ALFWorld NoSkill 只有 0.53 左右，正确程序指导的收益空间更大。

因此当前跨域差异首先是 **High 召回条件的对象类别精度问题**，不是图是否能发现社区，
也不是 Mid 数量不足。后续实验应审计并强化商品类别绑定，先做离线 Top-3 High 类别
命中率和失败样本替换实验，再决定是否重跑全量；不应继续盲调 Mid 阈值或上下文预算。
