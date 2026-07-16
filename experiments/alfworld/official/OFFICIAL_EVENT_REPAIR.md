# ALFWorld 官方事件管线修复

## 结论

ALFWorld 早期实现没有把官方事件接入图算法，实际退化为 change-point 分段、
纯语义聚类和旧版 Mid 检索。修复后，P310 原始训练池不扩池、不降低 High 支持
阈值，即可自动得到 4 个 Mid 和 12 条自然 High。运行时使用目标事件门控后，
139 个 `valid_seen` 初始状态上的 High 结构错配从 115 个降为 0。

当前正式候选为 `tower_4ff0dbf6b2420d21`，配置是 `include_high=true`、
graph retrieval、`mid_context_budget=3`。

## 修复边界

| 阶段 | 旧实现 | 当前实现 |
|---|---|---|
| primitive 解析 | `take_action.action` 字符串，但未进入事件图 | 同一原始字符串确定性解析为官方 primitive action |
| 事件分段 | embedding change point，无领域事件 | 按 ALFWorld/ALFRED 官方事件连续段分段 |
| embedding 输入 | 完整目标、观察或具体实体词 | 官方事件、动作 schema、片段长度；实体替换为角色 |
| Mid | 无事件图上的混合语义簇 | 事件片段、语义边、转移边、结果边联合谱聚类 |
| High | 无可信自然路径 | 成功/失败轨迹上的 Mid 序列对比挖掘 |
| 检索 | 旧版纯语义 Mid | 当前状态事件 + 目标事件门控的 graph retrieval |

这里没有使用任务族分区，也没有把专家 plan 对齐为训练标签。事件抽取有领域口径，
但“事件抽取后建图”仍是跨领域通用步骤。

## 结构对照

| 版本 | 片段签名 | 唯一签名 | 自动 Mid | 自然 High | 转移权重种类 | 连通分量 |
|---|---:|---:|---:|---:|---:|---:|
| v1 无事件消融 | change point / 完整语义 | - | 旧产物 | 0 | 1 | 52 |
| v2 中间诊断 | 官方事件 + 具体实体词 | 902 | 2 | 0 | 62 | 4 |
| v3 当前实现 | 官方事件 + 动作角色 schema | 56 | 4 | 12 | 62 | 1 |

v3 仍使用 `min_mid_clusters=2`、`max_mid_clusters=20` 自动 eigengap。
第 4 个候选后的特征值间隔约为 0.02310，明显高于前一个间隔约 0.00047，
因此 4 个 Mid 是图结构自动选择，不是手工指定。

## 预处理成本

| 指标 | v2 实体签名 | v3 角色签名 | 变化 |
|---|---:|---:|---:|
| 轨迹 | 1,240 | 1,240 | 不变 |
| transitions | 17,012 | 17,012 | 不变 |
| segments | 13,724 | 13,724 | 不变 |
| 唯一 embedding 文本 | 902 | 56 | -93.8% |
| embedding 请求 | 57 | 4 | -93.0% |
| embedding 输入 tokens | 19,514 | 1,721 | -91.2% |

## Mid 组成

| Mid | 节点数 | 主体事件 | 可解释功能 |
|---|---:|---|---|
| `mid_0000` | 6,788 | Pickup 29.1%、Open 28.1%、Put 22.6% | 容器访问、取物与放置 |
| `mid_0001` | 5,862 | Goto 84.6%、Scan 10.7% | 导航到目标并确认状态 |
| `mid_0002` | 717 | Scan 49.9%、Toggle 30.0%、Goto 12.7% | 观察、照明与局部检查 |
| `mid_0003` | 357 | Clean 59.1%、Close 20.7%、Scan 16.2% | 清洁与收尾状态检查 |

这些 Mid 是跨任务共享的功能阶段，不是 ALFWorld task-family 分桶。

## High 与渲染质量

原 2% `high_min_support_ratio` 下得到 12 条 High，`used_high_fallback=false`。
最强的零负支持路径正支持为 2.68%；多条清洁主路径正支持为 19.2% 至 26.3%，
且均高于负支持。

第一版 High 渲染错误地把 Mid 成员 ID 和 supporting trajectory ID 全量塞入 prompt，
12 张 High 的输入约 503 万 tokens，并造成少数轨迹叙事污染。修复为事件占比分层证据、
稳定 primitive grounding 和只传卡片正文后，全部 16 张 Mid/High 卡片输入降为
44,257 tokens，约减少 99.1%。第一版卡片已隔离到
`artifacts/trace2tower/alfworld/original-concept-v3/p310/deprecated/renderer-evidence-v1/`。

## Graph retrieval 审计

审计复用 Flash No-Skill 的 139 个 `repeat_id=0` 初始状态，只计算检索，不运行 agent。

| 目标事件 | 样本数 | High 行为 |
|---|---:|---|
| CleanObject | 26 | 26 个均召回 clean High |
| ToggleObject | 13 | 13 个均召回 toggle/inspection High |
| CoolObject | 22 | 图中无可信 cool High，回退 Mid-only |
| HeatObject | 14 | 图中无可信 heat High，回退 Mid-only |
| Cool + Heat | 1 | 无同时覆盖两事件的 High，回退 Mid-only |
| 普通搬运 | 63 | 无排他性目标事件，回退 Mid-only |

修复前强制 Top-1 High 导致 115/139（82.73%）目标外 clean/lamp 召回；加入路径事件
契约后，High 所含排他事件与目标事件的结构错配为 0/139。该门控复用现有 0.1
事件相容阈值，没有调低 High 门槛或修改 cap。

## 首轮 Flash 验证

`deepseek-v4-flash`、同一 139 个 `valid_seen` 样本、`repeat_id=0` 的正式配对
结果为：No-Skill 85/139，官方事件 Tower 85/139，成功率同为 61.15%。配对翻转
为 8 胜 8 负，bootstrap 95% CI 为 [-5.76, +5.76] 个百分点，McNemar 双侧
精确检验 `p=1.0`。

因此，修复版相对旧无事件 Mid-only cap3 的 55.40% 恢复了 5.76 个百分点，
消除了旧实现的整体负收益；但当前证据不支持它优于 No-Skill。详细分层结果见
`validation/OFFICIAL_EVENT_GRAPH_HIGH_VALIDATION.md`。
