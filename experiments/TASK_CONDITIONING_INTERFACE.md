# Trace2Tower 统一任务条件接口

## 结论

任务条件检索已经从 ALFWorld 的对象特化实现重构为领域无关核心。核心检索器不再
读取 `target_object`、`destination`、`clean`、`price` 等领域字段，只消费统一的
`TaskCondition`，并把条件抽取、兼容判断和具体任务绑定交给领域适配器。

这不意味着事件抽取或任务解析可以跨数据集共享。每个数据集仍需定义自己的领域
事件和任务条件；可复用的是“领域条件门控、语义重排、查询时具体化”的执行契约。

## 核心契约

| 组件 | 责任 | 不负责 |
|---|---|---|
| `TaskCondition` | 保存检索文本、领域事件和开放属性 | 解释属性含义 |
| `TaskConditionProfile` | 持久化 `skill_id -> TaskCondition` 并绑定领域 | 社区发现 |
| `DomainTaskAdapter` | 抽取查询、构建社区条件、判断兼容、绑定具体任务 | 语义向量排序 |
| `retrieve_task_conditioned_high` | 按兼容等级优先、语义相似度次优选择 High | 读取领域槽位 |
| `TaskConditionedHighProvider` | 加载 Tower、侧车和适配器并在任务开始注入一次 | 动态 Mid 检索 |

兼容等级为：

1. `exact`：当前任务与社区条件完全兼容；
2. `partial`：关键实体或约束兼容，但部分完成条件不同；
3. `workflow`：只共享可迁移的事件流程；
4. `incompatible`：不得注入。

核心始终先比较兼容等级，再在同等级候选中比较语义相似度。是否允许
`partial/workflow` 由运行配置的 `minimum_task_compatibility` 决定。任何非完全匹配
技能也必须经过领域适配器重新绑定，核心不假定某种领域的替换方式。

## 领域适配

| 数据集 | 查询条件 | 兼容规则 | 具体化方式 |
|---|---|---|---|
| ALFWorld | canonical goal、官方变换事件、目标对象、目的地 | 对象/事件/目的地完全匹配为 exact；同对象同事件为 partial；同事件为 workflow | 绑定当前对象、设备、目的地和动作链 |
| WebShop | 完整购买要求、购物事件链、约束词项 | 约束集合相同为 exact；有约束重合为 partial；仅流程相同为 workflow | 把当前商品类别、属性、规格和预算原文绑定进执行卡 |

WebShop 不使用商品名称完全相等作为硬门槛。训练和测试商品可以不同，社区提供的
是搜索、验证、选项选择和购买流程；当前任务的具体约束在注入时绑定。因此统一接口
不会把 ALFWorld 的实体相等规则错误迁移到购物任务。

## 新数据集接入

接入新数据集只需：

1. 完成该数据集自己的事件抽取；
2. 实现 `DomainTaskAdapter` 的四个方法；
3. 在社区构建阶段输出统一 `TaskConditionProfile` 侧车；
4. 为内置 benchmark 在 factory 注册适配器，或直接向 provider 传入适配器实例；
5. 通过兼容门控、具体化内容和无兼容候选三个测试。

核心检索函数不依赖 `Benchmark` 枚举。测试中的 Calendar fake adapter 没有注册到
factory，仍可直接完成条件检索，证明第三方数据集不需要修改核心算法。

## 验证

| 验证 | 结果 |
|---|---:|
| 统一接口及两域/第三方适配定向测试 | 18 passed |
| ALFWorld v11 侧车 | 160 条条件，覆盖 160 张 High |
| ALFWorld exact + workflow 在线冒烟 | 2/2 成功 |
| WebShop P100 T1 侧车 | 6 条条件，覆盖 6 张 High |
| WebShop 统一 provider 在线冒烟 | 正常完成，`webshop:0` 得分 0.6 |

WebShop 单样本只证明接口和约束绑定能够运行，不构成性能提升结论。现有 WebShop
graph-cap3 正式结果保持冻结；是否把任务条件检索并入最终方法，必须另做离线召回
审计和失败集门控实验。

