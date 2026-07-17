# SkillX 官方代码推理契约审计

本文只以冻结代码 `third_party/SkillX-native-36747f4` 为依据，不使用论文叙事解释
baseline。冻结 commit 为 `36747f424a17ea041e476adf2ff976a206ec9c30`。

## 官方主入口的实际控制流

`inference/skill_usage.py::SkillUsageService.prepare_prompt` 实际执行：

```text
task
-> Plan Top-3，默认余弦门槛 0.45
-> 只取第一条 Plan
-> 用 LLM rewrite，并以 rewritten Plan 覆盖原 Plan
-> rewritten Plan 存在时，逐 step 召回 Function，每步 Top-4，按名称去重
-> Plan 不存在时，直接用 task 召回 max_skills * 2 个 Function
-> 候选数超过 max_skills 才调用 SkillSelector，否则直接截断
-> 注入 rewritten Plan 与选中的 Function
```

官方 `AppWorldPromptFormatter` 会把 `plan` 写入 `# Reference Plan`。BFCL 和
Tau2Bench formatter 虽接收 `plan` 参数，但实际不使用。WebShop/ALFWorld 不属于官方
支持的 formatter，本项目采用 AppWorld 式通用注入：注入 rewritten Plan 和 Function，
不注入 AppWorld 专用 API 说明。

原始召回 Plan 不与 rewritten Plan 同时注入。原 Plan 只作为改写证据；rewrite 成功后，
agent 看到的是 rewritten Plan。

## 冻结代码中的接口错误

官方主入口调用：

```python
self.rewriter.rewrite(task=task, retrieved_plan=plan)
```

但 `PlanRewriter.rewrite` 的真实参数是 `retrieved_plans: List[Dict]`。该 commit 原样运行
会发生参数错误。本项目不修改 vendor，而在适配层把首个召回结果包装为单元素列表，
保留“Top-3 后只用第一条”的实际主入口控制流。

## 本项目的统一后处理

共享实现位于 `src/trace2tower/methods/skillx/native_inference.py`：

- rewrite prompt 直接从冻结 vendor 的 `PLAN_REWRITE_PROMPTS["default"]` 读取；
- selector prompt 直接从冻结 vendor 的 `SELECT_SKILL_PROMPT` 读取；
- rewrite 与 selector 都固定调用 `RENDERER_MODEL=gpt-5.4`；
- agent 执行使用 `AGENT_MODEL=deepseek-v4-flash`；
- rewrite 和 selector 均按官方实现最多尝试 3 次；
- selector 只在候选超过 8 条时调用，失败退回检索顺序前 8 条。

SkillX baseline 使用 P100 官方构建库：51 个 Plan、2 个 Function。因为 Function 只有
2 个，本轮 WebShop 基本不会触发 selector，旧 baseline 与新 baseline 的主要差异就是
Plan rewrite 及其后续注入。

Tower 复用相同 LLM 后处理，但保留方法自身的 High 契约：召回 Top-3 后只用首个 High
改写；注入 rewritten High；rewrite 失败则回退原始 High，保证每个任务始终有一个
端到端 High。原始 High 与 rewritten High 不同时注入。

## WebShop 冻结结果

均为 DeepSeek V4 Flash、repeat 0、100 个任务、0 个错误。

| 集合 | 方法 | mean reward | 满分率 | 平均步数 | 平均输入 token | 平均 context 字符 |
|---|---|---:|---:|---:|---:|---:|
| 验证集 | No-Skill | 0.65235 | 47% | 7.69 | 19,572 | 0 |
| 验证集 | SkillX，不 rewrite r1 | 0.64393 | 45% | 8.01 | 30,666 | 6,681 |
| 验证集 | SkillX，不 rewrite r2 | 0.66285 | 47% | 8.07 | 31,366 | 6,681 |
| 验证集 | **SkillX 官方实际入口** | **0.68427** | 48% | 7.95 | 34,044 | 6,758 |
| TestA | No-Skill | **0.68075** | **51%** | 7.98 | **21,388** | 0 |
| TestA | SkillX，不 rewrite | **0.72050** | **53%** | **7.43** | 28,072 | 6,667 |
| TestA | 旧 SkillX 实现，无 rewrite | 0.71224 | 49% | 6.92 | **24,141** | 6,679 |
| TestA | **SkillX 官方实际入口** | 0.66490 | 49% | 8.54 | 38,834 | 6,742 |

新 SkillX 在验证集相对 No-Skill 为 `+0.03192`，但在 TestA 为 `-0.01585`。相对旧的
未 rewrite 版本，TestA reward 下降 `0.04733`，满分率相同，平均步数增加 `1.62`。
因此旧 baseline 缺少 rewrite 不是可以忽略的实现差异；正式比较必须使用新结果。同时，
官方 rewrite 在 WebShop 并未带来稳定收益，反而增加了核验和试错。

在同一新 provider 内关闭 rewrite 后，SkillX 验证集两次独立运行分别为
`0.64393/0.66285`，均值 `0.65339`，与 No-Skill `0.65235` 基本持平；r2 单轮为
11 胜、79 平、10 负。TestA 则从 `0.66490` 升至 `0.72050`。这排除了旧 provider
其他实现差异，并证明 rewrite 的影响具有强 split interaction。正式 baseline 仍使用
官方默认的 `rewrite_plan=True`；no-rewrite 只作为机制消融。

## 权威运行

- TestA：`artifacts/runs/webshop-skillx-native-inference-p100-testa-flash-r1/`
- 验证集：`artifacts/runs/webshop-skillx-native-inference-p100-validation-flash-r1/`
- TestA no-rewrite：`artifacts/runs/webshop-skillx-no-rewrite-p100-testa-flash-r1/`
- 验证集 no-rewrite：`artifacts/runs/webshop-skillx-no-rewrite-p100-validation-flash-r1/`
- 验证集 no-rewrite 复跑：`artifacts/runs/webshop-skillx-no-rewrite-p100-validation-flash-r2/`
- SkillX 库：`artifacts/skillx/webshop-scale-v1/p100/execution/library.json`

这两个 run 是后续 WebShop SkillX 的冻结 baseline。旧 SkillX 数值只用于说明 rewrite
消融，不再作为官方 baseline。
