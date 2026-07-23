# ALFWorld 官方实验

最终实验使用一个全局轨迹池和完整的 `valid_unseen` 测试集。SkillX 与
Trace2Tower 都按全局口径构建和检索，任务族分区不属于算法契约。

- `train/manifest.json`：训练轨迹池与 skill 构建产物。
- `validation/manifest.json`：官方验证划分，以及不得据此选择 cap 的声明。
- `test/manifest.json`：Flash 与 Pro 的正式实验矩阵。

最终报告只能引用这些 manifest 明确列出的产物。

Trace2Tower 预处理使用 ALFWorld/ALFRED 官方高层事件词表，并从
`take_action.action` 确定性解析 primitive action。当前活动管线为
`original-concept-v3`。早期无事件标签的 change-point 管线已物理隔离到
`deprecated/no-event-ablation/`，只能作为机制消融使用。

官方事件修复、结构统计与检索审计见 `OFFICIAL_EVENT_REPAIR.md`；首轮
Flash repeat0 配对结果见 `validation/OFFICIAL_EVENT_GRAPH_HIGH_VALIDATION.md`。
统一事件执行策略的失败集门控、114/139 全量诊断和下一版算法约束见
`validation/MANUAL_EVENT_POLICY_DIAGNOSTIC.md`。
SkillX 与 Trace2Tower 的实际注入内容及跨域差异见
`../../SKILL_INJECTION_CROSS_DOMAIN_AUDIT.md`。
