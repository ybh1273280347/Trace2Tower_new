# ALFWorld P310 构建阶段 GPT 用量

本表只统计技能构建阶段的 GPT/LLM chat 调用，不含训练轨迹采集、embedding、测试时 plan rewrite、评估 Agent 或货币价格估算。

| 方法 | 构建模型 | GPT 调用 | 输入 token | 输出 token | 总 token |
|---|---|---:|---:|---:|---:|
| Trace2Tower | GPT-5.4 | 157 | 983,841 | 41,891 | 1,025,732 |
| SkillX | GPT-5.4 | 150 | 1,133,696 | 99,945 | 1,233,641 |
| ExpeL | GPT-5.4 | 74 | 未保留 | 未保留 | 未保留 |

Trace2Tower 相较 SkillX 的输入、输出和总 chat token 分别减少 13.22%、58.09% 和 16.85%；调用数增加 4.67%。因此，受证据支持的结论是 Trace2Tower 通过图聚合降低了总体生成量，尤其显著减少输出 token，而不是减少调用次数。

ExpeL 的全量 checkpoint 保留了 46 次成功/失败比较更新和 28 次成功规则更新，共 74 次 GPT 调用；但恢复构建时没有把原始 provider token usage 写回报告，`rule_input_tokens=0` 和 `rule_output_tokens=0` 表示记录缺失，不表示零成本。为避免误导，表中记为“未保留”。

机器可读数据见 `ALFWORLD_CONSTRUCTION_COST.json`。
