# WebShop Experiments

当前唯一有效的实验协议是 `event-tower-v2/`。它重新冻结方法身份、事件聚类契约、P50/P100 规模比较以及 validation/test 划分。

- `event-tower-v2/`: 当前实验，尚未 rollout。
- `deprecated/`: 2026-07-14 之前的配置、报告和诊断。旧 Tower 没有在 Mid 聚类阶段强制事件分层，不能作为完整 Trace2Tower 结果。

旧 artifacts 和训练轨迹没有删除。P50/P100 No-Skill 训练池继续使用；旧 Tower snapshot 禁止作为 v2 Full artifact。
