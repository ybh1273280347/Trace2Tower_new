# WebShop Event Tower V2

这是当前唯一有效的 WebShop 实验目录。

- `PROTOCOL.md`：方法身份、Trace2Tower 算法、数据划分、模型、统计和 artifact 契约。
- `EXPERIMENTS.md`：八个阶段的执行清单、cap3/cap5/cap8 验证和 Flash 门控。
- `manual-skill.md`：冻结的人工技能强基线。
- `manifests/`：固定 validation/test/ablation 样本，运行时展开 repeat IDs `0/1/2`。
- `stage-1-pools/`：P50/P100 训练轨迹池的冻结审计与机器清单。
- `stage-2-skills/`：P50 baseline libraries、Tower snapshots 和完整 provenance 审计。
- `stage-3-validation/`：3,600 个 validation episodes、cap 统计和冻结选择。
- `configs/experiments/webshop_event_tower_v2.json`：机器可读的同一份实验计划。

旧结果、旧配置和旧文档已进入各自的 `deprecated/` 目录，不得混入本协议。
