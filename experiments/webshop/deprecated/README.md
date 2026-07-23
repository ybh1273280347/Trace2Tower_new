# Deprecated WebShop Experiments

本目录保存 2026-07-14 之前的 WebShop 实验材料，仅用于追溯，不参与当前排名、调参或结论。

## 废弃原因

旧预处理已经抽取 WebShop 事件并按事件连续段切分轨迹，但主规模实验没有在 Mid 建图和聚类时启用 `event_type_stratification`。因此旧 Tower 允许查询、商品选择、属性检查、回退和购买段进入同一 Mid cluster，不能代表当前定义的 Trace2Tower。

## 内容

- `scale-study-v1/`: 旧 P50/P100 scale-study 协议、Manual 诊断和 baseline freeze。
- `reports-v1/`: 旧 random-300、Mid-only、完整实验报告和 freeze manifest。
- `scripts-experiments-readme-v1.md`: 旧实验命令说明。

相应运行结果仍位于 `artifacts/`。它们只能被标注为 deprecated evidence。旧 Global E2E、SkillX 和 Manual artifact 在满足新协议的训练池与内容哈希时可以复用；所有旧 Tower snapshot 必须重建。
