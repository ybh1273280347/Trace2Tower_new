# Active Experiment Configs

顶层只保留 WebShop Event Tower V2 的有效配置：

- `common.yaml`、`webshop.yaml`: 公共执行与 benchmark 配置。
- `evaluation.yaml`: 统一配对评估与 task bootstrap 配置。
- `webshop_no_skill.yaml`、`webshop_manual_skill.yaml`: 无训练池 baseline。
- `webshop_global_e2e.yaml`、`webshop_skillx.yaml`: 自动 baseline。
- `webshop_trace2tower.yaml`: Full event-aware Tower 构建配置。
- `webshop_trace2tower_runtime.yaml`: Full Tower 执行配置。
- `webshop_trace2tower_semantic_only.yaml`: 纯语义聚类消融。
- `webshop_trace2tower_mid_only.yaml`: 显式关闭 High 的消融。
- `webshop_trace2tower_no_mixed.yaml`: success-only 证据消融。
- `webshop_event_tower_v2.json`: 固定数据划分、条件矩阵与 artifact policy。

`deprecated/legacy-v1/` 保存全部旧 YAML/JSON，不得用于新实验。
