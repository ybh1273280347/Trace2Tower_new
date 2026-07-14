# Active Experiment Configs

顶层只保留 WebShop Event Tower V2 的有效配置：

- `common.yaml`、`webshop.yaml`: 公共执行与 benchmark 配置。
- `evaluation.yaml`: 统一配对评估与 task bootstrap 配置。
- `webshop_no_skill.yaml`、`webshop_manual_skill.yaml`: 无训练池 baseline。
- `webshop_global_e2e.yaml`、`webshop_skillx.yaml`: 自动 baseline。
- `webshop_trace2tower.yaml`: Full event-aware Tower 构建配置。
- `webshop_trace2tower_runtime.yaml`: Full Tower 执行配置；直接 Mid cap 必须由运行命令显式指定为 3、5 或 8。
- `webshop_semantic_clustering.yaml`、`webshop_semantic_clustering_runtime.yaml`: 纯语义聚类 baseline 的构建与执行配置。
- `webshop_trace2tower_no_event.yaml`、`webshop_trace2tower_no_event_runtime.yaml`: 只关闭事件分层的 signed-graph 消融。
- `webshop_trace2tower_mid_only.yaml`: 显式关闭 High 的消融。
- `webshop_trace2tower_no_mixed.yaml`、`webshop_trace2tower_no_mixed_runtime.yaml`: success-only 证据消融的构建与执行配置。
- `webshop_event_tower_v2.json`: 固定数据划分、条件矩阵与 artifact policy。

`deprecated/legacy-v1/` 保存全部旧 YAML/JSON，不得用于新实验。
