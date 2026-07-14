# Active Experiment Configs

顶层只保留 WebShop 当前实验的有效配置：

- `common.yaml`、`webshop.yaml`: 公共执行与 benchmark 配置。
- `evaluation.yaml`: 统一配对评估与 task bootstrap 配置。
- `webshop_no_skill.yaml`、`webshop_manual_skill.yaml`: 无训练池 baseline。
- `webshop_global_e2e.yaml`、`webshop_skillx.yaml`: 自动 baseline。
- `webshop_trace2tower.yaml`: 原始概念稿 Full Tower 构建配置。
- `webshop_trace2tower_runtime.yaml`: Full Tower 执行配置；直接 Mid cap 必须由运行命令显式指定为 3、5 或 8。
- `webshop_semantic_clustering.yaml`、`webshop_semantic_clustering_runtime.yaml`: 去掉 EigenTrace 图、谱分解和 High 的图结构消融。
- `webshop_trace2tower_no_event.yaml`、`webshop_trace2tower_no_event_runtime.yaml`: 使用独立非事件分段输入的事件抽取消融；不得通过禁止跨事件连边实现。
- `webshop_trace2tower_mid_only.yaml`: 显式关闭 High 的消融。
- `webshop_trace2tower_no_mixed.yaml`、`webshop_trace2tower_no_mixed_runtime.yaml`: success-only 证据消融的构建与执行配置。
- `webshop_event_tower_v2.json`: 保留既有数据划分；其中旧硬分层条件矩阵已废弃。

`deprecated/legacy-v1/` 保存全部旧 YAML/JSON，不得用于新实验。
