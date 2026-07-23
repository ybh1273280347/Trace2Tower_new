# Trace2Tower 实验原始数据包

生成日期：2026-07-21

本目录包含论文主实验相关的机器可读数据，共 32 个正式运行、
3,830 条逐任务结果。下文给出了每张表、每组图与原始运行的对应关系。

所有文件平铺在本目录。`*.jsonl` 是逐任务结果，另外只有 ALFWorld Tower 快照和
任务族映射。

逐任务文件按 `sample_id` 对齐。ALFWorld Tower 只保留主实验结果，不包含补充探索结果。

## 实验对应关系

### ALFWorld 主表

| 方法 | 逐任务文件 |
|---|---|
| No-Skill | `alfworld-main-noskill.jsonl` |
| Expert-Crafted Skills | `alfworld-main-expert-crafted.jsonl` |
| Trace2Skill +Combined | `alfworld-main-trace2skill-combined.jsonl` |
| Trace2Skill +Error | `alfworld-main-trace2skill-error.jsonl` |
| SkillX | `alfworld-main-skillx.jsonl` |
| ExpeL | `alfworld-main-expel.jsonl` |
| Trace2Tower | `alfworld-main-trace2tower.jsonl` |

### WebShop 主表 j  

| 方法 | 逐任务文件 |
|---|---|
| No-Skill | `webshop-main-noskill.jsonl` |
| SkillX P100 | `webshop-main-skillx.jsonl` |
| ExpeL P100 | `webshop-main-expel.jsonl` |
| Trace2Skill +Combined | `webshop-main-trace2skill-combined.jsonl` |
| Trace2Skill +Error | `webshop-main-trace2skill-error.jsonl` |
| Expert-Crafted Skills | `webshop-main-expert-crafted.jsonl` |
| Trace2Tower P100 | `webshop-main-trace2tower.jsonl` |

### ALFWorld 消融

- Full：复用 `alfworld-main-trace2tower.jsonl`。
- High-only：`alfworld-ablation-high-only.jsonl`。
- No Transition：`alfworld-ablation-no-transition.jsonl`。
- No Outcome：`alfworld-ablation-no-outcome.jsonl`。
- No Contrastive：`alfworld-ablation-no-contrastive.jsonl`。
- Semantic-only 没有形成合规 High，因此没有在线结果。

### 部署优化

- Frozen：`alfworld-deployment-test1-frozen.jsonl` 与
  `alfworld-deployment-test2-frozen.jsonl`。
- TF-IDF Pareto：`alfworld-deployment-test1-tfidf-pareto.jsonl` 与
  `alfworld-deployment-test2-tfidf-pareto.jsonl`。
- Embedding Pareto：`alfworld-deployment-test1-embedding-pareto.jsonl` 与
  `alfworld-deployment-test2-embedding-pareto.jsonl`。

### 其他数据

- Tower 完整快照：`alfworld-main-trace2tower-snapshot.json`。
- ALFWorld 任务族标签：`alfworld-task-family-map.json`。
- 跨模型逐任务结果：`alfworld-cross-*`、`webshop-cross-*` 以及 Flash 主运行 JSONL。
