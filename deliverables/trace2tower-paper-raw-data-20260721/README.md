# Trace2Tower 实验原始数据包

生成日期：2026-07-21

本目录包含论文主实验相关的机器可读数据，共 32 个正式运行、
3,830 条逐任务结果。下文给出了每张表、每组图与原始运行的对应关系。

所有文件平铺在本目录。`*-main-table.json` 是主表汇总，`*.jsonl` 是逐任务结果，
其余 JSON 是现有的机器可读结果、聚合数据、任务族映射、Tower 快照和校验记录。
`MANIFEST.json` 给出文件大小与 SHA-256。

## 数据边界

逐任务文件只保留评分、步骤、token、技能 ID、上下文哈希等白名单字段，不包含任务
正文、observation、action trajectory 或模型原始回复。训练轨迹、评估轨迹、私有
manifest、`.env` 和密钥均未打包。没有纳入 ExpeL mini、ALFWorld Tower r1/r2、
诊断、废弃或无效运行；ALFWorld Tower 主结果只保留正式 r0。

## 使用

主表可直接读取 `main-tables.json`。逐任务配对分析按 `sample_id` 对齐各个
`*.jsonl`。Tower 完整快照为 `alfworld-p310-tower.json`。完整来源关系
见下文。

## 实验对应关系

### ALFWorld 主表

| 方法 | 逐任务文件 |
|---|---|
| No-Skill | `alfworld-test-v1-flash-noskill-r0.jsonl` |
| Expert-Crafted Skills | `alfworld-test-v1-flash-manual-event-policy-r0.jsonl` |
| Trace2Skill +Combined | `trace2skill-gpt54-p310-alfworld-test-r0.jsonl` |
| Trace2Skill +Error | `trace2skill-gpt54-p310-error-alfworld-test-r0.jsonl` |
| SkillX no-rewrite | `alfworld-test-v1-flash-skillx-global-p310-r0.jsonl` |
| ExpeL | `alfworld-test-expel-p310-flash-r0.jsonl` |
| Trace2Tower | `alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0.jsonl` |

### WebShop 主表

| 方法 | 逐任务文件 |
|---|---|
| No-Skill | `webshop-original-concept-v1-validation-flash-noskill-r1.jsonl` |
| SkillX P100 | `webshop-skillx-native-inference-p100-validation-flash-r1.jsonl` |
| ExpeL P100 | `webshop-validation-expel-p100-flash-r0.jsonl` |
| Trace2Skill +Combined | `trace2skill-gpt54-p100-webshop-validation-r0.jsonl` |
| Trace2Skill +Error | `trace2skill-gpt54-p100-error-webshop-validation-r0.jsonl` |
| Expert-Crafted Skills | `webshop-expert-crafted-skills-validation-flash-r0.jsonl` |
| Trace2Tower P100 | `webshop-alfworld-v17-replication-p100-validation-r0.jsonl` |

### ALFWorld 消融

- Full：`alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0.jsonl`。
- High-only：`alfworld-ablation-v1-plan-rewrite-high-only-flash-r0.jsonl`。
- No Transition：`alfworld-ablation-v3-formal-no-transition-flash-r0.jsonl`。
- No Outcome：`alfworld-ablation-v3-formal-no-outcome-flash-r0.jsonl`。
- No Contrastive：`alfworld-ablation-v3-formal-no-contrastive-flash-r0.jsonl`。
- Semantic-only 没有形成合规 High，因此没有在线结果。

### 部署优化

- Frozen v0：`alfworld-deployment-v1-gate-v0-r0.jsonl` 与
  `alfworld-deployment-v1-holdout-v0-noop-r0.jsonl`。
- TF-IDF Pareto：`alfworld-deployment-v2-dev-graph-full-r0.jsonl` 与
  `alfworld-deployment-v2-holdout-graph-full-r0.jsonl`。
- Embedding Pareto：`alfworld-deployment-v2-dev-embedding-graph-r0.jsonl` 与
  `alfworld-deployment-v2-holdout-embedding-graph-r0.jsonl`。

### 其他数据

- 现有聚合数据：`report-data.json`。
- Tower 完整快照：`alfworld-p310-tower.json`。
- ALFWorld 任务族标签：`alfworld-task-family-map.json`。
- 跨模型聚合数据：`cross-model-analysis-data.json`；对应逐任务结果是
  `cross-*`、`generalize-*` 以及 Flash 主运行 JSONL。
