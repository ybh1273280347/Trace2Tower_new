# Trace2Tower 主表与图像数据包

生成日期：2026-07-21

本目录包含主实验报告全部主表与 13 张图所需的数据，共 34 个正式运行、
4,098 条逐任务结果。建议先看 `SOURCE_MAP.json`，其中给出了每张表、每张图与
原始运行的对应关系。

## 目录

- `tables/`：从逐任务结果重新聚合的 ALFWorld/WebShop 主表。
- `task_level/runs/`：每个正式运行一份 JSONL，按 sample_id 排序。
- `figure_data/`：报告图的聚合数据、ALFWorld 任务族映射和 ExpeL mini 图数据。
- `formal_results/`：消融、部署、baseline 审计等机器可读正式记录。
- `tower/`：生成结构图、压缩图和 embedding 投影所用的 ALFWorld P310 Tower。
- `scripts/`：原始绘图脚本。
- `rendered_figures/`：当前报告使用的 PNG/PDF，便于核对。
- `docs/`：主报告、图表说明和跨模型分析。
- `verification.json`：主表结果与报告数字的自动复算核对。
- `MANIFEST.json`：所有文件的大小和 SHA-256。

## 数据边界

逐任务文件只保留评分、步骤、token、技能 ID、上下文哈希等白名单字段，不包含任务
正文、observation、action trajectory 或模型原始回复。训练轨迹、评估轨迹、私有
manifest、`.env` 和密钥均未打包。`expel-mini` 是 FIGURES.md 明确标注的补充探索图，
不属于主实验结论；除此之外没有纳入诊断、废弃或无效运行。

## 使用

主表可直接读取 `tables/main-tables.json`。逐任务配对分析按 `sample_id` 对齐
`task_level/runs/*.jsonl`。结构图使用 `tower/alfworld-p310-tower.json`。原脚本中的
仓库路径需要改为本目录对应路径；完整来源关系见 `SOURCE_MAP.json`。
