# Experiment Scripts

当前脚本按职责分为四类：

- `data/`: manifest、轨迹池、证据池和数据审计。
- `build/`: SkillX 与 Trace2Tower artifact 构建。
- `run/`: No Skill、SkillX、Expert-Crafted Skills 和 Trace2Tower 执行入口。
- `analyze/`: 当前结果聚合、配对比较和机制分析。

统一执行入口是：

```powershell
uv run python -m scripts.experiments.run.run_matrix --help
```

Trace2Tower 构建链为：

```text
preprocess_trajectories
  -> build_trace2tower_graph
  -> build_trace2tower_skills
  -> build_trace2tower_index
  -> build_tower_snapshot
```

SkillX 使用 `build_skillx_index` 生成执行 library。Expert-Crafted Skills 直接绑定冻结的
技能文本，不构建 artifact。Expel 目前仅有方法枚举占位，尚无构建或执行脚本。

`deprecated/` 保存已冻结的历史协议与一次性实验脚本，不得作为新实验入口。
