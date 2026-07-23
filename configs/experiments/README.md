# Active Experiment Configs

顶层只保留当前可执行方法、Trace2Tower 构建消融和 benchmark 公共配置。

- `common.yaml`、`alfworld.yaml`、`webshop.yaml`: 公共执行与 benchmark 配置。
- `evaluation.yaml`: 统一配对评估配置。
- `webshop_no_skill.yaml`: No Skill baseline。
- `webshop_expert_crafted_skills.yaml`: Expert-Crafted Skills baseline。
- `alfworld_expel.yaml`、`webshop_expel.yaml`: ExpeL baseline。
- `alfworld_skillx.yaml`、`webshop_skillx.yaml`: SkillX baseline。
- `alfworld_trace2tower_runtime.yaml`: ALFWorld Trace2Tower 执行配置。
- `webshop_trace2tower_alfworld_isomorphic*.yaml`: WebShop Trace2Tower 构建与执行配置。
- `alfworld_ablation_*.yaml`: ALFWorld 构建和部署消融。
- `webshop_trace2tower_constraint_branch_v1*.yaml`: WebShop 后续约束分支实验。

Expel 当前只在 `MethodName` 中保留枚举占位；provider、artifact schema 和运行配置将在
正式接入时一并添加，现阶段不能通过 `run_matrix.py` 执行。

`deprecated/` 保存旧配置，不得用于新实验。
