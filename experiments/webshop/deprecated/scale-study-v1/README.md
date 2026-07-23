# WebShop Skill Scale Study

本实验研究技能迁移效果如何随训练轨迹池扩大而变化。固定协议为 `configs/experiments/webshop_scale_v1.json`。

## 数据与模型

- P50 复用已经审计的 50 个 train 任务和 200 条 Flash NoSkill 轨迹。
- P100 新增 50 个 train 任务和 200 条轨迹。
- P200 在 P100 基础上再新增 100 个 train 任务和 400 条轨迹。
- 三个池按显式 sample ID 严格嵌套，训练均为 Flash、4 repeats、20 步上限。
- 评估固定一个新 seed 选择的 100 个 test 任务，统一使用 Pro、3 repeats。

## 构建与指标

每档分别构建 Flat Success-only、SkillX Success-only、Tower Success-only 和 Tower Mixed。构建报告只比较 builder chat 输入/输出 token、embedding token、最终技能数量、图路径支持和聚类稳定性；不报告依赖设备与并发策略的构建用时。

评估共 13 个条件：共享 NoSkill，加三档池各四种技能方法。主指标为 mean reward 和满分成功率，同时报告完成率、步骤、无效动作和 agent chat token。

## 阶段门槛

先完成 P50/P100 构建与评估。只有下列信号至少出现一个，才将 `p200-decision.template.json` 复制为 `p200-decision.json` 并记录 `continue: true`：

- Tower 相对 Flat 的差距改善；
- Mixed 的负效应缩小；
- 图路径支持或聚类稳定性改善；
- 构建 chat 成本或技能压缩曲线优于 Flat/SkillX。

启动器会拒绝没有该决策文件的 P200 采集、构建或评估。

## 命令

```powershell
uv run python -m scripts.experiments.run.run_webshop_scale collect --pool p100
uv run python -m scripts.experiments.run.run_webshop_scale materialize --pool p100
uv run python -m scripts.experiments.run.run_webshop_scale build --pool p100
uv run python -m scripts.experiments.run.run_webshop_scale evaluate --pool p100
```

所有凭据和模型 endpoint 仅从被 Git 忽略的 `.env` 读取。

## 人工 Skill 诊断

`diagnostics/manual-skill-v1.md` 与 `diagnostics/manual-skill-v2.md` 绕过检索直接注入 Pro agent。完整 100 tasks × 3 repeats 的结果与配对 CI 见 `docs/reports/webshop/manual-skill-diagnostic.md`。该诊断用于区分注入链路问题与 skill 内容/展开问题，不参与 P50/P100/P200 默认方法排名。
