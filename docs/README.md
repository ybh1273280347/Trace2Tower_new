# Trace2Tower 文档索引

本文档按证据用途组织仓库内文档。正式结论、当前实验协议和早期开发记录不可混用。

## 正式冻结证据

- [WebShop 完整实验报告](reports/webshop/complete-experiment-report.md)：当前论文 WebShop 主结果、消融、机制解释和结论边界。
- [WebShop 冻结清单](reports/webshop/freeze-manifest.json)：29 个条件、19,500 个 episode 的机器可验证清单。
- [WebShop 验证与正式测试口径](protocols/webshop-validation-and-final-test.md)：配置选择、样本边界和统计口径。
- [WebShop Random-300 报告](reports/webshop/final-random300-report.md)：正式 baseline 矩阵的原始汇总。

## 当前实验

- [ALFWorld 正式实验协议](protocols/alfworld-experiment.md)：AgentBench 指标、训练池、验证和测试边界。
- [WebShop 部署优化实验](protocols/webshop-deployment-optimization.md)：Pro exposure、Pareto lifecycle、success-only/mixed 优化和新 held-out 测试协议。

## 方法与实现契约

- [决策记录](decisions.md)：全局冻结决策和重要实验事实。
- [Pareto refinement](methods/pareto-refinement.md)：四维目标、非支配排序、证据审计和生命周期规则。
- [分段](methods/segmentation.md)、[图与谱分解](methods/graph-and-spectral.md)、[技能塔](methods/skill-tower.md)：Tower 构建主链。
- [检索](methods/retrieval.md)、[静态执行](methods/static-execution.md)、[执行矩阵](operations/execution-matrix.md)：在线执行与可比性契约。
- [评估](methods/evaluation.md)：结果聚合、bootstrap 与成本报告口径。

## Baseline

- [Flat Skill Summary](baselines/flat-skill-summary.md)
- [SkillX baseline](baselines/skillx.md)

SkillX 只使用成功轨迹。它与 mixed evidence 消融严格分离。

## 历史与诊断

- [模型 pilot](history/agent-model-pilot.md)、[Flash 轨迹池扩容](history/flash-pool-expansion.md)：早期模型选择和池扩容记录。
- [WebShop Mid-only](reports/webshop/mid-only-ablation.md)：High/Mid 机制定位的中间报告；正式结论以完整报告为准。

历史文档用于追溯实现和实验演进，不与冻结验证或测试结果合并统计。
