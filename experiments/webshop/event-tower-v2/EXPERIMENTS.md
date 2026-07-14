# WebShop Event Tower V2 实验清单

## 阶段 1：审计两个轨迹池

- [ ] 验证 P50 为 50 tasks x 4 repeats = 200 条轨迹。
- [ ] 验证 P100 为 100 tasks x 4 repeats = 400 条轨迹，且 P50 严格包含于 P100。
- [ ] 记录 source run、任务 ID、repeat、轨迹哈希、reward 分布和错误覆盖。
- [ ] 单独 commit 并 push 池审计清单；不重新采样 validation/test。

## 阶段 2：构建 P50 技能

- [ ] 构建或严格审计复用 P50 Global E2E artifact。
- [ ] 构建或严格审计复用 P50 SkillX artifact；运行时固定原生 cap8。
- [ ] 重建 P50 Full Event Tower mixed snapshot。
- [ ] 重建 P50 Semantic-only mixed snapshot。
- [ ] 重建 P50 No-mixed event-aware success-only snapshot。
- [ ] Mid-only 绑定 Full snapshot，不单独构建。
- [ ] 冻结 Manual Skill 文本及全部 artifact hashes，单独 commit 并 push。

P100 的新 Tower 构建推迟到阶段 8。这样 cap 在 P50 上冻结之后才做规模扩展，且 Flash 门控失败时不浪费 Flash 侧规模实验成本。

## 阶段 3：Validation 冻结 Tower cap

- [ ] 只运行 P50 Full Tower 的 cap3、cap5、cap8。
- [ ] Flash 三档各 100 tasks x 3 repeats，共 900 episodes。
- [ ] Pro 三档各 100 tasks x 3 repeats，共 900 episodes。
- [ ] 按 `PROTOCOL.md` 的配对 task-bootstrap 规则选出一个共同 cap。
- [ ] 将所选 cap、比较表、resolved configs 和结果哈希单独 commit 并 push。

SkillX 保留原生 cap8，不参加选择。No-mixed 是负证据消融，也不进入 validation。

## 阶段 4：P50 Baseline 与完整方法

- [ ] 在 test 上运行 NoSkill、Manual、P50 Global E2E、P50 SkillX、P50 Full Tower。
- [ ] Flash 与 Pro 各 5 conditions x 300 episodes = 1,500，共 3,000 episodes。
- [ ] Tower 使用阶段 3 冻结的 cap，禁止按模型重新选择。
- [ ] 完成覆盖与 artifact binding 审计后，单独 commit 并 push。

## 阶段 5：Flash 门控

- [ ] 在 Flash 上比较 Manual、Global E2E、SkillX、Full Tower 各自相对 NoSkill 的 mean reward。
- [ ] 对四个配对 task-bootstrap 检验做 Holm 校正，family-wise `alpha=0.05`。
- [ ] 至少一个方法显著正向则 `flash_gate=pass`，否则为 `fail`。
- [ ] 冻结门控判定和输入结果哈希，单独 commit 并 push。

门控失败时保留并报告阶段 4 的全部 Flash 主结果，但后续 Flash 消融和 P100 不运行。

## 阶段 6：P50 消融

- [ ] Pro 必跑 Semantic-only、Mid-only、No-mixed，共 900 episodes。
- [ ] 只有 `flash_gate=pass` 时才在 Flash 跑同三项，再增加 900 episodes。
- [ ] 三项均使用阶段 3 的冻结 cap；No-mixed 不重新调参。
- [ ] 分别回答 Event-aware signed Mid、High 诱导、mixed/负证据是否必要。
- [ ] 单独 commit 并 push 消融结果和配对统计。

## 阶段 7：冻结 P50 结论

- [ ] 汇总 baseline、Full Tower、消融、效率和失败模式。
- [ ] 明确区分 confirmatory 比较与诊断，不根据 test 修改配置。
- [ ] 冻结 P50 报告、表格输入和所有哈希，单独 commit 并 push。

## 阶段 8：P100 规模实验

- [ ] 构建或审计复用 P100 Global E2E、SkillX，并重建 P100 Full Event Tower。
- [ ] Pro 必跑 P100 Global E2E、SkillX、Full Tower，共 900 episodes。
- [ ] 只有 `flash_gate=pass` 时才在 Flash 跑同三项，再增加 900 episodes。
- [ ] 同方法、同模型、同 test keys 配对比较 P100 相对 P50。
- [ ] Tower 沿用阶段 3 的 cap，不在 P100 重新调参。
- [ ] 单独 commit 并 push 最终规模结果。

## 预算汇总

| 部分 | 固定 episodes | 条件增加 |
|---|---:|---:|
| Validation cap sweep | 1,800 | 0 |
| P50 baseline + Full | 3,000 | 0 |
| Pro P50 ablations | 900 | Flash gate pass 再加 900 |
| Pro P100 scale | 900 | Flash gate pass 再加 900 |
| 总计 | 6,600 | 最多 8,400 |

相对最初把 P50/P100 全方法铺成 14 条平面矩阵，这个顺序减少了无用的 P100 消融：规模问题只比较三种自动方法，机制问题只在 P50 解决；同时保留两模型 cap 验证，避免把 Pro 选出的检索预算未经验证地迁移到 Flash。
