# WebShop Event Tower V2 实验清单

> **已废弃：** 本清单属于被否决的硬事件分层实现。当前进度与结果见 `../original-concept-v1/`。
> 下方 No-event 和独立消融数据集任务不得继续执行；有效消融仅按当前 original-concept 协议运行。

## 阶段 1：审计两个轨迹池

- [x] 验证 P50 为 50 tasks x 4 repeats = 200 条轨迹。
- [x] 验证 P100 为 100 tasks x 4 repeats = 400 条轨迹，且 P50 严格包含于 P100。
- [x] 记录 source run、任务 ID、repeat、轨迹哈希、reward 分布和错误覆盖。
- [x] 单独 commit 并 push 池审计清单；不重新采样 validation/test。

冻结证据：`stage-1-pools/audit.json`，审计 ID `poolaudit_e4b43e1d73e97aad`。P50/P100 mean reward 分别为 `0.673292`/`0.682771`，满分轨迹为 `94`/`186`，source run error attempts 均为 0；主训练池、独立 ablation-train、validation/test/ablation 各自满足预注册的不重叠边界。

## 阶段 2：构建 P50 技能

- [x] 构建或严格审计复用 P50 Global E2E artifact。
- [x] 构建或严格审计复用 P50 SkillX artifact；运行时固定原生 cap8。
- [x] 重建 P50 Full Event Tower mixed snapshot。
- [x] 重建 P50 Semantic Clustering mixed baseline snapshot。
- [x] 重建 P50 No-mixed event-aware success-only snapshot。
- [x] Mid-only 绑定 Full snapshot，不单独构建。
- [x] 冻结 Manual Skill 文本及全部 artifact hashes，单独 commit 并 push。

冻结证据：`stage-2-skills/audit.json`，审计 ID `skillaudit_fbbf5a8fb72b2ee6`。Full、Semantic、No-mixed snapshot IDs 分别为 `tower_c69572a30d550032`、`tower_82fb7eb6f8ece365`、`tower_09e1e3351cc8e7ce`。

P100 的新 Tower 构建推迟到阶段 8。这样 cap 在 P50 上冻结之后才做规模扩展，且 Flash 门控失败时不浪费 Flash 侧规模实验成本。

## 阶段 3：Validation 冻结检索 cap

- [x] 分别运行 P50 Full Tower 和 Semantic Clustering 的 cap3、cap5、cap8。
- [x] Flash 六个条件，共 1,800 episodes。
- [x] Pro 六个条件，共 1,800 episodes。
- [x] 按 `PROTOCOL.md` 的规则为两个方法各冻结一个跨模型共同 cap。
- [x] 将所选 cap、比较表、resolved configs 和结果哈希单独 commit 并 push。

冻结证据：`stage-3-validation/audit.json`，审计 ID `validation_58700e4f11d40e27`。Semantic Clustering 与 Full Trace2Tower 的经验最优均为 cap5，但 cap3 相对经验最优的 95% 配对 task-bootstrap 区间均包含 0，因此按预注册的最小等价 cap 规则，两者都冻结为 cap3。

SkillX 保留原生 cap8，不参加选择。No-event、Mid-only 和 No-mixed 均不进入 validation。

## 阶段 4：P50 Baseline 与完整方法

- [ ] 在 test 上运行 NoSkill、Manual、P50 Global E2E、P50 SkillX、P50 Semantic Clustering、P50 Full Tower。
- [ ] Flash 与 Pro 各 6 conditions x 300 episodes = 1,800，共 3,600 episodes。
- [ ] Semantic Clustering 和 Tower 使用各自在阶段 3 冻结的 cap，禁止按模型重新选择。
- [ ] 完成覆盖与 artifact binding 审计后，单独 commit 并 push。

## 阶段 5：Flash 门控

- [ ] 在 Flash 上比较 Manual、Global E2E、SkillX、Semantic Clustering、Full Tower 各自相对 NoSkill 的 mean reward。
- [ ] 对五个配对 task-bootstrap 检验做 Holm 校正，family-wise `alpha=0.05`。
- [ ] 至少一个方法显著正向则 `flash_gate=pass`，否则为 `fail`。
- [ ] 冻结门控判定和输入结果哈希，单独 commit 并 push。

门控失败时保留并报告阶段 4 的全部 Flash 主结果，但后续 Flash 消融和 P100 不运行。

## 阶段 6：独立训练/测试消融

- [ ] 使用 seed `20260718` 从排除主 P100 tasks 的 train indices `1000..11999` 随机冻结 100 个 ablation-train tasks。
- [ ] 每个 ablation-train task 跑 4 个 Flash No-Skill repeats，形成独立的 400-trajectory pool。
- [ ] 在查看任何消融结果前同时冻结 ablation-train pool 与 seed `20260717` 的 100-task ablation test manifest。
- [ ] 禁止根据 High 数量、是否存在 High、簇数或预跑 reward 重采样训练池；无 High 也作为结果报告。
- [ ] 从独立训练池重建 Full、No-event、No-mixed；Mid-only 复用该 Full，不使用阶段 2 的 P50 No-mixed 作正式证据。
- [ ] Pro 必跑 Full、No-event、Mid-only、No-mixed，共 1,200 episodes。
- [ ] 只有 `flash_gate=pass` 时才在 Flash 跑同四项，再增加 1,200 episodes。
- [ ] 四项均使用 Full Tower 在阶段 3 冻结的 cap；消融不重新调参。
- [ ] 分别回答事件分层、High 诱导、mixed/负证据是否必要。
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
| Validation cap sweep | 3,600 | 0 |
| P50 baseline + Full | 3,600 | 0 |
| Ablation training pool | 400 | 0 |
| Pro P50 ablations | 1,200 | Flash gate pass 再加 1,200 |
| Pro P100 scale | 900 | Flash gate pass 再加 900 |
| 总计 | 9,700 | 最多 11,800 |

纯语义聚类同时移除了事件分层、signed graph 和 High，因此只作为 baseline。No-event 才是单变量事件分层消融，并和其他消融一起放到独立的 ablation tasks 上。规模问题仍只比较三种自动方法，机制问题只在 P50 解决。
