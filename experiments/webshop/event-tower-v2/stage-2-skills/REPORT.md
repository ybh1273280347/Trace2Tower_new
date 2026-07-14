# Stage 2: P50 技能 Artifact 冻结

状态：`complete`
审计 ID：`skillaudit_62e9fd8bde33bb9f`

## Baseline artifacts

| 方法 | Artifact ID | 训练轨迹 | 内容 |
|---|---|---:|---:|
| Global E2E GPT | `flatcorpus_cba9470f206835ef` | 94 | 4 cards |
| SkillX | `skillxlib_409dd86005b242ca` | 94 | 26 plans + 2 atomic skills |

Global E2E 复用的是已完成的 GPT-5.4 end-to-end induction。审计把旧 prompt blob 固定到 git revision `db29a80`，并验证 prompt SHA、P50 的 94 条成功轨迹、corpus SHA、卡片和 embedding index。它作为 `global_e2e_gpt` 执行，不恢复旧方法名。

SkillX 复用官方上游构建结果，绑定 commit `36747f424a17ea041e476adf2ff976a206ec9c30`、94 条成功轨迹和 source library SHA；运行时固定原生 `max_skills=8`。

## 新构建 snapshots

| 方法 | Snapshot ID | 训练轨迹 | Mid | High | Event stratification |
|---|---|---:|---:|---:|---|
| Full Trace2Tower | `tower_c69572a30d550032` | 173 | 19 | 17 | true |
| Semantic Clustering | `tower_82fb7eb6f8ece365` | 173 | 19 | 0 | false |
| No-mixed | `tower_09e1e3351cc8e7ce` | 94 | 16 | 26 | true |

Full 与 Semantic Clustering 使用同一 173 条 mixed evidence；No-mixed 只使用 94 条满分成功轨迹。Semantic Clustering 不包含 signed graph 或 High，因此只作为 baseline。No-event snapshot 延后到独立消融阶段构建。

Mid-only 不生成新 artifact，执行时复用 Full snapshot `tower_c69572a30d550032` 并设置 `include_high=false`。Manual Skill SHA-256 为 `5d901247b383634ae557aa08298a3244f32204c74a0a340c22b4d37a843d2dc4`。

完整路径、组件哈希、配置哈希、renderer token 和 artifact SHA-256 见 `audit.json`。
