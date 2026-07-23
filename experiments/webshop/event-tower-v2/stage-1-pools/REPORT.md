# Stage 1: WebShop 训练轨迹池审计

状态：`complete`
审计 ID：`poolaudit_e4b43e1d73e97aad`

## 结论

P50 和 P100 均满足 Event Tower V2 训练池契约。P50 的 200 条轨迹在 P100 中逐记录一致；P100 只新增 50 个任务和 200 条轨迹。两个池均为 Flash No-Skill train rollout，source metadata 中没有 error attempt 或 failed invocation。

| 池 | Tasks | Repeats | Trajectories | Mean reward | Full success | Partial | Zero | System failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P50 | 50 | 4 | 200 | 0.673292 | 94 | 77 | 29 | 0 |
| P100 | 100 | 4 | 400 | 0.682771 | 186 | 161 | 53 | 0 |

## 冻结输入

- P50 SHA-256：`7d8ae154746c398b1084acb130d8d1a13760b3be43e2d71ff0aabf60de9068f9`
- P100 SHA-256：`7a4377cac314e62e4382e2cb55603b28b670f7959013e20df663af0525bbb92f`
- P50 source run：`webshop-flash50-repeat4-pool-v1`
- P100 additional source run：`webshop-scale-v1-flash-p100-add50`
- Validation/Test selection：`selection_fa882612a3fbe29f`
- Ablation selection：`selection_50a063e627f4cd79`
- Ablation-train selection：`selection_6594d51ebe102927`

## 不变量

- P50 task/repeat coverage：完整的 50 x 4 笛卡尔积。
- P100 task/repeat coverage：完整的 100 x 4 笛卡尔积。
- P50 task set 是 P100 task set 的严格子集。
- P50 对应 trajectory records 在 P100 中完全相同。
- 训练任务与 validation/test/ablation indices `0..999` 零重叠。
- 独立 ablation-train tasks 与主实验 P100 tasks 零重叠。
- 本阶段只读取并审计已有轨迹，没有重新 rollout，也没有修改 validation/test manifests。

完整 sample IDs、reward histogram、finish reasons、source metadata hashes 和机器可验证不变量
见 `audit.json`。
