# ALFWorld G2-G4 Rewrite 审计

## 结论

此前 `ALFWORLD_BUILD_ABLATION_RESULTS.json` 中的 G2/G3/G4 运行解析为
`retrieval_contract: high_to_mid` + `rewrite_plan: true`。该路由进入
`HighToMidSkillProvider`，其 rewrite 调用的是 `NativeSkillXInference`，因此这些运行不能作为
Trace2Tower 正式 rewrite 消融的证据，只保留为历史诊断。

本次补跑统一使用 `PlanRewriteTrace2TowerProvider`，合同为
`retrieval_strategy: plan_rewrite` + `rewrite_contract_version: budgeted_v2`。该 provider 在
`run_matrix` 中对缺省开关使用 `rewrite_plan=true`，因此三项运行实际执行了正式 rewrite；
新增的 `alfworld_ablation_formal_rewrite_runtime.yaml` 将该开关显式写出，供后续复现。
三项运行使用同一 `alfworld_test.jsonl`、同一 agent model、同一 repeat，
仅替换对应构建消融 Tower。

## 正式补跑

| 消融 | Run | 成功 | 成功率 |
| --- | --- | ---: | ---: |
| G2 No Transition | `artifacts/runs/alfworld-ablation-v3-formal-no-transition-flash-r0` | 94/134 | 70.15% |
| G3 No Outcome | `artifacts/runs/alfworld-ablation-v3-formal-no-outcome-flash-r0` | 99/134 | 73.88% |
| G4 No Contrastive | `artifacts/runs/alfworld-ablation-v3-formal-no-contrastive-flash-r0` | 99/134 | 73.88% |

完整统计见 `ALFWORLD_BUILD_ABLATION_FORMAL_REWRITE_RESULTS.json`。三份 run 的
`resolved-config.yaml` 均记录 `retrieval_strategy: plan_rewrite` 和
`rewrite_contract_version: budgeted_v2`；代码路径对该合同的 rewrite 默认值为 true。
