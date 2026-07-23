# Stage 3: Validation Cap 冻结

状态：`complete`
审计 ID：`validation_58700e4f11d40e27`

## 覆盖

12 个条件均完成 100 tasks x 3 repeats，共 3,600 个 official episodes。每个条件覆盖率均为 1.0，结果键完全一致。Validation 只用于分别选择 Semantic Clustering 和 Full Trace2Tower 的直接 Mid cap。

## 条件结果

| 方法 | 模型 | Cap | Mean reward | Full success | Mean steps | Mean chat input tokens |
|---|---|---:|---:|---:|---:|---:|
| semantic_clustering | flash | 3 | 0.668100 | 0.470 | 8.233 | 26480.8 |
| semantic_clustering | pro | 3 | 0.648967 | 0.490 | 9.277 | 31018.8 |
| semantic_clustering | flash | 5 | 0.678717 | 0.483 | 8.367 | 30523.5 |
| semantic_clustering | pro | 5 | 0.671833 | 0.490 | 8.633 | 31860.0 |
| semantic_clustering | flash | 8 | 0.688600 | 0.503 | 7.827 | 28881.2 |
| semantic_clustering | pro | 8 | 0.660922 | 0.483 | 8.483 | 33127.6 |
| trace2tower | flash | 3 | 0.639817 | 0.447 | 7.520 | 23731.3 |
| trace2tower | pro | 3 | 0.653028 | 0.473 | 8.743 | 28849.9 |
| trace2tower | flash | 5 | 0.665922 | 0.470 | 7.493 | 23840.7 |
| trace2tower | pro | 5 | 0.629861 | 0.460 | 9.253 | 33199.9 |
| trace2tower | flash | 8 | 0.634233 | 0.453 | 7.550 | 24688.8 |
| trace2tower | pro | 8 | 0.625817 | 0.443 | 9.207 | 33551.1 |

## 冻结选择

| 方法 | cap3 | cap5 | cap8 | 经验最优 | 冻结 cap |
|---|---:|---:|---:|---:|---:|
| semantic_clustering | 0.658533 | 0.675275 | 0.674761 | 5 | **3** |
| trace2tower | 0.646422 | 0.647892 | 0.630025 | 5 | **3** |

选择规则按 task 聚合：先平均 3 repeats，再平均 Flash/Pro。以经验均值最优 cap 为参照，对“最优减候选”的 task differences 做 10,000 次配对 bootstrap；选择 95% 区间包含 0 的最小 cap。完整区间、结果哈希、resolved config 哈希和运行 metadata 见 `audit.json`；机器冻结结果见 `selected-caps.json`。
