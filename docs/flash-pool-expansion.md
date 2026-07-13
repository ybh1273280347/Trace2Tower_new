# WebShop Flash Pool Expansion

## Extension Contract

The original five-episode `deepseek-v4-flash` pilot was extended in place to 20 episodes with `scripts/experiments/extend_no_skill_pool.py`. The extension preserves the original run ID, model, benchmark, method, shard assignment, agent execution settings, and trajectory pool. Before making provider calls it audits the existing prefix, checks the immutable pilot configuration hash, and requires the requested target to grow within the selected shard. After execution it requires exact result and trajectory coverage before writing an extension report.

The shard-0 extension selected 20 WebShop training episodes, skipped the five already complete episodes, completed 15 new episodes, and recorded no failures or coverage gaps. The resulting trajectory pool SHA-256 is `7dfc8680c08c33bf90908d8a5b871a1009a9bd44b8dc65bf49e1e92bc9e82136`.

## Pool Quality

The 20 trajectories contain 13 full successes and have mean reward `0.8292`. Completion rate is `0.95`, valid-action rate is `0.9710`, mean step count is `6.9`, and mean successful step count is `6.38`. Mean repeated-action rate is `0.0132`; every consecutive observation changed. Reported token use is 333,628 input and 5,459 output tokens, or 26,083.6 total reported tokens per full success.

These values support using the expanded pool for another bounded Tower build. They do not establish benchmark-level model performance because the episodes are one deterministic shard prefix.

## Rebuilt Tower

The expanded pool produced 138 transitions, 125 segments, seven Mid clusters, six High paths, seven Mid cards, and six High cards. An immediate renderer rerun reused all 13 cards with zero new model calls, and an immediate index rerun reused all 13 embeddings. Rebuilding the snapshot produced the same bytes and content-addressed identity:

- Snapshot: `tower_999171fa3b9f880b`
- SHA-256: `0eb21f9bdfeefc4809febaab97cc334e7ef08e9fe66f95881560d9f3080ab287`
- Training provenance: 20 trajectories

## Pool-External Gate

Fresh paired runs used WebShop training samples `1001` and `1002`, neither of which contributed to the Tower. No-Skill scored `1.0` on both. Static Tower scored `1.0` and `0.3333`, for mean reward `0.6667` versus `1.0`; the paired reward difference was `-0.3333` with a two-pair bootstrap interval of `[-0.6667, 0.0]`. Static also used 2.5 more steps and 13,481 more input tokens per episode on average.

Both tasks retrieved `high_a4b61e1af12c`. Its High path is supported by one successful training trajectory, repeats a Mid child in the order `mid_0003 -> mid_0000 -> mid_0003`, and renders to seven substantially overlapping strategy steps. The observed High cosine similarities were `0.4656` and `0.5062`, below the corresponding direct-Mid matches (`0.6180` through `0.6663`). All six High paths in this build have only one supporting successful trajectory.

The expanded Tower therefore still fails the bounded quality gate and is not eligible for full rollout. Samples `1001` and `1002` are treated as diagnostic calibration only and are not reported as held-out evidence.

## High Confidence Gate Ablation

Retrieval now supports an explicit High cosine-similarity threshold while preserving the rejected Top-1 candidate separately from the accepted match in diagnostic reports. Existing behavior is represented by threshold `-1.0`; no stored Tower schema or identity changed. A calibrated threshold of `0.55` rejected High on samples `1001` and `1002` and reduced injected context from roughly 5.5k to 2.5-2.6k characters without changing direct-Mid ranking.

The actual ablation used fresh pool-external training samples `1003` and `1004`. Against the same No-Skill pair, both gated and ungated Static scored mean reward `0.75` versus `0.5`, with one win and one tie. The gated version averaged 9 steps and 21,147 reported input tokens; the ungated version averaged 7 steps and 19,106.5 input tokens. Both used one invalid action across the pair and neither provider reported billable tokens.

The threshold therefore did not improve held-out reward and worsened observed step and input-token efficiency in this two-task diagnostic. It remains an executable ablation control, but the Static default stays `-1.0` rather than promoting the calibrated `0.55`. This sample is too small for a performance claim; its purpose is to reject an unsupported default change.

## Fifty-Trajectory Build

The same immutable Flash run and shard were extended from 20 to 50 episodes. All 30 new episodes completed without unresolved errors or coverage gaps, producing pool SHA-256 `13799d701d9033b0a0b0c4bea05f3e678be119e8e45c78078716c7629cf55464`. Across all 50 episodes, full-success rate is `0.5`, mean reward is `0.7018`, completion rate is `0.88`, valid-action rate is `0.9702`, and repeated-action rate is `0.0074`. The later shard prefix is harder than the first 20, but action and observation quality did not degrade.

The 369 transitions produced 347 segments and six Mid clusters. A dedicated build config raised High minimum positive support from `0.02` to `0.05`, requiring at least two of the 25 full-success trajectories. No High path passed. At the old `0.02` threshold, the only candidate had one positive and two negative supporting trajectories (`0.04` positive support, `0.08` negative support), so rendering it would have promoted adverse rather than repeatable evidence.

The resulting complete Mid-only snapshot is `tower_1f063f3414b3f90f`, with six Mid cards, zero High cards, and SHA-256 `0048159a493f168589cc71a1f9b4ca695ea4ff6c0317a02df1f5e49e6d36069c`. An immediate index rebuild reused all six vectors with no provider call, and a second snapshot build was byte-identical. The first preprocessing attempt reached the embedding provider's TPM limit; completed batches remained in the transition cache and the retry finished without recomputing them.

Four new pool-external training samples (`1005` through `1008`) formed the next paired gate. Static had one win, three ties, and no losses against No-Skill: mean reward `0.7431` versus `0.6875`, paired difference `+0.0556`, and bootstrap interval `[0, 0.1667]`. Mean steps and invalid actions were equal. Static completed sample `1007` with reward `0.2222` where No-Skill reached the task limit with reward zero. It used 1,883.25 more reported input tokens per episode; billable-token coverage remained zero.

This is the first non-negative pool-external gate for the rebuilt Tower, but four pairs are still diagnostic. All four tasks retrieved the same two substantially overlapping Mid cards, so the next bounded experiment tests Top-1 direct-Mid retrieval as a cost ablation before any larger rollout.

## Direct-Mid Top-1 Ablation

A content-addressed Top-1 variant, `tower_f2d5abf612995707`, reused the same 50 trajectories, six Mid cards, zero High cards, and retrieval index; only `direct_mid_top_k` changed from two to one. The retrieval contract now permits either one or two direct Mid cards while preserving Top-2 as the executable default.

On calibration samples `1005` through `1008`, Top-1 cut injected context from 2,920 to 1,387 characters but reduced mean reward from Top-2's `0.7431` to `0.6875`. Sample `1007` regressed from completed reward `0.2222` to task-limit reward zero. Mean steps increased from 8 to 11, invalid actions from 0.25 to 0.75 per episode, and reported input tokens from 21,161.25 to 30,156.25 per episode because longer trajectories outweighed the shorter prefix.

Top-1 therefore fails the calibration gate and is not run on fresh validation samples. Top-2 remains the selected retrieval setting: the second card is textually overlapping, but this experiment provides behavioral evidence that it still carries useful guidance.

## Twenty-Pair Validation

The four-pair directional result was expanded to 20 new WebShop training samples. The deterministic selection chose the lowest numeric sample IDs absent from both the 50-trajectory Tower provenance and every prior WebShop training result: `1009`, `1011-1019`, `1021-1029`, and `1031`. Both methods completed exact `20/20` result and trajectory coverage with no unresolved errors.

No-Skill achieved mean reward `0.7542`; the six-Mid, zero-High Static Tower achieved `0.7729`. The paired difference was only `+0.01875`, with 3 wins, 15 ties, 2 losses, and a 95% bootstrap interval of `[-0.05, 0.09375]`. Static added 0.65 steps, 0.1 invalid actions, and 4,272.7 reported input tokens per episode. Billable-token coverage remained zero. The result does not satisfy a broader-rollout gate because the reward interval includes harm and every observed efficiency measure regressed.

All 20 tasks retrieved the identical `mid_0002 + mid_0003` pair. Full-index diagnostics showed that positive, negative, and tied outcomes had overlapping cosine scores and Top-2 margins, so a similarity threshold cannot separate beneficial injection from harmful injection. Cluster audit then exposed the structural cause: the six clusters had weighted event-type purity `0.2161` and each mixed search, navigation, option, inspection, backtracking, and purchase stages. Their rendered cards therefore described nearly the same end-to-end workflow.

## Event-Stratified Ablation

An optional `event_type_stratification` build contract was added without changing legacy snapshot identities. When enabled, semantic and transition neighbors, outcome smoothing neighborhoods, and final spectral clustering remain within deterministic WebShop event types. The final K is clamped to cover every observed event type; additional clusters are allocated within event strata. A strictly positive contrastive-score guard also prevents High paths that are equally or more common in unsuccessful trajectories from becoming formal skills.

On the same 50 trajectories, weighted event purity increased from `0.2161` to `1.0`. The build produced 14 stage-specific Mid cards and six positive-contrastive High paths at `10%` minimum positive support. Immediate reruns reused all 14 Mid and six High cards and all 20 index vectors. The complete snapshot is `tower_94993ee334a6a439`, SHA-256 `30210d91f48fa1b0b15bfacad7afeb89e818dbcd38ee87c2f933a5ef791bc997`.

Four previously diagnosed samples formed a calibration-only quality check: one old Static win (`1019`), two losses (`1022`, `1027`), and one high-cost tie (`1012`). With High enabled, the event-stratified Tower selected the same generic High and six total skills on every task and scored `1.30` total reward versus the old Tower's `1.75`. Rejecting High reduced context from 5,856 to 2,159 characters and restored sample `1019`, but regressed sample `1012` to a 20-step zero-reward result; total reward was only `1.4167`.

The event-stratified variant therefore proves and fixes the structural purity defect but fails the behavioral calibration gate under reset-time retrieval. It is not run on fresh validation samples and is not promoted. The evidence points to the remaining boundary: one-time retrieval at the generic WebShop search page selects the same broad strategy for unrelated goals, even when the underlying cards are structurally distinct.

## Compact High Context Ablation

Static retrieval now exposes an execution-time `include_high_child_context` switch. Disabling it preserves the High ID, every ordered child Mid ID, direct-Mid IDs, and the complete Tower; only the model-visible body omits child cards already summarized by the High procedure. The default remains enabled, so existing execution behavior is unchanged.

On the four event-stratified calibration tasks, compact High context reduced injected text from 5,856 to 3,341 characters. Total reward nevertheless fell from the full event-stratified Tower's `1.30` to `1.25`, and sample `1019` regressed from completed reward `0.05` to a 20-step zero-reward result. Total reported input tokens changed only from 119,401 to 117,460 because the longer failed trajectory consumed the prefix repeatedly.

The compact variant therefore fails before fresh validation. Child prose is redundant at the text level, but this single-run behavioral evidence does not prove it is valueless; the default retains it. The large path variation across otherwise close configurations also motivates replicated evaluation before further method promotion.

## Replicated Event-Tower Check

The event-stratified Full High, No High, and compact High variants were rerun on the same four calibration tasks with repeat IDs `0`, `1`, and `2`. Each variant completed exact `12/12` coverage and was compared to the shared repeated No-Skill baseline using task-cluster bootstrap.

| Variant | Mean reward difference | 95% task-cluster CI | Mean step difference | Mean input difference |
|---|---:|---:|---:|---:|
| Full High | -0.0681 | [-0.2361, 0.1000] | -1.67 | +1,988.8 |
| No High | -0.1167 | [-0.2778, 0.0944] | +4.00 | +23,161.0 |
| Compact High | -0.0556 | [-0.1944, 0.0833] | -1.00 | +1,468.5 |

Compact High is the least harmful event-stratified variant, but all three have negative mean reward and intervals crossing zero. No variant proceeds to fresh samples. The replicated result confirms that event purity and concise context are insufficient when reset-time retrieval supplies generic workflow guidance rather than task-specific experience.

## Four-Rollout Experience Pool

The 50 fixed WebShop training tasks were rerun independently with repeat IDs 0-3 using `deepseek-v4-flash`. Exact coverage is 200/200 trajectories with no execution errors. The pool contains 94 full successes, split 23/24/24/23 across repeats. This replaces the earlier 25-success evidence base for the new retrieval experiments.

Two explicit evidence policies were materialized from the same pool. The SkillX-style success-only control contains all 94 full successes. The Trace2Tower mixed policy contains those 94 successes, 77 partial-reward failures, and two zero-reward failures whose task also has a successful repeat; 27 zero-reward trajectories without a successful same-task anchor are excluded. Every selection reason and excluded trajectory ID is recorded in the pool audit.

Rendering and rollout use separate model roles. Flat, Mid, and High cards call `ModelRole.RENDERER`, bound to `gpt-5.4`; agent trajectories and validation call `ModelRole.AGENT`, bound to `deepseek-v4-flash`. New build reports persist `renderer_model` explicitly.

## Pool And Retrieval Evaluation

A held-out WebShop test evaluation used 50 tasks and three repeats per task. The Flat 2x2 experiment compared a four-card pool versus the new 94-card pool and legacy Top-3 versus staged retrieval. Direct paired effects show that pool expansion improves reward by `+0.0258` under legacy retrieval and `+0.0544` under staged retrieval. The latter 95% task-bootstrap interval is `[+0.0060, +0.1139]`, establishing that the larger pool is useful when retrieval can exploit it.

The initial staged version selected up to eight cards and failed on 100 additional held-out tasks: `-0.0404` reward versus Top-3, with CI `[-0.0795, -0.0059]`. A GPT-5.4 self-filter reduced context but did not recover the Top-3 reward. The accepted staged configuration therefore caps selection at three cards while retaining Top-100 candidates, absolute threshold 0.45, best-relative margin 0.08, pairwise deduplication above 0.95, and MMR relevance weight 0.75.

Across 150 held-out tasks and 450 paired episodes, staged-cap3 versus legacy Top-3 has reward difference `+0.0035`, CI `[-0.0263, +0.0327]`. It reduces mean steps by 0.25, invalid actions by 0.064, and reported input tokens by 1,335 per episode. The claim is behavioral equivalence with lower execution cost, not a proven reward improvement.

## Informative Failure Ablation

High-path support is counted by distinct task separately within positive and negative evidence, so multiple repeats cannot inflate support. With a 10% positive-task threshold, the success-only Tower has 16 Mid and 26 High skills; the mixed Tower has 19 Mid and 17 High skills because negative evidence removes unsupported paths. Both complete snapshots use GPT-5.4-rendered cards and staged-cap3 direct retrieval.

The first held-out 50-task run actually used direct-Mid cap 8, not cap 3. Under that configuration, success-only Tower scores `-0.0298` versus NoSkill while mixed Tower scores `+0.0298`. Directly pairing mixed against success-only yields `+0.0596`, CI `[+0.0087, +0.1201]`. This is retained as an encouraging local result, but it is not sufficient on its own to establish the mixed policy.

An independent WebShop test set of 100 tasks, each with three repeats, was then evaluated under caps 3, 5, 8, and 12. WebShop full success is reported as the derived event `primary_score >= 0.999`; completion remains a separate environment outcome. Both reward and full-success differences use 10,000 bootstrap resamples over the 100 independent tasks after averaging each task's repeats.

| Cap | Success-only reward | Mixed reward | Success-only full success | Mixed full success | Mixed reward difference | Mixed success difference |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.6966 | 0.6748 | 48.7% | 44.0% | -0.0218 | -4.7% `[-9.0%, -1.0%]` |
| 5 | 0.7043 | 0.6798 | 45.0% | 45.0% | -0.0245 | 0.0% `[-4.7%, +4.7%]` |
| 8 | 0.6760 | 0.6862 | 44.3% | 45.7% | +0.0102 | +1.3% `[-4.0%, +6.7%]` |
| 12 | 0.6756 | 0.6748 | 44.0% | 43.7% | -0.0008 | -0.3% `[-4.7%, +3.7%]` |

Cap 3 significantly suppresses mixed full success. Cap 8 is the local optimum for mixed and restores a positive direction, while cap 12 adds almost no mixed context and does not improve outcomes. On all 150 held-out tasks available under cap 8, mixed versus success-only is `+0.0267` reward with CI `[-0.0073, +0.0611]` and `+0.9%` full success with CI `[-2.9%, +4.9%]`. Mixed versus NoSkill is `+0.0063` reward and `-0.4%` full success; both intervals cross zero. The current evidence therefore supports a WebShop-specific positive trend for mixed evidence at cap 8, not a significant general improvement.

A cap-8 High ablation further localizes the trend. Adding High to success-only changes reward by `-0.0186` and full success by `-1.0%`; adding High to mixed changes reward by `+0.0100` and full success by `+1.3%`. These intervals also cross zero, but the directions are consistent with negative evidence removing nine weaker High paths. Mixed Mid-only does not outperform success-only Mid-only, so any mixed advantage appears to come primarily from the smaller contrastively filtered High set rather than from adding more Mid clusters.

The cap sweep consumes test IDs 50-149 and is treated as validation rather than final test evidence. After fixing cap 8, the untouched WebShop test IDs 150-199 were run once with NoSkill, success-only Tower, and mixed Tower, each with three repeats and exact 150/150 coverage.

| Final holdout method | Mean reward | Full success | Completion | Mean steps | Mean input tokens |
|---|---:|---:|---:|---:|---:|
| NoSkill | 0.6927 | 48.0% | 92.0% | 7.11 | 17,670 |
| Success-only Tower | 0.6661 | 48.0% | 92.0% | 7.06 | 26,510 |
| Mixed Tower | 0.6740 | 48.7% | 90.7% | 7.12 | 23,500 |

Mixed versus success-only is `+0.0079` reward with CI `[-0.0181, +0.0353]` and `+0.7%` full success with CI `[0.0%, +2.0%]`; the success interval touches zero and is not claimed significant. Mixed uses 3,010 fewer input tokens per episode. Mixed versus NoSkill is `-0.0187` reward with CI `[-0.0929, +0.0609]` and `+0.7%` full success with CI `[-7.3%, +8.7%]`. The final conclusion is narrow: cap 8 prevents mixed evidence from being harmed by aggressive retrieval truncation and mixed is at least competitive with success-only, but neither Tower establishes an improvement over NoSkill on the untouched holdout.

## 中文结论：Mixed 与检索上限的交互

可以稳定得出以下经验结论：Mixed Tower 在全局最优的 cap 3 下失效，并不是因为失败轨迹天然有害，而是因为 mixed 证据把行为边界拆得更细，形成 19 个 Mid；success-only 只有 16 个 Mid。两者都只能向模型注入最多三个直接 Mid 时，mixed 中互补的查询、核验和决策步骤更容易被截断在 Top-3 之外。这个解释由检索覆盖统计支持，但仍属于机制推断，不等同于单独证明某一张被裁掉的卡必然有用。

在新增的 100 个 WebShop 验证任务上，mixed 相对 success-only 的满分成功率差随 cap 改变为：cap 3 `-4.7%`，95% CI `[-9.0%, -1.0%]`；cap 5 `0.0%`；cap 8 `+1.3%`；cap 12 `-0.3%`。因此 cap 3 对 mixed 的压制是可重复且达到区间不跨零的结果，而放宽 cap 只能消除该负效应，尚未证明 mixed 会稳定显著优于 success-only。

变化并非“cap 越大越好”。Mixed 从 cap 3 到 5 只有 24/100 个任务的检索集合变化，从 cap 5 到 8 只剩 9 个，从 cap 8 到 12 仅 1 个；平均证明技能数也在 `6.25 -> 6.35 -> 6.36` 附近饱和。cap 12 的回落因此不能解释为更多 mixed 信息持续进入上下文，更可能是边界任务变化与 rollout 方差。cap 8 只是 mixed 的局部最优覆盖点。

全局配置仍选择 cap 3。对标准 success-only 控制，cap 3 比 cap 8 的满分成功率高 `4.3%`，95% CI `[0.7%, 9.0%]`，每个 episode 少约 7,053 个输入 token；mixed 从 cap 3 放宽到 cap 8 只增加 `1.7%` 满分成功率，区间下界为零，reward 区间仍跨零。为了 mixed 的局部趋势把所有方法统一放宽到 cap 8，会牺牲更稳定的 success-only 成功率和成本。因此 cap 3 是全局默认，mixed-cap8 保留为解释 mixed 潜力的专项消融，而不是默认执行配置。

High 消融进一步表明，mixed 的正向趋势主要可能来自对比证据筛掉较弱 High：success-only 有 26 个 High，mixed 只有 17 个；加入 High 对 success-only 的 reward 方向为 `-0.0186`，对 mixed 为 `+0.0100`。两者区间仍跨零，所以当前不能声称 High 已被证明有效，只能说明 mixed 的潜在价值更可能位于“对高层路径做负证据过滤”，而不是简单增加更多 Mid。

## 中文结论：Flat 对 NoSkill 的最终对照

为了判断问题是否只来自 Tower 的层级构建算法，使用同一批 94 条满分成功经验构建的 Flat Skill staged-cap3，在未参与 Flat 参数选择的 WebShop test ID 150-199 上运行了三次重复。Flat 和已完成的 NoSkill 使用完全相同的 50 个任务与 150 个 episode key，均为完整覆盖且无未解决错误。

| 方法 | 平均 reward | 满分成功率 | Completion | 平均步数 |
|---|---:|---:|---:|---:|
| NoSkill | 0.6927 | 48.0% | 92.0% | 7.11 |
| Flat staged-cap3 | 0.6813 | 49.3% | 88.7% | 8.06 |

Flat 相对 NoSkill 的 reward 差为 `-0.0113`，95% CI `[-0.0687, +0.0513]`；满分成功率差为 `+1.3%`，95% CI `[-6.0%, +8.7%]`。两种指标都不能证明 Flat 更强，同时 Flat 的 completion 更低、步数更多。

因此，当前负结果不能只归因于 Tower 的聚类、High 路径或 mixed 证据算法：把相同成功经验改成更直接的 Flat 卡片后也没有稳定超过 NoSkill。更可能的共同瓶颈是 reset-time 一次性经验注入的迁移性、任务与经验的语义匹配精度，或经验文本干扰 DeepSeek 原本已较强的 WebShop 策略。Flat 满分成功率方向略正，说明经验并非被证明完全无效；准确结论是现有 Flat/Tower 都未在最终 holdout 上建立相对 NoSkill 的显著优势。
