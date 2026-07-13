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

The expanded Tower therefore still fails the bounded quality gate and is not eligible for full rollout. The next experiment treats these two tasks as diagnostic calibration only and tests a confidence-gated High retrieval rule on fresh pool-external samples; it must not report the calibration pair as held-out evidence.
