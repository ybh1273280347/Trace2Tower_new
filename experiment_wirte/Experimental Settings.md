## 5.1 Experimental Setup

### 5.1.1 Benchmarks and Metrics

We evaluate on two interactive decision-making benchmarks. **ALFWorld** (Shridhar et al., 2021) represents household
tasks as text environments in which agents locate objects, satisfy manipulation prerequisites, apply transformations
such as cleaning, heating, or cooling, and complete a final placement. Following AgentBench (Liu et al., 2023), we
evaluate all 134 solvable `valid_unseen` tasks across six families. Skills are constructed from P310: 310 training tasks with four
No-Skill rollouts each (1,240 trajectories). We report success rate, steps, invalid actions, cumulative agent input
tokens, and injected skill-context characters. The split contains unseen rooms and layouts, so success requires
recombining learned object interactions and state transitions rather than replaying a training environment.

**WebShop** (Yao et al., 2022) requires agents to search, compare products, bind options, verify constraints, and
purchase from a catalog of approximately 1.18 million products using natural-language shopping instructions. We construct skills
from P100: 100 training tasks with four No-Skill rollouts each (400 trajectories), and evaluate on a frozen 100-task
partition. The primary metric is mean reward in $[0,1]$, which grades satisfaction of product type, attributes, options,
and price; we also report steps, invalid actions, and cumulative input tokens. Both benchmarks allow at most 20
environment steps.

### 5.1.2 Models and Runtime Protocol

GPT-5.4 serves as the Skill Author and Plan Rewriter, and DeepSeek-V4-Flash as the Skill User. The Author converts
trajectory evidence into reusable graph skills; the Rewriter specializes retrieved High knowledge to the current goal;
and the User acts in the environment. Agent decoding uses temperature 0 and at most 512 output tokens per interaction;
each test task is executed once. Both Trace2Tower variants retrieve the top-three High skills and rewrite them into a
task-specific plan. **Full** additionally injects up to eight step-aligned Mid skills, whereas **High-only** uses the
rewritten High plan alone. API-period robustness is studied in Section 6.3.

### 5.1.3 Baselines

We compare with **No-Skill** and **Expert-Crafted Skills**, a fixed human-authored domain policy. The automatic
baselines cover three experience representations: **SkillX** (Wang et al., 2026) retrieves planning and functional skills from a
multi-level library; **ExpeL** (Zhao et al., 2024) combines shared insights with retrieved successful trajectories; and
**Trace2Skill** (Ni et al., 2026) consolidates trajectory-local patches into one directory injected without test-time
retrieval. We report its predefined **+Error** and **+Combined** variants, constructed from failed trajectories and from
both successful and failed trajectories, respectively.

### 5.1.4 Comparison Protocol

All automatic methods receive the same benchmark-specific trajectory pool, start without human-authored skills, and are
constructed once before evaluation on the frozen test tasks. The Skill User, task set, and interaction budget are fixed,
while each method retains its native experience-use mechanism. Expert-Crafted Skills is reported separately as a
human-prior reference.
