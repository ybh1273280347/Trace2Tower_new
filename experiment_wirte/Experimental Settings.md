# Benchmarks and Metrics

We evaluate Trace2Tower on ALFWorld and WebShop, two long-horizon interactive-agent benchmarks.
For ALFWorld, we construct skills from P310, comprising 310 training tasks and four No-Skill
rollouts per task. Following the AgentBench protocol, we evaluate on the frozen `valid_unseen`
split of 134 tasks with a 20-step interaction budget. For WebShop, we use P100 for skill
construction (100 training tasks, four rollouts each) and evaluate on a frozen 100-task test
partition. Following AgentBench's WebShop LLM evaluation protocol, each WebShop episode is also
limited to 20 interaction rounds.

The primary metric is task success rate on ALFWorld and mean reward on WebShop. We additionally
report perfect-completion rate, mean interaction steps, mean invalid actions, mean agent input
tokens, and injected skill-context length when applicable. These metrics jointly measure task
quality and execution efficiency.

For ALFWorld, we also evaluate the frozen deployment runtime on two disjoint 120-task sets and
report paired success differences against No-Skill. Unless otherwise specified, all evaluations use deepseek-v4-flash 
with zero temperature, and skill construction and structured plan rewriting use gpt-5.4 and all results are averaged over three runs. 
We use two-sided exact McNemar tests for paired ALFWorld success comparisons and task-level paired
bootstrap confidence intervals for WebShop reward comparisons.
