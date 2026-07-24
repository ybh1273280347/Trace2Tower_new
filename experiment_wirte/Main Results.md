## 5.2 Main Results

### 5.2.1 ALFWorld: Governing Long-Horizon Tasks

Table 1 reports results on the 134-task `valid_unseen` split. Trace2Tower Full and High-only reach 88.06% and 85.82%,
improving over No-Skill by 35.07 and 32.84 points and exceeding SkillX (81.34%) and ExpeL (80.60%). Section 6.2 examines
their relative ordering across API states.

**Table 1: Main results on ALFWorld `valid_unseen`.**

| Method                | Success rate | Avg. steps | Avg. invalid actions | Avg. input tokens | Avg. context characters |
|-----------------------|-------------:|-----------:|---------------------:|------------------:|------------------------:|
| No-Skill              |       52.99% |      14.84 |                 0.54 |            45,677 |                       0 |
| Expert-Crafted Skills |       76.12% |      11.49 |                 0.37 |            38,889 |                   3,286 |
| Trace2Skill +Combined |       58.96% |      14.11 |                 0.27 |            68,449 |                   9,508 |
| Trace2Skill +Error    |       61.94% |      14.14 |                 0.39 |            73,131 |                  11,447 |
| SkillX                |       81.34% |      11.69 |                 0.49 |            64,651 |                  13,092 |
| ExpeL                 |       80.60% |      11.57 |                 0.31 |            52,831 |                   7,469 |
| **Trace2Tower Full**  |   **88.06%** |  **10.06** |             **0.22** |            43,758 |                   5,422 |
| Trace2Tower High-only |       85.82% |      10.50 |             **0.22** |        **37,410** |               **1,960** |

Full exceeds Expert-Crafted Skills by 11.94 points (95% CI $[+4.48,+19.40]$; McNemar $p=0.00522$), and reduces the
No-Skill trajectory from 14.84 to 10.06 steps and invalid actions from 0.54 to 0.22. High-only retains most of the gain
with the smallest input and injected context among automatic methods, showing that rewritten High skills already encode
substantial long-range structure. The frozen Tower also improves over No-Skill by 16.67 and 21.67 points on two disjoint
deployment test sets; Section 6.5 reports the corresponding feedback optimization.

### 5.2.2 WebShop: Cross-Domain Potential and Boundary

Trace2Tower obtains the highest mean reward on the frozen WebShop partition, 0.71477: +0.06242 over No-Skill (95% CI
$[+0.00500,+0.12450]$), +0.03050 over SkillX, and +0.08129 over ExpeL.

**Table 2: Main results on the frozen WebShop test partition.**

| Method                | Mean reward | Avg. steps | Avg. invalid actions | Avg. input tokens |
|-----------------------|------------:|-----------:|---------------------:|------------------:|
| No-Skill              |     0.65235 |       7.69 |                 0.30 |            19,572 |
| Expert-Crafted Skills |     0.70085 |   **6.22** |             **0.06** |        **15,526** |
| Trace2Skill +Combined |     0.59685 |      11.50 |                 0.54 |            47,043 |
| Trace2Skill +Error    |     0.62833 |      11.54 |                 0.74 |            42,828 |
| SkillX                |     0.68427 |       7.95 |                 0.29 |            34,044 |
| ExpeL                 |     0.63348 |       9.86 |                 0.31 |            54,092 |
| **Trace2Tower Full**  | **0.71477** |       9.90 |                 0.16 |            41,977 |
| Trace2Tower High-only |     0.62068 |       9.95 |                 0.30 |            37,135 |

The 0.01392 advantage over Expert-Crafted Skills is not significant (95% CI $[-0.02809,+0.05758]$). Automatic experience
is also not uniformly beneficial: ExpeL, both Trace2Skill variants, and Trace2Tower High-only fall below No-Skill,
whereas SkillX and Trace2Tower Full improve reward. Full exceeds High-only by 0.09408 mean reward; in the paired comparison,
Full wins 19 tasks, High-only wins 5, and 76 tie. WebShop reveals product attributes, option states, and recovery decisions
only after interaction, so experience selected or activated from the initial request may not match the execution branch
revealed by the page. Full also uses more steps and tokens than No-Skill, SkillX, and Expert-Crafted Skills. Section 6.6
analyzes this task-dependent pattern and its relationship to each method's experience representation.

## 5.3 Efficiency and Cost

### 5.3.1 Inference-Time Efficiency

On ALFWorld, Full improves over SkillX by 6.72 points while reducing cumulative input tokens by 32.3% and injected
context by 58.6%; relative to ExpeL, the reductions are 17.2% and 27.4%. High-only reaches 85.82% with 37,410 input
tokens and 1,960 context characters, reducing these costs by 14.5% and 63.8% relative to Full for a 2.24-point success
difference. Input tokens measure the acting agent's full episode context, whereas context characters measure injected
experience text. Full also uses 4.2% fewer cumulative input tokens than No-Skill despite adding external context, because
its shorter trajectories reduce the interaction history carried into later steps.

### 5.3.2 Construction Cost

**GPT chat usage for ALFWorld skill construction.**

| Method      | GPT calls | Input tokens | Output tokens |  Total tokens |
|-------------|----------:|-------------:|--------------:|--------------:|
| Trace2Tower |       157 |  **983,841** |    **41,891** | **1,025,732** |
| SkillX      |   **150** |    1,133,696 |        99,945 |     1,233,641 |

Trace2Tower uses 13.22% fewer input, 58.09% fewer output, and 16.85% fewer total GPT tokens than SkillX, with 4.67% more
calls. The saving therefore comes from lower generation volume rather than fewer invocations; most of the reduction is
in generated output, consistent with consolidating evidence in the graph before rendering the final hierarchy.
