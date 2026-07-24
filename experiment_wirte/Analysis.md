# 6 Analysis

## 6.1 Graph Structure and Construction Mechanisms

Trace2Tower collapses 13,724 segment instances into 3,764 quotient nodes (72.57% reduction), then constructs 39 Mid and
118 High skills. High paths contain 2.924 Mid nodes on average and cover 28 Mid communities. At inference, 53 High and 30
Mid skills are used across the ALFWorld test set, corresponding to 44.9% and 76.9% of the two layers. The compression is
therefore not followed by retrieval from a small fixed core: test tasks activate a substantial fraction of the learned
hierarchy.

**Table 3: Component ablation of ALFWorld graph construction.**

| Configuration     | Retained signals                                  | Mid / High | Success rate |
|-------------------|---------------------------------------------------|-----------:|-------------:|
| G0 Full           | Semantic + transition + outcome + signed contrast |   39 / 118 |   **88.06%** |
| G1 Semantic-only  | Semantic, fixed $K=39$                            |     39 / 0 | Not executed |
| G2 No Transition  | Semantic + outcome + signed contrast              |    19 / 76 |       70.15% |
| G3 No Outcome     | Semantic + transition + signed contrast           |   39 / 106 |       73.88% |
| G4 No Contrastive | Semantic + transition + outcome                   |    10 / 44 |       73.88% |

All variants start from the same quotient nodes. Removing transition edges causes the largest drop and reduces the
hierarchy to 19/76 Mid/High skills. Removing outcome consistency preserves most Mid communities but yields fewer High
paths, whereas removing signed contrast collapses the hierarchy to 10/44. Thus, temporal, outcome, and contrastive
signals affect both task success and the structure available for long-range composition.

## 6.2 Runtime-Policy Stability Across API States

We compare paired Full and High-only outcomes in three API periods.

| State | Full only | High-only only | Ties | Difference |           95% CI | McNemar $p$ |
|-------|----------:|---------------:|-----:|-----------:|-----------------:|------------:|
| A     |         7 |              4 |  123 |   +2.24 pp |  $[-2.24,+7.46]$ |       0.549 |
| B     |        23 |              9 |  102 |  +10.45 pp | $[+2.24,+18.66]$ |      0.0201 |
| C     |        12 |             14 |  108 |   -1.49 pp |  $[-8.96,+5.97]$ |       0.845 |

Only State B significantly favors Full; State C slightly favors High-only. Neither runtime policy dominates across API
states. In particular, the State-A difference of 2.24 points should not be interpreted as a fixed estimate of Mid-skill
value: both its magnitude and sign change across periods.

## 6.3 Robustness Across API States

We next compare complete evaluations from the same three periods while fixing the task set, artifacts, executor name,
temperature, and interaction budget.

**Table 4: Performance across API states.**

| Method                | State A | State B | State C |
|-----------------------|--------:|--------:|--------:|
| No-Skill              |  52.99% |  43.28% |  42.54% |
| Trace2Skill +Combined |  58.96% |  62.69% |  62.69% |
| Trace2Skill +Error    |  61.94% |  61.94% |  59.70% |
| SkillX                |  81.34% |  70.90% |  73.88% |
| ExpeL                 |  80.60% |  82.09% |  82.09% |
| Trace2Tower Full      |  88.06% |  80.60% |  78.36% |
| Trace2Tower High-only |  85.82% |  70.15% |  79.85% |

No-Skill falls by approximately 10 points after State A, establishing a period-level shift in executor behavior. The
response is method dependent: Trace2Skill remains within a narrow range, ExpeL rises slightly, and SkillX and Trace2Tower
change more substantially. Section 6.6 relates this pattern to how each method exposes experience.

## 6.4 Cross-Model Generalization and Author-User Effects

**Table 5: Skill Author by executor success rate on ALFWorld.**

| Experience condition             | DeepSeek-V4-Flash executor | DeepSeek-V4-Pro executor |
|----------------------------------|---------------------------:|-------------------------:|
| No-Skill                         |                     52.99% |                   66.42% |
| GPT-5.4-authored Tower           |                 **88.06%** |               **85.82%** |
| DeepSeek-V4-Flash-authored Tower |                     65.67% |                   79.10% |

The matrix supports two controlled readings: rows compare executors under a fixed Author, while columns compare Authors
under a fixed executor. This separates cross-user reuse from differences in how models render trajectory evidence.

### 6.4.1 Fixed-Author Transfer Across Executors

With the GPT-5.4-authored Tower fixed, Flash and Pro improve by 35.07 and 19.40 points over their No-Skill controls and
finish within 2.24 points of each other. The learned graph therefore transfers across heterogeneous Skill Users.

### 6.4.2 Controlled Cross-Author Analysis

The Flash-authored Tower improves both executors by 12.69 points. With Flash as executor, the GPT-authored Tower records
34 exclusive wins against 4 for the Flash-authored Tower (+22.39 points; McNemar $p=6.04\times10^{-7}$). With Pro, the
split is 20 versus 11 (+6.72 points; $p=0.150$). Graph structure transfers across Authors, while textual realization still
affects absolute performance.

### 6.4.3 Reciprocal Author-User Combinations

GPT-5.4 Author with Flash User reaches 88.06% (+35.07 points over No-Skill), while Flash Author with GPT-5.4 User reaches
85.07% (+33.58 points). Because both roles change, the 2.99-point difference is a joint model-role effect. On WebShop,
Trace2Tower also improves mean reward by 0.06242, 0.04658, and 0.01642 with Flash, GPT-5.4, and Pro executors.

## 6.5 Feedback-Based Graph Optimization

Graph feedback supports four operations: **Split** separates heterogeneous Mid evidence, **Merge** combines compatible
communities, **Promote** elevates supported Mid paths into High motifs, and **Downweight** reduces harmful High paths. The
learned update splits one Mid into two shadow children, promotes eight High motifs, downweights one High, and performs no
merge. Retrieval expands Mid-to-High and child-parent relations, then applies Pareto selection over semantic relevance,
child relevance, and feedback evidence.

| Method                               |     Test-1 |     Test-2 |    Overall |
|--------------------------------------|-----------:|-----------:|-----------:|
| No-Skill                             |     63.33% |     59.17% |     61.25% |
| Frozen Tower v0                      |     80.00% |     80.83% |     80.42% |
| Four-action graph + TF-IDF Pareto    | **86.67%** |     82.50% | **84.58%** |
| Four-action graph + Embedding Pareto |     81.67% | **85.83%** |     83.75% |

The frozen Tower improves over No-Skill by 16.67 and 21.67 points on the two test sets. Feedback raises the overall score
from 80.42% to 84.58% with TF-IDF and 83.75% with embeddings. Their 0.83-point difference and opposite test-set maxima
show that the gain is not specific to one relevance representation. The shared improvement instead points to the
editable graph, relation-aware expansion, and Pareto treatment of relevance and feedback as the common mechanism.

## 6.6 Task Structure Determines How Experience Transfers

ALFWorld exposes the task family and target transformation in its initial goal, and its families reuse stable prerequisite
chains. WebShop instead reveals the evidence governing inspection, backtracking, and purchase only through interaction.
This difference organizes the otherwise non-monotonic results across methods.

### 6.6.1 Stable Transition Topology in ALFWorld

The six ALFWorld families share locating, prerequisite satisfaction, state transformation, and final-placement events.
No-Skill is weakest on heating (26.09%), cooling (33.33%), and placing two objects (41.18%). Full raises these rates to
78.26%, 85.71%, and 76.47%. High-only reaches 82.35% on the two-object family, showing that High skills alone capture
substantial long-range organization.

![ALFWorld task-family success rates](../clean_docs/figures/alfworld-family-heatmap.png)

The baseline profiles locate different forms of reuse. Trace2Skill improves common operations but not two-object tasks;
ExpeL is strongest on cooling (95.24%) and reaches 70.59% on two-object placement, where concrete trajectories are
directly reusable; SkillX reaches 100% on look and 93.55% on cleaning through plan and function abstractions. Trace2Tower
is not best in every family, but improves the transition-heavy families by representing both local operations and
cross-stage dependencies.

### 6.6.2 State-Revealed Branching in WebShop

Similar shopping requests can expose different candidates, missing attributes, options, prices, and recovery paths, so
task similarity only partially predicts execution-path similarity. Trace2Skill's globally active directory and ExpeL's
task-level trajectory recall both fall below No-Skill. These results align with Trace2Skill's reported patch-irrelevant
regressions (Ni et al., 2026) and ExpeL's distinction between WebShop's combined reasoning/action demands and ALFWorld's
reusable action sequences (Zhao et al., 2024).

SkillX gains from plan/function abstraction, but its initial pseudo-plan still precedes decisive page evidence; its own
analysis likewise reports over-imitation and pseudo-plan mismatch in dynamic environments (Wang et al., 2026).
Trace2Tower attains the best primary-partition reward but performs more checks and changes sign on a second partition.
The graph captures search, verification, recovery, and purchase transitions, while the current runtime does not re-query
them as page state evolves.

### 6.6.3 Representation Fit

The results distinguish four regimes: global skills suit broadly invariant procedures; episodic recall suits tasks whose
similarity predicts action sequences; hierarchical skills suit goals that reveal a reliable plan decomposition; and event
graphs suit recurring transitions, prerequisites, and outcome-conditioned recovery paths. State-revealed branching calls
for experience selection during interaction rather than only from the initial query.

The API results add a second axis: Trace2Skill and ExpeL expose more direct experience, whereas SkillX and Trace2Tower
require the executor to compose task-conditioned guidance, producing larger ALFWorld gains but greater API-state
sensitivity. ALFWorld's stable task families therefore support interpretable graph ablations and feedback optimization;
WebShop is used to characterize cross-domain transfer and the boundary imposed by state-revealed branching. This
benchmark-specific allocation mirrors SkillX's decision to place component studies only where the benchmark's task and
tool structure directly exercises the component being isolated (Wang et al., 2026).
