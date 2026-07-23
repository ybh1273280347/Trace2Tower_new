# Current Experiment Conclusions

This document consolidates the completed, auditable evidence as of
2026-07-15. It separates empirical conclusions from proposed future work.
WebShop-specific failure features are diagnostics only and are not part of the
general Trace2Tower algorithm contract.

## Evidence scope

- WebShop Test-A contains 100 frozen tasks. Primary results use real repeat
  IDs 0, 1, and 2 and aggregate within task before averaging across tasks.
- WebShop Test-B contains another 100 frozen tasks and is used as a
  cross-split robustness check.
- ALFWorld collection is still running. No cross-domain performance claim is
  supported until the ALFWorld artifact and evaluation are complete.

## Final WebShop results

### Flash Test-A repeat3

| Method | Reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| **Final Trace2Tower, graph cap3** | **0.71119** | 53.67% | 6.83 | 0.16 | 20,461 |
| Legacy Trace2Tower, cap8 | 0.70958 | **54.67%** | 7.14 | 0.17 | 32,334 |
| P100 SkillX | 0.70627 | 49.33% | 7.04 | 0.31 | 25,009 |
| Graph cap8 | 0.70464 | 53.00% | 7.58 | 0.21 | 27,844 |
| Manual Skill | 0.69158 | 50.67% | **6.53** | **0.06** | **19,259** |
| NoSkill | 0.68492 | 50.33% | 7.89 | 0.38 | 20,610 |

Final Trace2Tower has the highest reward point estimate. Its reward advantage
over NoSkill is `+0.02628`, interval `[-0.02936, +0.08356]`; its advantage
over SkillX is `+0.00493`, interval `[-0.03094, +0.04072]`. These Flash
reward differences are directional, not statistically significant.

### Pro Test-A repeat3

| Method | Reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| **Final Trace2Tower, graph cap3** | **0.69919** | **51.33%** | 7.56 | 0.38 | 25,376 |
| Manual Skill | 0.69194 | 51.00% | **6.59** | **0.04** | **18,700** |
| P100 SkillX | 0.67453 | 48.33% | 7.68 | 0.19 | 28,444 |
| NoSkill | 0.59217 | 42.67% | 9.23 | 0.88 | 25,672 |

Final Trace2Tower minus NoSkill is `+0.10703`, interval
`[+0.06255, +0.15334]`, with a `+8.67` point full-success difference,
interval `[+3.33, +14.33]` points. Both improvements are significant. Final
Trace2Tower minus SkillX is `+0.02467`, interval
`[-0.01806, +0.07022]`, so superiority over SkillX remains directional.

## Mechanism evidence

| Matched comparison | Reward | Full success | Steps | Input tokens |
|---|---:|---:|---:|---:|
| **Full Tower, legacy cap8** | **0.72092** | **56%** | **7.17** | 32,360 |
| Semantic-only, legacy cap8 | 0.70157 | 50% | 8.30 | 32,469 |
| **Mixed evidence, graph cap3** | **0.70892** | 52% | **7.17** | **22,067** |
| No-Mixed, graph cap3 | 0.69825 | 52% | 7.75 | 23,381 |

- Full Tower minus Semantic-only is `+0.01935` reward. The reward interval
  crosses zero, but Full Tower saves `1.13` steps with interval
  `[-1.93, -0.34]`. Relational graph induction and High paths therefore have
  clear step-efficiency evidence beyond semantic clustering.
- Removing partial and failure evidence changes reward by `-0.01067` and adds
  `0.58` steps. Both intervals cross zero. Mixed evidence is supported as a
  directional regularizer, not as a statistically established reward gain.

## Retrieval budget

Graph cap8 does not improve over graph cap3. Its repeat3 reward is `0.70464`
versus `0.71119`, while adding about 7,383 input tokens and 0.75 steps per
episode. Graph cap3 remains the current WebShop runtime configuration. This
control also shows that the graph result is not obtained by injecting more
skill text.

## Cross-split robustness

| Test-B method | Reward | Full success | Steps | Invalid actions | Input tokens |
|---|---:|---:|---:|---:|---:|
| **Final Trace2Tower** | **0.75123** | **53%** | **6.87** | **0.08** | 19,637 |
| NoSkill repeat0 | 0.73323 | 48% | 7.32 | 0.32 | **16,571** |
| NoSkill repeat1 | 0.72798 | 49% | 7.79 | 0.33 | 18,868 |
| P100 SkillX | 0.69573 | 47% | 7.50 | 0.34 | 27,330 |

The high Test-B NoSkill score is stable: the two-repeat task mean is
`0.73061`. Final Trace2Tower is `+0.02062` above that mean, but the available
interval crosses zero. Final Trace2Tower minus SkillX is `+0.05550`, interval
`[+0.01500, +0.10300]`, which is significant. The evidence supports robust
advantage over SkillX on Test-B, while benefit over NoSkill remains
split-sensitive.

## Construction efficiency

| Method | Source trajectories | Constructed skills | LLM calls | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| **Trace2Tower** | 351 mixed | 9 Mid + 5 High | **14** | **1,269,049** | **7,147** | **1,276,196** |
| SkillX | 186 full-success | 51 planning + 2 atomic | 429 | 1,686,214 | 197,111 | 1,883,325 |

Trace2Tower uses 96.7% fewer LLM calls, 24.7% fewer input tokens, 96.4%
fewer output tokens, and 32.2% fewer total construction chat tokens. This
supports a graph-first evidence-compression claim. It does not prove a billing
cost ratio because cached-token proportions and evidence-selection policies
differ.

## Failure evidence

Final Trace2Tower and SkillX largely fail on the same Test-A tasks.

| Failure definition | Final | SkillX | Shared | Jaccard |
|---|---:|---:|---:|---:|
| Zero reward | 7 | 7 | 6 | 0.750 |
| Non-full reward | 46 | 51 | 44 | 0.830 |

Their per-task reward correlation is `0.878`, and 40 of 100 tasks are
non-full for Final Trace2Tower, SkillX, legacy Tower, and NoSkill together.
This supports a shared task-difficulty regime rather than method-specific
catastrophic failures.

The following WebShop measurements explain that regime but do not define the
general algorithm:

| Diagnostic | Shared zero, n=6 | Shared non-full, n=44 | Shared full, n=47 |
|---|---:|---:|---:|
| Dataset query / instruction overlap | 0.056 | 0.152 | 0.244 |
| Instruction / target-title overlap | 0.160 | 0.303 | 0.389 |
| Mean search actions | 4.67 | 1.85 | 1.12 |
| Mean search backtracks | 2.92 | 0.63 | 0.05 |
| Mean repeated actions | 5.83 | 1.46 | 0.26 |
| Both methods exhaust 20 steps | 66.7% | 9.1% | 0% |
| Both methods buy but remain non-full | 33.3% | 86.4% | 0% |

The common failures separate into search-stagnation failures and premature
terminal-decision failures. Product constraints, option codes, prices, search
pages, and `Buy Now` are WebShop-level interpretations used only to explain
the observed results. This diagnostic does not establish a missing event type
or motivate a change to the Trace2Tower core algorithm.

### Retrieved-skill audit and post-hoc intervention

The six shared zero-reward tasks do not fail because retrieval returns no
skills. Across their 18 Final Trace2Tower trajectories, the most frequent
retrieved cards are:

| Skill | Retrievals | Sample coverage |
|---|---:|---:|
| `mid_0002`, select required options before purchase | 150 | 6/6 |
| `high_efbf322a092b`, search, open, and configure | 127 | 6/6 |
| `mid_0006`, select matching product options | 121 | 6/6 |
| `mid_0000`, search with core attributes | 114 | 6/6 |
| `high_f4ff56f0acaa`, search and buy a direct match | 113 | 5/6 |
| `mid_0001`, refine search terms | 90 | 6/6 |

These cards provide local positive actions but weakly represent cross-step
control state: which searches and candidates have already failed, which hard
requirements remain unsupported, when purchase must be blocked, and how much
episode budget remains. The result is repeated retrieval of search/configure/
purchase paths without a strong recovery path for a near miss.

A post-hoc manual intervention tested that hypothesis. The fixed skill contains
no sample IDs, product names, candidate identifiers, or target answers. It adds
a three-state hard-constraint checklist, a purchase gate for unknown evidence,
query and candidate ledgers, and a bounded recovery plan.

| Method on shared-zero set, n=6 | Mean reward | Full success | Zero reward | Mean steps |
|---|---:|---:|---:|---:|
| Final Trace2Tower | 0.0000 | 0.0% | 100.0% | 14.67 |
| SkillX | 0.0000 | 0.0% | 100.0% | 15.50 |
| Existing generic Manual Skill | 0.0000 | 0.0% | 100.0% | 14.83 |
| **Manual recovery intervention** | **0.1111** | **11.1%** | **88.9%** | 16.00 |

The gain comes entirely from `webshop:969`, which succeeds in two of three
repeats after inspecting and rejecting plausible but mismatched candidates.
The other premature-purchase task still buys an incorrect candidate, and all
four search-stagnation tasks still exhaust 20 steps. Thus there is a real but
limited recoverable guidance gap. Static skill text is not sufficient to
reliably enforce cross-step memory, constraint gates, or action budgets.

This suggests a general improvement direction after domain event extraction:
learn contrastive recovery skills from failed-to-successful event transitions,
retrieve against the current unresolved state rather than only the task text,
and expose explicit execution state such as tried alternatives, unresolved
requirements, and remaining budget. Those state variables are domain-grounded
but the event-to-recovery-skill construction principle is general.

## Generality boundary

Trace2Tower assumes that trajectories have already been converted into
domain-meaningful event segments. Event extraction is necessarily
domain-specific: WebShop and ALFWorld use different event definitions and may
use different extraction procedures. The method does not require, propose, or
claim a universal event extraction algorithm.

The general Trace2Tower contribution begins after event extraction:

\[
\mathcal{D}_{seg}
\rightarrow G_{ET}
\rightarrow Z_{ET}
\rightarrow \mathcal{T}
\]

Given extracted events, the same downstream principles apply across domains:

1. represent event segments semantically;
2. connect them using observed temporal transitions;
3. encode success and failure consistency;
4. perform contrastive spectral decomposition;
5. induce Low, Mid, and High skills from the resulting structure.

Generality therefore means that any domain with a suitable event extractor can
use the same event-to-graph-to-Tower procedure. It does not mean that event
names, event boundaries, or extraction rules must be shared across domains.

The current WebShop failure analysis is a diagnostic of the evaluated agent
and task set. It does not by itself identify a missing component in the
general algorithm. Any future algorithmic improvement motivated by these
failures must operate on the already extracted events, their graph relations,
or their outcome evidence, rather than redefining domain event extraction as a
new research problem.

## Defensible current claim

The completed evidence supports the following statement:

> Trace2Tower compresses outcome-labeled trajectories into a relational,
> hierarchical skill representation. Under generic retrieval it is already in
> the same empirical band as SkillX; graph-aware dynamic retrieval improves
> execution efficiency and produces the strongest held-out SkillX comparison.
> Reward superiority over NoSkill is significant for Pro but remains
> split-sensitive for Flash.

It does not yet support universal cross-domain superiority because ALFWorld is
unfinished.

## Authoritative artifacts

- `final-flash-repeat3.json`
- `final-pro-repeat3.json`
- `final-flash-graph-cap8-repeat3.json`
- `final-mechanism-ablations.json`
- `final-algorithm-results.json`
- `test-b-noskill-variance.json`
- `failure-overlap.json`
- `failure-intervention/analysis.json`
- `failure-intervention/REPORT.md`
- `construction-cost.json`
