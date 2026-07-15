# Trace2Tower Evidence Narrative

## Two-stage claim

The evidence supports a two-stage claim rather than discarding the earlier
experiments.

1. Under the same generic semantic retrieval family used by SkillX, a legacy
   Tower is already competitive with SkillX. This establishes that the Tower
   representation and induced skills are useful even when deployment does not
   exploit their graph structure.
2. Graph-aware dynamic retrieval then aligns deployment with the Tower's
   directed paths and current state. It improves the strongest held-out
   comparison and consistently reduces context cost, showing that a graph
   representation benefits from a graph-specific retriever.

The historical generic-retrieval runs remain valid evidence for stage 1. They
are not relabeled as final-algorithm runs and are not discarded.

## Stage 1: generic retrieval remains competitive

| Model / split | Legacy P100 Tower | P100 SkillX | Difference |
|---|---:|---:|---:|
| Flash Test-A repeat3 | 0.70958 | 0.70627 | +0.00332 |
| Pro Test-A repeat3 | 0.65458 | 0.67453 | -0.01994 |
| Flash Test-B repeat0 | 0.71465 | 0.69573 | +0.01892 |

All three reward intervals for Tower minus SkillX include zero. The methods are
therefore in the same empirical performance band under generic semantic
retrieval; the evidence does not support calling the legacy Tower ineffective.

The P100 graph-structure ablation uses this same legacy retrieval and cap8.
Full Tower scores `0.72092/56%`, versus Semantic-only `0.70157/50%`. Full saves
1.13 steps per task, paired 95% interval `[-1.93, -0.34]`. Thus relational
EigenTrace structure and High induction improve execution even before a
graph-specific retriever is introduced.

## Stage 2: graph-aware retrieval realizes Tower structure

| Model / split | Final Tower | P100 SkillX | Reward delta | Tower tokens | SkillX tokens |
|---|---:|---:|---:|---:|---:|
| Flash Test-A repeat3 | 0.71119 | 0.70627 | +0.00493 | 20,461 | 25,009 |
| Pro Test-A repeat3 | 0.69919 | 0.67453 | +0.02467 | 25,376 | 28,444 |
| Flash Test-B repeat0 | 0.75123 | 0.69573 | +0.05550 | 19,637 | 27,330 |

On Test-B, Final Tower minus SkillX has paired 95% interval
`[+0.01500, +0.10300]`, so the reward gain is significant. Input tokens fall
by 18.2% on Flash repeat3, 10.8% on Pro repeat3, and 28.2% on Test-B. Final
Tower also has the highest reward and full-success point estimates in both
Flash and Pro repeat3 matrices.

The retrieval audit explains the difference. Generic retrieval ranks skill
cards independently and can repeatedly inject near-duplicate or
stage-inappropriate Mids. Graph-aware retrieval selects a High path from the
goal, locates its active Mid from the current observation and learned event
profile, expands at most one directed successor, and enforces a total Mid
budget. The promoted detail High is selected on detail pages and not on search
pages.

## Efficiency is supporting evidence

Token cost is not the primary optimization objective. It is supporting evidence
that the gain is not purchased by a larger prompt. The final retriever reduces
tokens because it injects fewer, more state-appropriate skills and often
finishes in fewer steps. This directly addresses the main weakness of the
legacy Tower deployment while preserving its competitive reward evidence.

## Defensible conclusion

The defensible paper claim is:

> Hierarchical graph induction is already competitive with a strong skill
> baseline under generic semantic retrieval. A Tower-specific dynamic graph
> retriever better realizes that structure, producing the strongest held-out
> SkillX comparison and materially lower deployment context cost.

This is stronger and cleaner than claiming universal reward significance on
every split. Reward superiority is significant on Test-B versus SkillX and on
Pro repeat3 versus NoSkill; other comparisons remain directional.
