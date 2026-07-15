# Failure-Set Skill Intervention

> Post-hoc diagnostic on six repeat-0 common failures. This is not a held-out generalization estimate.

## Aggregate result

| Method | Mean reward | Full success | Zero reward | Mean steps |
|---|---:|---:|---:|---:|
| final_t1 | 0.0000 | 0.0% | 100.0% | 14.67 |
| generic_manual | 0.0000 | 0.0% | 100.0% | 14.83 |
| recovery_skill | 0.1111 | 11.1% | 88.9% | 16.00 |
| skillx | 0.0000 | 0.0% | 100.0% | 15.50 |

## Per-task mean reward across repeats 0-2

| Sample | Final T1 | SkillX | Generic manual | Recovery skill |
|---|---:|---:|---:|---:|
| webshop:232 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| webshop:492 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| webshop:499 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| webshop:664 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| webshop:951 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| webshop:969 | 0.0000 | 0.0000 | 0.0000 | 0.6667 |

## Paired task-level comparison

| Recovery skill minus | Mean delta | W/T/L |
|---|---:|---:|
| final_t1 | +0.1111 | 1/5/0 |
| skillx | +0.1111 | 1/5/0 |
| generic_manual | +0.1111 | 1/5/0 |

## What Final T1 retrieved on these failures

| Skill | Name | Retrievals | Sample coverage |
|---|---|---:|---:|
| `mid_0002` | Select required options before purchase | 150 | 6/6 |
| `high_efbf322a092b` | Search, open, and configure the chosen product | 127 | 6/6 |
| `mid_0006` | Select matching product options | 121 | 6/6 |
| `mid_0000` | Search with core attributes and open a matching result | 114 | 6/6 |
| `high_f4ff56f0acaa` | Search and buy a direct match | 113 | 5/6 |
| `mid_0004` | Search and open a likely matching product | 113 | 5/6 |
| `mid_0001` | Refine search terms and inspect matching results | 90 | 6/6 |
| `mid_0007` | Search and open the closest matching product | 62 | 3/6 |
| `mid_0005` | Search or return to search | 17 | 3/6 |
| `high_69655a587d87` | Check description, then features, for requirement evidence | 8 | 1/6 |
| `mid_6fd1c9c5ff45` | Inspect features and return to the product page | 8 | 1/6 |

## Why the retrieved skills did not break through

| Retrieved guidance family | What it supplies | Missing control signal |
|---|---|---|
| Search/open/refine | A plausible next query or candidate | No memory of rejected queries and candidates; no remaining-step budget |
| Configure/purchase | Select visible options, then buy | No hard rule that an unknown requirement blocks purchase |
| Detail inspection | Check hidden evidence | Retrieved only 8 times with 1/6 sample coverage |
| Return to search | Backtrack after a mismatch | No systematic pivot order or stop condition |

## Failure behavior

| Sample | Pattern | Repeated exact searches | Zero-reward purchases | 20-step runs |
|---|---|---:|---:|---:|
| webshop:232 | search exhaustion | 2 | 0 | 3 |
| webshop:492 | search exhaustion | 1 | 0 | 3 |
| webshop:499 | premature purchase | 0 | 3 | 0 |
| webshop:664 | search exhaustion | 0 | 0 | 3 |
| webshop:951 | search exhaustion | 6 | 0 | 3 |
| webshop:969 | premature purchase | 0 | 3 | 0 |

## What changed under the recovery skill

| Sample | Mean reward | Search actions | Zero-reward purchases | 20-step runs |
|---|---:|---:|---:|---:|
| webshop:232 | 0.0000 | 15 | 0 | 3 |
| webshop:492 | 0.0000 | 28 | 0 | 3 |
| webshop:499 | 0.0000 | 3 | 3 | 0 |
| webshop:664 | 0.0000 | 25 | 0 | 3 |
| webshop:951 | 0.0000 | 31 | 0 | 3 |
| webshop:969 | 0.6667 | 7 | 1 | 0 |

## Interpretation

The retrieved skills repeatedly cover search, opening a likely match, selecting options, and buying. They do not strongly encode a cross-step rejection ledger, a hard gate for unknown constraints, or a bounded recovery plan. The intervention tests those missing control signals without adding product answers.

The recovery skill raises mean reward from 0 to 0.1111, entirely through `webshop:969`, which succeeds in two of three repeats after rejecting plausible but mismatched candidates. The other premature-purchase task still buys the wrong candidate, and all four search-exhaustion tasks still reach 20 steps. Static guidance can help, but it does not reliably enforce the ledger, constraint gate, or query budget across steps.

Because the six tasks were selected after observing failure, any gain is evidence of a recoverable guidance gap on this failure class, not an unbiased estimate of general performance.
