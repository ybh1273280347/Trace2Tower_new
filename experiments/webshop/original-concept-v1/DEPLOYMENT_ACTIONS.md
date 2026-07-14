# Deployment Refinement Action Mapping

The original material defines four deployment refinement actions. They are
mapped to the current Tower implementation as follows:

| Original action | Current object | Structural effect | Runtime effect |
|---|---|---|---|
| Split | `LegalSplitProposal` | Replace one old Mid cluster by its lineage descendants; project affected High paths | New Mid IDs are indexed and rendered |
| Merge | `LegalMergeProposal` | Replace two overlapping old Mid clusters by one lineage candidate; project affected High paths | Merged Mid ID is indexed and rendered |
| Promote | `LegalPromoteProposal` / `RankedPromoteProposal` | Add one legal, stable ordered Mid path as a High path | New High card and embedding are indexed |
| Prune / Downweight | `LifecycleUpdate` with `LifecycleAction.DOWNWEIGHT` | First round changes status only; no physical deletion | Retrieval subtracts the fixed `status_tie_epsilon` from the card cosine |

`ComplexRepartitionProposal` is not a fifth action. It is a coordinated
Split+Merge transaction used when several old Mid clusters jointly repartition
into several candidate clusters. The action plan records it under the
structural Split/Merge stage, while the lineage report preserves the detailed
transaction.

The four actions are represented by `RefinementActionPlan` and serialized under
the keys `split`, `merge`, `promote`, and `downweight`. A refinement round may
select at most one proposal for each key. Pareto ranking only orders legal
proposals; it does not invent a structural proposal or silently convert a
co-injected context bundle into independent per-card evidence.
