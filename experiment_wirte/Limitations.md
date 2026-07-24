# 7 Limitations and Future Work

## 7.1 Benchmark and Interface Scope

Our evaluation covers two text-based interactive environments. ALFWorld provides a stable symbolic action space, while
WebShop combines search, attribute reasoning, and page-dependent decisions. These settings do not cover visual
observations, continuous control, changing tool schemas, or irreversible external actions. Evaluation on multimodal
embodied tasks, dialogue-centered tool use, and realistic software environments is needed to determine which graph
structures transfer across domains and which must be reconstructed for each environment.

## 7.2 Domain-Adapted Event Semantics

Trace2Tower does not use manually assigned task clusters, but its event extractor relies on a domain adapter specifying
which actions, entities, state changes, and outcomes should be preserved. An incomplete interface can omit distinctions
that later matter for retrieval, while an overly detailed interface can fragment recurring behavior into sparse
patterns. Future work should investigate automatic event-schema induction, uncertainty-aware extraction, and
provenance-preserving graph updates to reduce domain-specific engineering.

## 7.3 State-Conditioned Experience Access

The current runtime retrieves experience from the initial task description and exposes it before execution. This matches
ALFWorld, where the requested transformation largely identifies the required transition chain, but is less reliable when
decisive evidence appears only after interaction. In WebShop, observed products, attributes, and option states can change
whether the appropriate next step is verification, recovery, or termination. Future systems should re-query or revise
the active subgraph as the environment state evolves. Such mechanisms must control retrieval latency, context growth,
and the risk of continuing unnecessary checks after a goal condition has already been satisfied.

## 7.4 Feedback Optimization

Feedback-based optimization is demonstrated only on ALFWorld with one bounded action space consisting of Split, Merge,
Promote, and Downweight. The results do not establish convergence, long-term stability, or transfer of graph edits across
environments. Update quality also depends on whether the feedback pool represents the later deployment distribution.
Future work should study multiple update cycles on predeclared optimization sets, evaluate edits on temporally and
structurally separated tasks, and allow the optimizer to retain the current graph when evidence for modification is weak.
Latency and context cost could also be incorporated as explicit deployment objectives.

## 7.5 Model-Service Sensitivity

The reported pipeline uses remotely served models for skill authoring, plan rewriting, and task execution. Cross-model
experiments show that Towers transfer across Authors and Users, but the API-state study demonstrates that a fixed model
name and temperature do not fully determine behavior over time. This limits exact reproducibility, particularly for
methods that require the executor to interpret and compose hierarchical guidance. Replication with versioned open-weight
models, broader Author-User combinations, and temporally repeated evaluations would provide stronger control over this
source of variation.

## 7.6 Construction and Continual Scaling

Construction remains resource intensive even though Trace2Tower uses fewer GPT tokens than the closest multi-stage
baseline. The ALFWorld build requires 1,240 training trajectories and more than one million GPT chat tokens, and the
current pipeline produces a frozen hierarchy rather than maintaining it continuously. Incremental quotient-graph
maintenance, selective community reconstruction, targeted skill re-rendering, and smaller local Author models are
promising directions for incorporating growing experience pools without repeatedly rebuilding the full Tower.
