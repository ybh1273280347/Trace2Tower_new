from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from statistics import fmean

from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.results import MethodName


class SkillLevel(StrEnum):
    MID = "mid"
    HIGH = "high"


class SkillStatus(StrEnum):
    ACTIVE = "active"
    DOWNWEIGHTED = "downweighted"


class LifecycleAction(StrEnum):
    DOWNWEIGHT = "downweight"


@dataclass(frozen=True, slots=True, order=True)
class EpisodePairKey:
    benchmark: Benchmark
    sample_id: str
    repeat_id: int

    def to_record(self) -> dict:
        return {
            "benchmark": self.benchmark.value,
            "sample_id": self.sample_id,
            "repeat_id": self.repeat_id,
        }


@dataclass(frozen=True, slots=True)
class RefinementEpisode:
    run_id: str
    benchmark: Benchmark
    split: ExperimentSplit
    method: MethodName
    sample_id: str
    repeat_id: int
    primary_score: float
    steps: int
    billable_tokens: int | None
    skill_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.primary_score <= 1:
            raise ValueError("refinement episode score must be in [0, 1]")
        if self.steps < 0 or self.repeat_id < 0:
            raise ValueError("refinement episode counts must be non-negative")
        if self.billable_tokens is not None and self.billable_tokens < 0:
            raise ValueError("refinement billable tokens must be non-negative")
        if len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("refinement episode contains duplicate skill IDs")

    @property
    def pair_key(self) -> EpisodePairKey:
        return EpisodePairKey(self.benchmark, self.sample_id, self.repeat_id)

    @classmethod
    def from_record(cls, record: Mapping) -> RefinementEpisode:
        return cls(
            run_id=str(record["run_id"]),
            benchmark=Benchmark(record["benchmark"]),
            split=ExperimentSplit(record["split"]),
            method=MethodName(record["method"]),
            sample_id=str(record["sample_id"]),
            repeat_id=int(record["repeat_id"]),
            primary_score=float(record["primary_score"]),
            steps=int(record["steps"]),
            billable_tokens=(
                int(record["billable_tokens"])
                if record.get("billable_tokens") is not None
                else None
            ),
            skill_ids=tuple(record.get("skill_ids", ())),
        )


@dataclass(frozen=True, slots=True)
class RefinementEvidenceAudit:
    skill_methods: tuple[MethodName, ...]
    paired_episode_keys: tuple[EpisodePairKey, ...]
    missing_baseline_keys: tuple[EpisodePairKey, ...]
    duplicate_baseline_keys: tuple[EpisodePairKey, ...]
    duplicate_skill_keys: tuple[EpisodePairKey, ...]
    non_training_keys: tuple[EpisodePairKey, ...]
    invalid_baseline_method_keys: tuple[EpisodePairKey, ...]
    invalid_skill_method_keys: tuple[EpisodePairKey, ...]
    empty_skill_selection_keys: tuple[EpisodePairKey, ...]
    missing_baseline_billable_keys: tuple[EpisodePairKey, ...]
    missing_skill_billable_keys: tuple[EpisodePairKey, ...]
    unknown_skill_ids: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        issues = (
            self.missing_baseline_keys,
            self.duplicate_baseline_keys,
            self.duplicate_skill_keys,
            self.non_training_keys,
            self.invalid_baseline_method_keys,
            self.invalid_skill_method_keys,
            self.empty_skill_selection_keys,
            self.missing_baseline_billable_keys,
            self.missing_skill_billable_keys,
            self.unknown_skill_ids,
        )
        return (
            bool(self.paired_episode_keys)
            and len(self.skill_methods) == 1
            and not any(issues)
        )

    def require_complete(self) -> None:
        if not self.is_complete:
            raise ValueError("refinement evidence is incomplete; inspect the audit fields")

    def to_record(self) -> dict:
        key_fields = (
            "paired_episode_keys",
            "missing_baseline_keys",
            "duplicate_baseline_keys",
            "duplicate_skill_keys",
            "non_training_keys",
            "invalid_baseline_method_keys",
            "invalid_skill_method_keys",
            "empty_skill_selection_keys",
            "missing_baseline_billable_keys",
            "missing_skill_billable_keys",
        )
        record = {
            field: [key.to_record() for key in getattr(self, field)]
            for field in key_fields
        }
        record["skill_methods"] = [method.value for method in self.skill_methods]
        record["unknown_skill_ids"] = list(self.unknown_skill_ids)
        record["is_complete"] = self.is_complete
        return record


@dataclass(frozen=True, slots=True)
class RefinementExecutionContract:
    tower_snapshot_id: str
    skill_report_snapshot_id: str
    baseline_run_ids: tuple[str, ...]
    skill_run_ids: tuple[str, ...]
    baseline_agent_models: tuple[str, ...]
    skill_agent_model: str

    def to_record(self) -> dict:
        return {**asdict(self), "is_valid": True}


@dataclass(frozen=True, slots=True)
class PairedEpisodeEvidence:
    pair_key: EpisodePairKey
    baseline_run_id: str
    skill_run_id: str
    baseline_score: float
    skill_score: float
    paired_reward_gain: float
    baseline_steps: int
    skill_steps: int
    raw_step_saving: float
    guarded_step_saving: float
    baseline_billable_tokens: int | None
    skill_billable_tokens: int | None
    raw_cost_saving: float | None
    guarded_cost_saving: float | None
    injected_skill_ids: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "pair_key": self.pair_key.to_record(),
            "baseline_run_id": self.baseline_run_id,
            "skill_run_id": self.skill_run_id,
            "baseline_score": self.baseline_score,
            "skill_score": self.skill_score,
            "paired_reward_gain": self.paired_reward_gain,
            "baseline_steps": self.baseline_steps,
            "skill_steps": self.skill_steps,
            "raw_step_saving": self.raw_step_saving,
            "guarded_step_saving": self.guarded_step_saving,
            "baseline_billable_tokens": self.baseline_billable_tokens,
            "skill_billable_tokens": self.skill_billable_tokens,
            "raw_cost_saving": self.raw_cost_saving,
            "guarded_cost_saving": self.guarded_cost_saving,
            "injected_skill_ids": self.injected_skill_ids,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    performance_level: float
    paired_reward_gain: float
    guarded_step_saving: float
    guarded_cost_saving: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("Pareto objectives must be finite")

    @property
    def values(self) -> tuple[float, float, float, float]:
        return (
            self.performance_level,
            self.paired_reward_gain,
            self.guarded_step_saving,
            self.guarded_cost_saving,
        )


@dataclass(frozen=True, slots=True)
class SkillObjective:
    skill_id: str
    benchmark: Benchmark
    skill_level: SkillLevel
    refinement_round: int
    objective_vector: ObjectiveVector
    exposure_count: int
    paired_episode_keys: tuple[EpisodePairKey, ...]

    def __post_init__(self) -> None:
        if self.refinement_round <= 0 or self.exposure_count <= 0:
            raise ValueError("skill objective requires a positive round and exposure count")
        if self.exposure_count != len(self.paired_episode_keys):
            raise ValueError("skill exposure count must match paired episode provenance")
        if len(set(self.paired_episode_keys)) != len(self.paired_episode_keys):
            raise ValueError("skill objective contains duplicate episode provenance")


@dataclass(frozen=True, slots=True)
class RankedSkillObjective:
    skill_id: str
    benchmark: Benchmark
    skill_level: SkillLevel
    refinement_round: int
    objective_vector: ObjectiveVector
    exposure_count: int
    paired_episode_keys: tuple[EpisodePairKey, ...]
    pareto_front_rank: int
    dominated_by: tuple[str, ...]
    dominates: tuple[str, ...]

    def to_record(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "benchmark": self.benchmark.value,
            "skill_level": self.skill_level.value,
            "refinement_round": self.refinement_round,
            "objective_vector": asdict(self.objective_vector),
            "exposure_count": self.exposure_count,
            "paired_episode_keys": [
                key.to_record() for key in self.paired_episode_keys
            ],
            "pareto_front_rank": self.pareto_front_rank,
            "dominated_by": self.dominated_by,
            "dominates": self.dominates,
        }


@dataclass(frozen=True, slots=True)
class LegalSplitProposal:
    proposal_id: str
    source_skill_id: str


@dataclass(frozen=True, slots=True)
class LegalMergeProposal:
    proposal_id: str
    left_skill_id: str
    right_skill_id: str
    member_overlap: float
    centroid_drift: float

    def __post_init__(self) -> None:
        if self.left_skill_id >= self.right_skill_id:
            raise ValueError("merge skill IDs must be distinct and canonical")
        if not 0 <= self.member_overlap <= 1 or self.centroid_drift < 0:
            raise ValueError("merge structural metrics are invalid")


@dataclass(frozen=True, slots=True)
class MergePrioritization:
    eligible: tuple[LegalMergeProposal, ...]
    pareto_protected: tuple[LegalMergeProposal, ...]


@dataclass(frozen=True, slots=True)
class LegalPromoteProposal:
    path_id: str
    benchmark: Benchmark
    refinement_round: int
    child_mid_ids: tuple[str, ...]
    contrastive_path_score: float
    positive_support: float

    def __post_init__(self) -> None:
        if self.refinement_round <= 0 or len(self.child_mid_ids) < 2:
            raise ValueError("promote proposal requires a round and at least two children")
        if len(set(self.child_mid_ids)) != len(self.child_mid_ids):
            raise ValueError("promote proposal contains duplicate child Mid IDs")
        if not math.isfinite(self.contrastive_path_score):
            raise ValueError("promote contrastive score must be finite")
        if not 0 <= self.positive_support <= 1:
            raise ValueError("promote support must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RankedPromoteProposal:
    proposal: LegalPromoteProposal
    path_objective: ObjectiveVector
    child_exposure_count: int
    pareto_front_rank: int


@dataclass(frozen=True, slots=True)
class LifecycleUpdate:
    skill_id: str
    action: LifecycleAction
    previous_status: SkillStatus
    new_status: SkillStatus
    refinement_round: int
    pareto_front_rank: int

    def to_record(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "action": self.action.value,
            "previous_status": self.previous_status.value,
            "new_status": self.new_status.value,
            "refinement_round": self.refinement_round,
            "pareto_front_rank": self.pareto_front_rank,
        }


def audit_refinement_evidence(
    baseline_episodes: Sequence[RefinementEpisode],
    skill_episodes: Sequence[RefinementEpisode],
    skill_levels: Mapping[str, SkillLevel],
) -> RefinementEvidenceAudit:
    baseline_counts = Counter(episode.pair_key for episode in baseline_episodes)
    skill_counts = Counter(episode.pair_key for episode in skill_episodes)
    baseline_by_key = {episode.pair_key: episode for episode in baseline_episodes}
    skill_by_key = {episode.pair_key: episode for episode in skill_episodes}
    missing_baselines = set(skill_by_key) - set(baseline_by_key)
    paired_keys = set(skill_by_key) & set(baseline_by_key)
    all_episodes = (*baseline_episodes, *skill_episodes)
    return RefinementEvidenceAudit(
        skill_methods=tuple(sorted({episode.method for episode in skill_episodes})),
        paired_episode_keys=tuple(sorted(paired_keys)),
        missing_baseline_keys=tuple(sorted(missing_baselines)),
        duplicate_baseline_keys=tuple(
            sorted(key for key, count in baseline_counts.items() if count > 1)
        ),
        duplicate_skill_keys=tuple(
            sorted(key for key, count in skill_counts.items() if count > 1)
        ),
        non_training_keys=tuple(
            sorted(
                {
                    episode.pair_key
                    for episode in all_episodes
                    if episode.split is not ExperimentSplit.TRAIN
                }
            )
        ),
        invalid_baseline_method_keys=tuple(
            sorted(
                episode.pair_key
                for episode in baseline_episodes
                if episode.method is not MethodName.NO_SKILL
            )
        ),
        invalid_skill_method_keys=tuple(
            sorted(
                episode.pair_key
                for episode in skill_episodes
                if episode.method is MethodName.NO_SKILL
            )
        ),
        empty_skill_selection_keys=tuple(
            sorted(
                episode.pair_key for episode in skill_episodes if not episode.skill_ids
            )
        ),
        missing_baseline_billable_keys=tuple(
            sorted(
                key
                for key in paired_keys
                if baseline_by_key[key].billable_tokens is None
            )
        ),
        missing_skill_billable_keys=tuple(
            sorted(
                key for key in paired_keys if skill_by_key[key].billable_tokens is None
            )
        ),
        unknown_skill_ids=tuple(
            sorted(
                {
                    skill_id
                    for episode in skill_episodes
                    for skill_id in episode.skill_ids
                    if skill_id not in skill_levels
                }
            )
        ),
    )


def validate_execution_contract(
    *,
    tower_snapshot_id: str,
    benchmark: Benchmark,
    baseline_episodes: Sequence[RefinementEpisode],
    skill_episodes: Sequence[RefinementEpisode],
    baseline_metadata: Sequence[Mapping],
    skill_report: Mapping,
) -> RefinementExecutionContract:
    baseline_run_ids = {episode.run_id for episode in baseline_episodes}
    skill_run_ids = {episode.run_id for episode in skill_episodes}
    baseline_models = {record.get("agent_model") for record in baseline_metadata}
    metadata_run_ids = {record.get("run_id") for record in baseline_metadata}
    skill_methods = {episode.method.value for episode in skill_episodes}
    checks = (
        skill_report.get("snapshot_id") == tower_snapshot_id,
        skill_report.get("benchmark") == benchmark.value,
        skill_run_ids == {skill_report.get("run_id")},
        skill_methods == {skill_report.get("method")},
        metadata_run_ids == baseline_run_ids,
        all(record.get("method") == "no_skill" for record in baseline_metadata),
        all(record.get("benchmark") == benchmark.value for record in baseline_metadata),
        len(baseline_models) == 1,
        None not in baseline_models,
        skill_report.get("agent_model") in baseline_models,
    )
    if not all(checks):
        raise ValueError(
            "refinement run metadata does not bind to one fair execution contract"
        )
    return RefinementExecutionContract(
        tower_snapshot_id=tower_snapshot_id,
        skill_report_snapshot_id=str(skill_report["snapshot_id"]),
        baseline_run_ids=tuple(sorted(baseline_run_ids)),
        skill_run_ids=tuple(sorted(skill_run_ids)),
        baseline_agent_models=tuple(sorted(baseline_models)),
        skill_agent_model=str(skill_report["agent_model"]),
    )


def build_skill_objectives(
    baseline_episodes: Sequence[RefinementEpisode],
    skill_episodes: Sequence[RefinementEpisode],
    skill_levels: Mapping[str, SkillLevel],
    *,
    refinement_round: int,
) -> tuple[SkillObjective, ...]:
    audit = audit_refinement_evidence(baseline_episodes, skill_episodes, skill_levels)
    audit.require_complete()
    baseline_by_key = {episode.pair_key: episode for episode in baseline_episodes}
    exposures: dict[str, list[tuple[RefinementEpisode, RefinementEpisode]]] = (
        defaultdict(list)
    )
    for skill_episode in skill_episodes:
        baseline = baseline_by_key[skill_episode.pair_key]
        for skill_id in skill_episode.skill_ids:
            exposures[skill_id].append((skill_episode, baseline))

    objectives = []
    for skill_id in sorted(exposures):
        pairs = exposures[skill_id]
        performance = []
        reward_gains = []
        step_savings = []
        cost_savings = []
        pair_keys = []
        for skill_episode, baseline in pairs:
            performance.append(skill_episode.primary_score)
            reward_gains.append(
                skill_episode.primary_score - baseline.primary_score
            )
            raw_step_saving = (baseline.steps - skill_episode.steps) / max(
                baseline.steps, 1
            )
            raw_cost_saving = (
                baseline.billable_tokens - skill_episode.billable_tokens
            ) / max(baseline.billable_tokens, 1)
            score_decreased = skill_episode.primary_score < baseline.primary_score
            step_savings.append(
                min(raw_step_saving, 0) if score_decreased else raw_step_saving
            )
            cost_savings.append(
                min(raw_cost_saving, 0) if score_decreased else raw_cost_saving
            )
            pair_keys.append(skill_episode.pair_key)
        objectives.append(
            SkillObjective(
                skill_id=skill_id,
                benchmark=pairs[0][0].benchmark,
                skill_level=skill_levels[skill_id],
                refinement_round=refinement_round,
                objective_vector=ObjectiveVector(
                    performance_level=fmean(performance),
                    paired_reward_gain=fmean(reward_gains),
                    guarded_step_saving=fmean(step_savings),
                    guarded_cost_saving=fmean(cost_savings),
                ),
                exposure_count=len(pairs),
                paired_episode_keys=tuple(sorted(pair_keys)),
            )
        )
    return tuple(objectives)


def build_paired_episode_evidence(
    baseline_episodes: Sequence[RefinementEpisode],
    skill_episodes: Sequence[RefinementEpisode],
) -> tuple[PairedEpisodeEvidence, ...]:
    baseline_counts = Counter(episode.pair_key for episode in baseline_episodes)
    skill_counts = Counter(episode.pair_key for episode in skill_episodes)
    baseline_by_key = {episode.pair_key: episode for episode in baseline_episodes}
    skill_by_key = {episode.pair_key: episode for episode in skill_episodes}
    unique_pairs = tuple(
        sorted(
            key
            for key in set(baseline_by_key) & set(skill_by_key)
            if baseline_counts[key] == 1 and skill_counts[key] == 1
        )
    )
    evidence = []
    for key in unique_pairs:
        baseline = baseline_by_key[key]
        skill = skill_by_key[key]
        reward_gain = skill.primary_score - baseline.primary_score
        raw_step_saving = (baseline.steps - skill.steps) / max(baseline.steps, 1)
        guarded_step_saving = (
            min(raw_step_saving, 0) if reward_gain < 0 else raw_step_saving
        )
        if baseline.billable_tokens is None or skill.billable_tokens is None:
            raw_cost_saving = None
            guarded_cost_saving = None
        else:
            raw_cost_saving = (
                baseline.billable_tokens - skill.billable_tokens
            ) / max(baseline.billable_tokens, 1)
            guarded_cost_saving = (
                min(raw_cost_saving, 0) if reward_gain < 0 else raw_cost_saving
            )
        evidence.append(
            PairedEpisodeEvidence(
                pair_key=key,
                baseline_run_id=baseline.run_id,
                skill_run_id=skill.run_id,
                baseline_score=baseline.primary_score,
                skill_score=skill.primary_score,
                paired_reward_gain=reward_gain,
                baseline_steps=baseline.steps,
                skill_steps=skill.steps,
                raw_step_saving=raw_step_saving,
                guarded_step_saving=guarded_step_saving,
                baseline_billable_tokens=baseline.billable_tokens,
                skill_billable_tokens=skill.billable_tokens,
                raw_cost_saving=raw_cost_saving,
                guarded_cost_saving=guarded_cost_saving,
                injected_skill_ids=skill.skill_ids,
            )
        )
    return tuple(evidence)


def dominates(left: ObjectiveVector, right: ObjectiveVector) -> bool:
    comparisons = tuple(
        left_value - right_value
        for left_value, right_value in zip(left.values, right.values, strict=True)
    )
    return all(value >= 0 for value in comparisons) and any(
        value > 0 for value in comparisons
    )


def rank_skill_objectives(
    objectives: Sequence[SkillObjective],
) -> tuple[RankedSkillObjective, ...]:
    groups = defaultdict(list)
    for objective in objectives:
        scope = (
            objective.benchmark,
            objective.skill_level,
            objective.refinement_round,
        )
        groups[scope].append(objective)
    ranked = []
    for scope in sorted(groups):
        group = groups[scope]
        if len({objective.skill_id for objective in group}) != len(group):
            raise ValueError("duplicate skill objective within a Pareto scope")
        vectors = {
            objective.skill_id: objective.objective_vector for objective in group
        }
        ranks = _front_ranks(vectors)
        for objective in group:
            dominated_by = tuple(
                sorted(
                    skill_id
                    for skill_id, vector in vectors.items()
                    if skill_id != objective.skill_id
                    and dominates(vector, objective.objective_vector)
                )
            )
            dominates_ids = tuple(
                sorted(
                    skill_id
                    for skill_id, vector in vectors.items()
                    if skill_id != objective.skill_id
                    and dominates(objective.objective_vector, vector)
                )
            )
            ranked.append(
                RankedSkillObjective(
                    skill_id=objective.skill_id,
                    benchmark=objective.benchmark,
                    skill_level=objective.skill_level,
                    refinement_round=objective.refinement_round,
                    objective_vector=objective.objective_vector,
                    exposure_count=objective.exposure_count,
                    paired_episode_keys=objective.paired_episode_keys,
                    pareto_front_rank=ranks[objective.skill_id],
                    dominated_by=dominated_by,
                    dominates=dominates_ids,
                )
            )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                item.benchmark,
                item.skill_level,
                item.refinement_round,
                item.pareto_front_rank,
                item.skill_id,
            ),
        )
    )


def prioritize_splits(
    proposals: Sequence[LegalSplitProposal],
    ranked_skills: Mapping[str, RankedSkillObjective],
) -> tuple[LegalSplitProposal, ...]:
    _require_known_skills(
        (proposal.source_skill_id for proposal in proposals), ranked_skills
    )
    _require_one_scope(
        ranked_skills[proposal.source_skill_id] for proposal in proposals
    )
    return tuple(
        sorted(
            proposals,
            key=lambda proposal: _weak_skill_key(
                ranked_skills[proposal.source_skill_id]
            )
            + (proposal.proposal_id,),
        )
    )


def prioritize_merges(
    proposals: Sequence[LegalMergeProposal],
    ranked_skills: Mapping[str, RankedSkillObjective],
) -> MergePrioritization:
    _require_known_skills(
        (
            skill_id
            for proposal in proposals
            for skill_id in (proposal.left_skill_id, proposal.right_skill_id)
        ),
        ranked_skills,
    )
    _require_one_scope(
        ranked_skills[skill_id]
        for proposal in proposals
        for skill_id in (proposal.left_skill_id, proposal.right_skill_id)
    )
    protected = []
    eligible = []
    for proposal in proposals:
        left = ranked_skills[proposal.left_skill_id]
        right = ranked_skills[proposal.right_skill_id]
        if _scope(left) != _scope(right):
            raise ValueError("merge proposal crosses Pareto scopes")
        mutually_non_dominating_f1 = (
            left.pareto_front_rank == 1
            and right.pareto_front_rank == 1
            and proposal.right_skill_id not in left.dominates
            and proposal.left_skill_id not in right.dominates
        )
        (protected if mutually_non_dominating_f1 else eligible).append(proposal)
    eligible.sort(
        key=lambda proposal: (
            -proposal.member_overlap,
            proposal.centroid_drift,
            -(
                ranked_skills[proposal.left_skill_id].exposure_count
                + ranked_skills[proposal.right_skill_id].exposure_count
            ),
            proposal.left_skill_id,
            proposal.right_skill_id,
            proposal.proposal_id,
        )
    )
    return MergePrioritization(
        tuple(eligible),
        tuple(sorted(protected, key=lambda proposal: proposal.proposal_id)),
    )


def rank_promotions(
    proposals: Sequence[LegalPromoteProposal],
    ranked_skills: Mapping[str, RankedSkillObjective],
) -> tuple[RankedPromoteProposal, ...]:
    grouped = defaultdict(list)
    objective_by_path = {}
    exposure_by_path = {}
    for proposal in proposals:
        _require_known_skills(proposal.child_mid_ids, ranked_skills)
        children = [ranked_skills[skill_id] for skill_id in proposal.child_mid_ids]
        if any(child.skill_level is not SkillLevel.MID for child in children):
            raise ValueError("promote proposal children must be Mid skills")
        if any(
            child.benchmark is not proposal.benchmark
            or child.refinement_round != proposal.refinement_round
            for child in children
        ):
            raise ValueError("promote proposal crosses Pareto scopes")
        total_exposure = sum(child.exposure_count for child in children)
        values = tuple(
            sum(
                child.exposure_count * child.objective_vector.values[index]
                for child in children
            )
            / total_exposure
            for index in range(4)
        )
        objective_by_path[proposal.path_id] = ObjectiveVector(*values)
        exposure_by_path[proposal.path_id] = total_exposure
        grouped[(proposal.benchmark, proposal.refinement_round)].append(proposal)
    ranked = []
    for scope in sorted(grouped):
        group = grouped[scope]
        if len({proposal.path_id for proposal in group}) != len(group):
            raise ValueError("duplicate promote path within a Pareto scope")
        ranks = _front_ranks(
            {proposal.path_id: objective_by_path[proposal.path_id] for proposal in group}
        )
        ranked.extend(
            RankedPromoteProposal(
                proposal,
                objective_by_path[proposal.path_id],
                exposure_by_path[proposal.path_id],
                ranks[proposal.path_id],
            )
            for proposal in group
        )
    return tuple(
        sorted(
            ranked,
            key=lambda item: (
                item.proposal.benchmark,
                item.proposal.refinement_round,
                item.pareto_front_rank,
                -item.proposal.contrastive_path_score,
                -item.proposal.positive_support,
                item.proposal.path_id,
            ),
        )
    )


def select_promote(
    ranked_proposals: Sequence[RankedPromoteProposal],
) -> RankedPromoteProposal | None:
    scopes = {
        (item.proposal.benchmark, item.proposal.refinement_round)
        for item in ranked_proposals
    }
    if len(scopes) > 1:
        raise ValueError("promote selection requires one Pareto scope")
    front = [proposal for proposal in ranked_proposals if proposal.pareto_front_rank == 1]
    return min(
        front,
        key=lambda item: (
            -item.proposal.contrastive_path_score,
            -item.proposal.positive_support,
            item.proposal.path_id,
        ),
        default=None,
    )


def select_downweight(
    ranked_skills: Sequence[RankedSkillObjective],
    active_skill_ids: set[str],
) -> LifecycleUpdate:
    candidates = [
        skill for skill in ranked_skills if skill.skill_id in active_skill_ids
    ]
    if not candidates:
        raise ValueError("no ranked active skill is available for downweighting")
    scopes = {_scope(skill) for skill in candidates}
    if len(scopes) != 1:
        raise ValueError("downweight selection requires one Pareto scope")
    selected = min(candidates, key=_weak_skill_key)
    return LifecycleUpdate(
        skill_id=selected.skill_id,
        action=LifecycleAction.DOWNWEIGHT,
        previous_status=SkillStatus.ACTIVE,
        new_status=SkillStatus.DOWNWEIGHTED,
        refinement_round=selected.refinement_round,
        pareto_front_rank=selected.pareto_front_rank,
    )


def _front_ranks(vectors: Mapping[str, ObjectiveVector]) -> dict[str, int]:
    remaining = set(vectors)
    ranks = {}
    rank = 1
    while remaining:
        front = sorted(
            skill_id
            for skill_id in remaining
            if not any(
                dominates(vectors[other_id], vectors[skill_id])
                for other_id in remaining
                if other_id != skill_id
            )
        )
        if not front:
            raise RuntimeError("Pareto sorting produced no non-dominated front")
        for skill_id in front:
            ranks[skill_id] = rank
            remaining.remove(skill_id)
        rank += 1
    return ranks


def _scope(
    skill: RankedSkillObjective,
) -> tuple[Benchmark, SkillLevel, int]:
    return skill.benchmark, skill.skill_level, skill.refinement_round


def _weak_skill_key(skill: RankedSkillObjective) -> tuple:
    vector = skill.objective_vector
    return (
        -skill.pareto_front_rank,
        vector.paired_reward_gain,
        vector.performance_level,
        -skill.exposure_count,
        skill.skill_id,
    )


def _require_known_skills(
    skill_ids: Sequence[str],
    ranked_skills: Mapping[str, RankedSkillObjective],
) -> None:
    missing = sorted(set(skill_ids) - set(ranked_skills))
    if missing:
        raise ValueError(f"proposal references unranked skills: {missing}")


def _require_one_scope(skills: Sequence[RankedSkillObjective]) -> None:
    if len({_scope(skill) for skill in skills}) > 1:
        raise ValueError("proposal prioritization requires one Pareto scope")
