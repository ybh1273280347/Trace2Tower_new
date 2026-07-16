from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    canonical_goal,
    goal_destination,
    goal_target_object,
    goal_transformation,
)
from trace2tower.methods.trace2tower.object_conditioned_retrieval import (
    ObjectConditionedHighProfile,
    retrieve_object_conditioned_high,
)
from trace2tower.methods.trace2tower.retrieval import format_tower_context
from trace2tower.methods.trace2tower.skills import HighSkillCard
from trace2tower.methods.trace2tower.tower import TowerSnapshot


class ObjectConditionedHighProvider:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        snapshot: TowerSnapshot,
        profiles: dict[str, ObjectConditionedHighProfile],
        *,
        similarity_threshold: float = -1.0,
        allow_query_conditioning: bool = False,
    ):
        snapshot.require_complete()
        if set(profiles) != set(snapshot.high_index.skill_ids):
            raise ValueError("object-conditioned profile does not bind to the Tower")
        self.runtime = runtime
        self.snapshot = snapshot
        self.profiles = profiles
        self.similarity_threshold = similarity_threshold
        self.allow_query_conditioning = allow_query_conditioning
        self.high_cards = {card.skill_id: card for card in snapshot.high_cards}

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        profile_path: Path,
        **kwargs,
    ) -> ObjectConditionedHighProvider:
        snapshot = TowerSnapshot.from_record(
            json.loads(snapshot_path.read_text(encoding="utf-8"))
        )
        structure = json.loads(profile_path.read_text(encoding="utf-8"))
        profiles = {
            item["community_id"]: ObjectConditionedHighProfile.from_record(
                item["prototype"]
            )
            for item in structure["discovery"]["communities"]
        }
        return cls(runtime, snapshot, profiles, **kwargs)

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        goal = canonical_goal(state.observation) or task_goal
        query = await self.runtime.embed([goal])
        match = retrieve_object_conditioned_high(
            query.vectors[0],
            self.snapshot.high_index,
            self.profiles,
            target_object=goal_target_object(goal),
            transformation=goal_transformation(goal),
            destination=goal_destination(goal),
            similarity_threshold=self.similarity_threshold,
        )
        card = self.high_cards[match.skill_id] if match else None
        if card is None and self.allow_query_conditioning:
            ranked = self.snapshot.high_index.search(
                query.vectors[0], len(self.snapshot.high_index.skill_ids)
            )
            target = goal_target_object(goal)
            transformation = goal_transformation(goal)
            destination = goal_destination(goal)
            fallback = next(
                (
                    candidate
                    for candidate in ranked
                    if self.profiles[candidate.skill_id].target_object == target
                    and self.profiles[candidate.skill_id].transformation == transformation
                ),
                None,
            ) or next(
                (
                    candidate
                    for candidate in ranked
                    if self.profiles[candidate.skill_id].transformation == transformation
                ),
                None,
            )
            if fallback and fallback.cosine_similarity >= self.similarity_threshold:
                card = self._condition_card(
                    self.high_cards[fallback.skill_id],
                    target,
                    transformation,
                    destination,
                )
        return SkillSelection(
            skill_ids=(card.skill_id,) if card else (),
            context=format_tower_context(card, ()),
            model_input_tokens=query.usage.input_tokens,
            model_output_tokens=0,
            context_skill_ids=(card.skill_id,) if card else (),
        )

    def _condition_card(
        self,
        source_card: HighSkillCard,
        target: str,
        transformation: str,
        destination: str,
    ) -> HighSkillCard:
        devices = Counter(
            device
            for profile in self.profiles.values()
            if profile.transformation == transformation
            for device in profile.transformation_devices
        )
        device = devices.most_common(1)[0][0] if devices else "required appliance"
        verb = transformation.casefold()
        return HighSkillCard(
            skill_id=source_card.skill_id,
            ordered_mid_ids=(),
            name=f"{verb.capitalize()} {target} and place it in {destination}",
            description=(
                f"Concrete task: {verb} the {target} with the {device}, then place "
                f"that same {target} in the {destination}."
            ),
            procedure=(
                f"Search surfaces and containers until the {target} is visible.",
                f"Take the {target} and keep that exact object held.",
                f"Go to the {device} and execute `{verb} {target} with {device}`.",
                f"Go to the {destination}; open it first only if placement requires it.",
                f"Execute `move {target} to {destination}` and stop after successful placement.",
            ),
            constraints=source_card.constraints,
            member_mid_ids=source_card.member_mid_ids,
        )

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        return SkillSelection((), "", 0, 0, ())
