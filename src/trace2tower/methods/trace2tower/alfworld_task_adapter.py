from __future__ import annotations

from collections.abc import Mapping

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    canonical_goal,
    goal_destination,
    goal_target_object,
    goal_transformation,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard
from trace2tower.methods.trace2tower.task_conditioning import (
    TaskCompatibility,
    TaskCondition,
    TaskProperty,
)


class AlfworldTaskAdapter:
    domain = "alfworld"

    def extract_query(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> TaskCondition:
        goal = canonical_goal(state.observation) or task_goal
        return TaskCondition(
            task_text=goal,
            retrieval_text=goal,
            required_events=(goal_transformation(goal),),
            properties=(
                TaskProperty("target", (goal_target_object(goal),)),
                TaskProperty("destination", (goal_destination(goal),)),
            ),
        )

    def profile_condition(self, record: Mapping) -> TaskCondition:
        goal = str(record["canonical_goal"])
        return TaskCondition(
            task_text=goal,
            retrieval_text=goal,
            required_events=(str(record["transformation"]),),
            properties=(
                TaskProperty("target", (str(record["target_object"]),)),
                TaskProperty(
                    "destination",
                    tuple(str(value) for value in record["destination_receptacles"]),
                ),
                TaskProperty(
                    "device",
                    tuple(str(value) for value in record["transformation_devices"]),
                ),
            ),
        )

    def compatibility(
        self,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> TaskCompatibility:
        same_event = query.required_events == candidate.required_events
        same_target = query.values("target") == candidate.values("target")
        destination = query.values("destination")
        same_destination = bool(
            destination and set(destination) <= set(candidate.values("destination"))
        )
        if same_event and same_target and same_destination:
            return TaskCompatibility.EXACT
        if same_event and same_target:
            return TaskCompatibility.PARTIAL
        if same_event:
            return TaskCompatibility.WORKFLOW
        return TaskCompatibility.INCOMPATIBLE

    def bind(
        self,
        source_card: HighSkillCard,
        query: TaskCondition,
        candidate: TaskCondition,
    ) -> HighSkillCard:
        if self.compatibility(query, candidate) is TaskCompatibility.EXACT:
            return source_card
        target = query.values("target")[0]
        destination = query.values("destination")[0]
        transformation = query.required_events[0]
        device = next(iter(candidate.values("device")), "required appliance")
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
