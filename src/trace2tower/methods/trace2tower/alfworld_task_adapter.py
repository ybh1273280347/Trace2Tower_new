from __future__ import annotations

from collections.abc import Mapping

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    canonical_goal,
    goal_destination,
    goal_target_count,
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
    plan_rewrite_instructions = """
ALFWorld plan semantics:
- If the target's source is not explicitly observed, do not copy or guess a source location from a reference plan. Search each candidate location at most once and prefer already visible or open locations before closed ones.
- Avoid initial look or inventory actions when the initial state already provides that information. Take the exact target immediately when it appears and never substitute a related object type.
- Preserve navigation, receptacle state, possession, appliance or tool, transformation, quantity, and final-placement prerequisites required by the current task.
- Do not close an empty searched container unless closure is required by a later demonstrated action. Stop when the environment confirms the current objective.
"""

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
                TaskProperty("count", (str(goal_target_count(goal)),)),
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
                TaskProperty("count", (str(record.get("target_count", 1)),)),
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
        candidate_destination = candidate.values("destination")
        same_destination = (
            not destination and not candidate_destination
        ) or bool(
            destination and set(destination) <= set(candidate_destination)
        )
        same_count = query.values("count") == candidate.values("count")
        if same_event and same_target and same_destination and same_count:
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
        target = next(iter(query.values("target")), "target object")
        destination = next(iter(query.values("destination")), "")
        transformation = query.required_events[0]
        device = next(iter(candidate.values("device")), "required appliance")
        target_count = int(next(iter(query.values("count")), "1"))

        if transformation == "TOGGLE":
            procedure = (
                f"Search visible surfaces and containers until the exact {target} is found; "
                "open a closed candidate only when needed and do not revisit cleared locations.",
                f"Take the {target}, then locate and go to the {device} while keeping that "
                "same object held.",
                f"Activate the {device}, then examine the held {target} under the active light.",
                "Stop as soon as the environment confirms the examination objective.",
            )
            description = (
                f"Find the exact {target}, carry it to the {device}, activate the light, "
                "and complete the examination."
            )
        elif transformation == "NONE":
            quantity = "two distinct instances" if target_count == 2 else "one instance"
            procedure = (
                f"Search unvisited surfaces and containers for {quantity} of {target}; open "
                "a closed candidate only when needed and record locations already cleared.",
                f"Take an exact {target} instance when found; do not substitute a related object.",
                f"Go to the {destination}, open it only if placement requires it, and move "
                f"the held {target} there.",
                (
                    f"Repeat search, pickup, and placement for a different {target} until two "
                    "instances have been delivered; never remove the first completed instance."
                    if target_count == 2
                    else "Stop after the environment confirms the target is at the destination."
                ),
            )
            description = (
                f"Find and place {quantity} of {target} in the specified {destination}."
            )
        else:
            verb = transformation.casefold()
            procedure = (
                f"Search unvisited surfaces and containers until the exact {target} is visible; "
                "open a closed candidate only when needed and do not revisit cleared locations.",
                f"Take the {target} and keep that exact object held.",
                f"Go to the {device} and execute `{verb} {target} with {device}`; repair only "
                "a prerequisite explicitly reported missing by the environment.",
                f"Go to the {destination}, open it only if placement requires it, then execute "
                f"`move {target} to {destination}`.",
                "Stop after the environment confirms both the required state change and final placement.",
            )
            description = (
                f"Find the exact {target}, {verb} it with the {device}, and place that same "
                f"object in the {destination}."
            )

        return HighSkillCard(
            skill_id=source_card.skill_id,
            ordered_mid_ids=source_card.ordered_mid_ids,
            name=query.task_text.rstrip(".").capitalize(),
            description=description,
            procedure=procedure,
            constraints=source_card.constraints,
            member_mid_ids=source_card.member_mid_ids,
        )
