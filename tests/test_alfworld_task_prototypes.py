from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    extract_task_prototype,
    goal_destination,
    goal_target_object,
    goal_transformation,
    normalize_entity,
)


def _record(
    actions: list[str],
    goal: str = "clean the ladle and put it on the countertop.",
) -> dict:
    return {
        "steps": [
            {
                "observation": (
                    f"Your task is to: {goal}\n"
                )
                if index == 0
                else "",
                "action_arguments": {"action": action},
            }
            for index, action in enumerate(actions)
        ]
    }


def test_normalize_entity_removes_only_instance_number() -> None:
    assert normalize_entity("ladle 2") == "ladle"
    assert normalize_entity("counter top 12") == "counter top"


def test_goal_target_object_covers_official_goal_templates() -> None:
    assert goal_target_object("look at book under the desklamp.") == "book"
    assert goal_target_object("heat some egg and put it in garbagecan.") == "egg"
    assert goal_target_object("put a hot mug in cabinet.") == "mug"
    assert goal_target_object("find two spraybottle and put them in drawer.") == "spraybottle"


def test_goal_structure_normalizes_task_paraphrases() -> None:
    goal = "Place a rinsed fork on a countertop."
    assert goal_target_object(goal) == "fork"
    assert goal_transformation(goal) == "CLEAN"
    assert goal_destination(goal) == "countertop"


def test_extract_task_prototype_preserves_object_and_receptacles() -> None:
    prototype = extract_task_prototype(
        _record(
            [
                "go to drawer 3",
                "open drawer 3",
                "take ladle 2 from drawer 3",
                "go to sinkbasin 1",
                "clean ladle 2 with sinkbasin 1",
                "go to countertop 4",
                "move ladle 2 to countertop 4",
            ]
        )
    )
    assert prototype.target_object == "ladle"
    assert prototype.source_receptacles == ("drawer",)
    assert prototype.transformation == "CLEAN"
    assert prototype.transformation_devices == ("sinkbasin",)
    assert prototype.destination_receptacles == ("countertop",)
    assert prototype.canonical_goal == "clean the ladle and put it on the countertop."


def test_extract_plain_move_without_transformation() -> None:
    prototype = extract_task_prototype(
        _record(
            ["take cd 2 from drawer 4", "move cd 2 to safe 1"],
            "put a cd in safe.",
        )
    )
    assert prototype.transformation == "NONE"
    assert prototype.target_object == "cd"
    assert prototype.source_receptacles == ("drawer",)
    assert prototype.destination_receptacles == ("safe",)


def test_extract_light_task_as_object_conditioned_toggle() -> None:
    prototype = extract_task_prototype(
        {
            "steps": [
                {
                    "observation": "Your task is to: examine the alarmclock with the desklamp.\n",
                    "action_arguments": {"action": "use desklamp 1"},
                },
                {
                    "observation": "",
                    "action_arguments": {"action": "take alarmclock 2 from desk 2"},
                },
            ]
        }
    )
    assert prototype.transformation == "TOGGLE"
    assert prototype.target_object == "alarmclock"
    assert prototype.transformation_devices == ("desklamp",)


def test_canonical_goal_overrides_incidental_picked_object() -> None:
    prototype = extract_task_prototype(
        {
            "steps": [
                {
                    "observation": "Your task is to: look at book under the desklamp.\n",
                    "action_arguments": {"action": "take cd 1 from desk 1"},
                },
                {
                    "observation": "",
                    "action_arguments": {"action": "use desklamp 1"},
                },
            ]
        }
    )
    assert prototype.target_object == "book"
    assert prototype.source_receptacles == ()


def test_destination_falls_back_to_canonical_goal_after_early_success() -> None:
    prototype = extract_task_prototype(
        {
            "steps": [
                {
                    "observation": "Your task is to: cool a pan and put it on the diningtable.\n",
                    "action_arguments": {"action": "take pan 1 from stoveburner 2"},
                },
                {
                    "observation": "",
                    "action_arguments": {"action": "cool pan 1 with fridge 1"},
                },
            ]
        }
    )
    assert prototype.destination_receptacles == ("diningtable",)
