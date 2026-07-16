from trace2tower.benchmarks.alfworld import parse_alfworld_observation_goal


def test_parse_alfworld_observation_goal_uses_environment_task() -> None:
    observation = (
        "-= Welcome to TextWorld, ALFRED! =-\n\n"
        "You are in the middle of a room.\n\n"
        "Your task is to: clean some ladle and put it in countertop.\n"
    )
    assert (
        parse_alfworld_observation_goal(observation)
        == "clean some ladle and put it in countertop."
    )
