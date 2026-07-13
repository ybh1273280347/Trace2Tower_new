from __future__ import annotations

from collections.abc import Sequence

from trace2tower.trajectory import EpisodeTrajectory


def adapt_trajectory(trajectory: EpisodeTrajectory) -> dict:
    if not trajectory.steps:
        raise ValueError("SkillX adapter requires a non-empty trajectory")
    history = [
        {
            "role": "system",
            "content": (
                "Solve the benchmark task by calling exactly one provided tool each turn. "
                "Use only actions shown as available."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{trajectory.task_goal}\n\n"
                f"Observation:\n{trajectory.steps[0].observation}"
            ),
        },
    ]
    for step in trajectory.steps:
        call_id = f"{trajectory.trajectory_id}:call:{step.step_index}"
        if step.action_name is None:
            history.extend(
                (
                    {"role": "assistant", "content": "No valid tool call was produced."},
                    {"role": "user", "content": step.next_observation},
                )
            )
            continue
        history.extend(
            (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": step.action_name,
                            "arguments": step.action_arguments or {},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": step.next_observation,
                    "id": call_id,
                },
            )
        )
    return {
        "trajectory_id": trajectory.trajectory_id,
        "benchmark": trajectory.benchmark.value,
        "task_id": trajectory.sample_id,
        "user_task": trajectory.task_goal,
        "task_history": history,
        "reward": trajectory.primary_score,
        "metadata": {},
    }


def adapt_tool_schemas(tool_schemas: Sequence[dict]) -> dict[str, dict]:
    schemas = {}
    for schema in tool_schemas:
        function = schema.get("function")
        if schema.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("SkillX requires OpenAI function tool schemas")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in schemas:
            raise ValueError("SkillX tool schemas require unique function names")
        schemas[name] = function
    return schemas
