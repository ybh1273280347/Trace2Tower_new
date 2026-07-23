from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.components.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.core.trajectory import TrajectoryReader
from trace2tower.methods.trace2tower.inference.provider import (
    HighToMidSkillProvider,
)


async def main(options: argparse.Namespace) -> None:
    load_dotenv(options.env)
    trajectories = tuple(
        trajectory
        for trajectory in TrajectoryReader.read_episode_files(options.trajectories)
        if trajectory.primary_score < 1 and trajectory.steps
    )[: options.limit]
    runtime = CommonLLMRuntime(
        max_concurrency=options.concurrency,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    provider = HighToMidSkillProvider.from_path(
        runtime,
        options.tower,
        reference_high_top_k=options.high_top_k,
        skills_per_step=options.skills_per_step,
        max_mid_skills=options.max_mid_skills,
        mid_similarity_threshold=options.mid_threshold,
        rewrite_model_role=ModelRole(options.rewrite_model_role),
        rewrite_max_output_tokens=1200,
        rewrite_plan=options.rewrite_plan,
    )

    async def audit(trajectory):
        first = trajectory.steps[0]
        state = EnvironmentState(
            observation=first.observation,
            admissible_actions=first.admissible_actions,
            clickable_types=first.clickable_types,
            search_available=False,
            reward=0.0,
            done=False,
            valid_action=True,
        )
        selection = await provider.select_task(trajectory.task_goal, state)
        return {
            "sample_id": trajectory.sample_id,
            "task_goal": trajectory.task_goal,
            "source_run_score": trajectory.primary_score,
            "skill_ids": selection.skill_ids,
            "context": selection.context,
            "model_input_tokens": selection.model_input_tokens,
            "model_output_tokens": selection.model_output_tokens,
        }

    try:
        records = await asyncio.gather(*(audit(item) for item in trajectories))
    finally:
        await runtime.close()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"audited": len(records), "output": str(options.output)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--high-top-k", type=int, default=3)
    parser.add_argument("--skills-per-step", type=int, default=4)
    parser.add_argument("--max-mid-skills", type=int, default=8)
    parser.add_argument("--mid-threshold", type=float, default=0.45)
    parser.add_argument(
        "--rewrite-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rewrite-model-role",
        choices=(ModelRole.AGENT.value, ModelRole.RENDERER.value),
        default=ModelRole.RENDERER.value,
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
