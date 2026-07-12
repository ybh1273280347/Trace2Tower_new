from __future__ import annotations

import json
import time

from trace2tower.benchmarks.models import BenchmarkEnvironment, EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.manifests import Benchmark, ManifestEntry
from trace2tower.results import EpisodeResult, FinishReason, MethodName
from trace2tower.trajectory import EpisodeTrajectory, StepRecord, TrajectoryWriter


class AgentEvaluator:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        trajectory_writer: TrajectoryWriter,
        *,
        temperature: float,
        max_output_tokens: int,
    ):
        self.runtime = runtime
        self.trajectory_writer = trajectory_writer
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    async def run_episode(
        self,
        *,
        entry: ManifestEntry,
        environment: BenchmarkEnvironment,
        run_id: str,
        method: MethodName,
        skill_context: str | None,
        shard_id: int,
        max_steps: int,
    ) -> EpisodeResult:
        started = time.perf_counter()
        episode = await environment.reset(entry)
        state = episode.state
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the benchmark task by calling exactly one provided tool each turn. "
                    "Use only actions shown as available."
                ),
            },
            {
                "role": "user",
                "content": self._prompt(episode.task_goal, state, skill_context),
            },
        ]
        trajectory_steps = []
        invalid_actions = 0
        input_tokens = 0
        output_tokens = 0
        token_usage_available = True
        finish_reason = FinishReason.TASK_LIMIT_REACHED

        try:
            for step_index in range(max_steps):
                llm_result = await self.runtime.chat(
                    ModelRole.AGENT,
                    messages,
                    tools=environment.tool_schemas,
                    tool_choice="required",
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                if llm_result.usage.input_tokens is None or llm_result.usage.output_tokens is None:
                    token_usage_available = False
                else:
                    input_tokens += llm_result.usage.input_tokens
                    output_tokens += llm_result.usage.output_tokens

                if len(llm_result.tool_calls) != 1:
                    invalid_actions += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": "No single executable tool call was found. Call one tool.",
                        }
                    )
                    trajectory_steps.append(
                        StepRecord(
                            step_index,
                            state.observation,
                            None,
                            None,
                            state.observation,
                            0,
                            False,
                            False,
                            state.admissible_actions,
                            state.clickable_types,
                        )
                    )
                    continue

                tool_call = llm_result.tool_calls[0]
                assistant_message = {
                    "role": "assistant",
                    "content": llm_result.content,
                    "tool_calls": [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                    ],
                }
                messages.append(assistant_message)
                try:
                    arguments = json.loads(tool_call.arguments)
                    next_state = await environment.execute(tool_call.name, arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = None
                    next_state = EnvironmentState(
                        "Invalid tool arguments.",
                        state.admissible_actions,
                        state.clickable_types,
                        state.search_available,
                        0,
                        False,
                        False,
                    )
                if not next_state.valid_action:
                    invalid_actions += 1
                trajectory_steps.append(
                    StepRecord(
                        step_index,
                        state.observation,
                        tool_call.name,
                        arguments,
                        next_state.observation,
                        next_state.reward,
                        next_state.done,
                        next_state.valid_action,
                        state.admissible_actions,
                        state.clickable_types,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.call_id,
                        "content": self._state_text(next_state),
                    }
                )
                state = next_state
                if state.done:
                    finish_reason = FinishReason.COMPLETED
                    break
        finally:
            await environment.close()

        primary_score = state.reward if state.done else 0.0
        success = bool(primary_score) if entry.benchmark is Benchmark.ALFWORLD else None
        result = EpisodeResult(
            run_id=run_id,
            benchmark=entry.benchmark,
            split=entry.split,
            method=method,
            sample_id=entry.sample_id,
            repeat_id=entry.repeat_id,
            shard_id=shard_id,
            primary_score=primary_score,
            success=success,
            steps=len(trajectory_steps),
            invalid_actions=invalid_actions,
            finish_reason=finish_reason,
            input_tokens=input_tokens if token_usage_available else None,
            output_tokens=output_tokens if token_usage_available else None,
            billable_tokens=None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            skill_ids=(),
            skill_context_chars=len(skill_context or ""),
        )
        self.trajectory_writer.write(
            EpisodeTrajectory(
                benchmark=entry.benchmark,
                split=entry.split,
                method=method,
                sample_id=entry.sample_id,
                repeat_id=entry.repeat_id,
                task_goal=episode.task_goal,
                steps=tuple(trajectory_steps),
                primary_score=primary_score,
                finish_reason=finish_reason,
            )
        )
        return result

    @classmethod
    def _prompt(
        cls, task_goal: str, state: EnvironmentState, skill_context: str | None
    ) -> str:
        sections = [f"Task:\n{task_goal}", cls._state_text(state)]
        if skill_context:
            sections.insert(1, f"# Retrieved Experience\n\n{skill_context}")
        return "\n\n".join(sections)

    @staticmethod
    def _state_text(state: EnvironmentState) -> str:
        available = "\n".join(f"- {action}" for action in state.admissible_actions)
        search = "yes" if state.search_available else "no"
        return (
            f"Observation:\n{state.observation}\n\n"
            f"Search available: {search}\nAvailable click/action values:\n{available}"
        )
