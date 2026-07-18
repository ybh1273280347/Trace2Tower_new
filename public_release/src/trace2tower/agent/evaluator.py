from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from trace2tower.core.environment import BenchmarkEnvironment, EnvironmentState
from trace2tower.agent.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.core.manifests import Benchmark, ManifestEntry
from trace2tower.core.results import EpisodeResult, FinishReason, MethodName
from trace2tower.core.trajectory import EpisodeTrajectory, StepRecord, TrajectoryWriter


@dataclass(frozen=True, slots=True)
class SkillSelection:
    skill_ids: tuple[str, ...]
    context: str
    model_input_tokens: int | None = 0
    model_output_tokens: int | None = 0
    context_skill_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("skill selection contains duplicate IDs")
        if self.context_skill_ids is None:
            object.__setattr__(self, "context_skill_ids", self.skill_ids)
        if len(set(self.context_skill_ids)) != len(self.context_skill_ids):
            raise ValueError("injected context contains duplicate skill IDs")
        if not set(self.context_skill_ids) <= set(self.skill_ids):
            raise ValueError("injected context references skills outside the selection")
        if self.skill_ids and not self.context:
            raise ValueError("a non-empty skill selection requires injected context")
        token_counts = (self.model_input_tokens, self.model_output_tokens)
        if any(value is not None and value < 0 for value in token_counts):
            raise ValueError("skill selection token counts must be non-negative")


SkillProvider = Callable[[str, EnvironmentState], Awaitable[SkillSelection]]


class AgentEvaluator:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        trajectory_writer: TrajectoryWriter,
        *,
        temperature: float,
        max_output_tokens: int,
        endpoint_role: ModelRole = ModelRole.AGENT,
    ):
        if endpoint_role is ModelRole.EMBEDDING:
            raise ValueError("agent evaluation requires a chat endpoint role")
        self.runtime = runtime
        self.trajectory_writer = trajectory_writer
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.endpoint_role = endpoint_role

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
        skill_ids: tuple[str, ...] = (),
        skill_provider: SkillProvider | None = None,
        refresh_skill_each_step: bool = False,
        state_skill_provider: SkillProvider | None = None,
    ) -> EpisodeResult:
        if skill_provider is not None and (skill_context or skill_ids):
            raise ValueError("skill_provider cannot be combined with precomputed skills")
        if refresh_skill_each_step and skill_provider is None:
            raise ValueError("dynamic skill retrieval requires a provider")
        if refresh_skill_each_step and state_skill_provider is not None:
            raise ValueError("use one dynamic skill retrieval contract")
        started = time.perf_counter()
        episode = await environment.reset(entry)
        state = episode.state
        trajectory_steps = []
        invalid_actions = 0
        finish_reason = FinishReason.TASK_LIMIT_REACHED

        try:
            selection = (
                await skill_provider(
                    episode.task_goal,
                    state,
                )
                if skill_provider and not refresh_skill_each_step
                else SkillSelection(skill_ids, skill_context or "")
            )
            task_selection = selection
            input_tokens = selection.model_input_tokens or 0
            output_tokens = selection.model_output_tokens or 0
            chat_input_tokens = 0
            chat_output_tokens = 0
            input_usage_available = selection.model_input_tokens is not None
            output_usage_available = selection.model_output_tokens is not None
            chat_input_usage_available = True
            chat_output_usage_available = True
            selected_skill_ids = list(selection.skill_ids)
            context_skill_ids = list(selection.context_skill_ids)
            injected_contexts = [selection.context] if selection.context else []
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
                    "content": self._prompt(
                        episode.task_goal,
                        state,
                        selection.context,
                    ),
                },
            ]
            for step_index in range(max_steps):
                dynamic_context = ""
                dynamic_provider = (
                    state_skill_provider
                    if state_skill_provider is not None
                    else skill_provider
                    if refresh_skill_each_step
                    else None
                )
                if dynamic_provider is not None:
                    state_selection = await dynamic_provider(
                        episode.task_goal,
                        state,
                    )
                    if state_selection.model_input_tokens is None:
                        input_usage_available = False
                    else:
                        input_tokens += state_selection.model_input_tokens
                    if state_selection.model_output_tokens is None:
                        output_usage_available = False
                    else:
                        output_tokens += state_selection.model_output_tokens
                    dynamic_context = state_selection.context
                    selection = (
                        state_selection
                        if refresh_skill_each_step
                        else SkillSelection(
                            tuple(
                                dict.fromkeys(
                                    (*task_selection.skill_ids, *state_selection.skill_ids)
                                )
                            ),
                            state_selection.context or task_selection.context,
                            state_selection.model_input_tokens,
                            state_selection.model_output_tokens,
                            tuple(
                                dict.fromkeys(
                                    (
                                        *task_selection.context_skill_ids,
                                        *state_selection.context_skill_ids,
                                    )
                                )
                            ),
                        )
                    )
                    selected_skill_ids.extend(
                        skill_id
                        for skill_id in selection.skill_ids
                        if skill_id not in selected_skill_ids
                    )
                    context_skill_ids.extend(
                        skill_id
                        for skill_id in selection.context_skill_ids
                        if skill_id not in context_skill_ids
                    )
                    if dynamic_context:
                        injected_contexts.append(dynamic_context)
                call_messages = messages
                if dynamic_context:
                    heading = (
                        "# Current Phase Skills\n\n"
                        "The task-level strategy in the initial prompt remains authoritative. "
                        "Use these Mid-level procedures and Low-level action templates only "
                        "to execute the current phase; never let them change the bound target, "
                        "skip a prerequisite, reorder the task strategy, or replace its "
                        "completion condition.\n\n"
                        if state_skill_provider is not None
                        else "# Retrieved Skills for the Current State\n\n"
                    )
                    call_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": f"{heading}{dynamic_context}",
                        },
                    ]
                llm_result = await self.runtime.chat(
                    self.endpoint_role,
                    call_messages,
                    tools=environment.tool_schemas,
                    tool_choice="required",
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                if llm_result.usage.input_tokens is None:
                    input_usage_available = False
                    chat_input_usage_available = False
                else:
                    input_tokens += llm_result.usage.input_tokens
                    chat_input_tokens += llm_result.usage.input_tokens
                if llm_result.usage.output_tokens is None:
                    output_usage_available = False
                    chat_output_usage_available = False
                else:
                    output_tokens += llm_result.usage.output_tokens
                    chat_output_tokens += llm_result.usage.output_tokens

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
                            selection.skill_ids,
                            selection.context_skill_ids,
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
                        selection.skill_ids,
                        selection.context_skill_ids,
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
        injected_context = "\x1e".join(injected_contexts)
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
            input_tokens=input_tokens if input_usage_available else None,
            output_tokens=output_tokens if output_usage_available else None,
            billable_tokens=None,
            latency_ms=round((time.perf_counter() - started) * 1000),
            skill_ids=tuple(selected_skill_ids),
            skill_context_chars=sum(len(context) for context in injected_contexts),
            context_skill_ids=tuple(context_skill_ids),
            skill_context_sha256=hashlib.sha256(injected_context.encode("utf-8")).hexdigest(),
            chat_input_tokens=(chat_input_tokens if chat_input_usage_available else None),
            chat_output_tokens=(chat_output_tokens if chat_output_usage_available else None),
        )
        self.trajectory_writer.write(
            EpisodeTrajectory(
                run_id=run_id,
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
    def _prompt(cls, task_goal: str, state: EnvironmentState, skill_context: str | None) -> str:
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
