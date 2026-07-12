"""Shared OpenAI-compatible transport for Trace2Tower-owned model calls."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)


class ModelRole(StrEnum):
    EMBEDDING = "embedding"
    AGENT = "agent"
    RENDERER = "renderer"


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int | None
    output_tokens: int | None
    billable_tokens: int | None


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: LLMUsage
    latency_ms: int


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    usage: LLMUsage
    latency_ms: int


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls, role: ModelRole) -> EndpointConfig:
        prefix = role.value.upper()
        values = {
            name: os.getenv(f"{prefix}_{name}")
            for name in ("BASE_URL", "API_KEY", "MODEL")
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(f"missing {prefix} settings: {', '.join(missing)}")
        return cls(
            base_url=values["BASE_URL"],
            api_key=values["API_KEY"],
            model=values["MODEL"],
        )


class CommonLLMRuntime:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_attempts: int,
        timeout_seconds: float,
        retry_base_seconds: float,
    ):
        if max_concurrency <= 0 or max_attempts <= 0:
            raise ValueError("concurrency and attempts must be positive")
        self._configs = {role: EndpointConfig.from_env(role) for role in ModelRole}
        self._clients = {
            role: AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=timeout_seconds,
                max_retries=0,
            )
            for role, config in self._configs.items()
        }
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_attempts = max_attempts
        self._retry_base_seconds = retry_base_seconds

    async def chat(
        self,
        role: ModelRole,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 4096,
    ) -> ChatResult:
        if role is ModelRole.EMBEDDING:
            raise ValueError("embedding role does not support chat")

        request: dict[str, Any] = {
            "model": self._configs[role].model,
            "messages": list(messages),
        }
        if tools is not None:
            request["tools"] = list(tools)
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        if temperature is not None:
            request["temperature"] = temperature
        if role is ModelRole.AGENT:
            request["max_tokens"] = max_output_tokens
            request["extra_body"] = {
                "thinking": {"type": os.getenv("AGENT_THINKING", "disabled")}
            }
        else:
            request["max_completion_tokens"] = max_output_tokens
            request["reasoning_effort"] = os.getenv("RENDERER_REASONING_EFFORT", "none")

        started = time.perf_counter()
        response = await self._invoke(
            lambda: self._clients[role].chat.completions.create(**request)
        )
        message = response.choices[0].message
        tool_calls = tuple(
            ToolCall(
                call_id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in message.tool_calls or ()
        )
        return ChatResult(
            content=message.content,
            tool_calls=tool_calls,
            usage=self._usage(response.usage),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("embedding input must not be empty")
        role = ModelRole.EMBEDDING
        started = time.perf_counter()
        response = await self._invoke(
            lambda: self._clients[role].embeddings.create(
                model=self._configs[role].model,
                input=list(texts),
            )
        )
        vectors = tuple(
            tuple(item.embedding) for item in sorted(response.data, key=lambda item: item.index)
        )
        return EmbeddingResult(
            vectors=vectors,
            usage=self._usage(response.usage),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()

    async def _invoke(self, request: Callable[[], Awaitable[Any]]) -> Any:
        for attempt in range(self._max_attempts):
            try:
                async with self._semaphore:
                    return await request()
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt + 1 == self._max_attempts:
                    raise
            except APIStatusError as exc:
                if exc.status_code < 500 or attempt + 1 == self._max_attempts:
                    raise
            await asyncio.sleep(self._retry_base_seconds * (2**attempt))
        raise RuntimeError("unreachable retry state")

    @staticmethod
    def _usage(usage: Any) -> LLMUsage:
        if usage is None:
            return LLMUsage(None, None, None)
        return LLMUsage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            billable_tokens=None,
        )
