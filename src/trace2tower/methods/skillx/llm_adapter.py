from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole


@dataclass(frozen=True, slots=True)
class SkillXAdapterUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


class SkillXLLMAdapter:
    def __init__(
        self,
        runtime: CommonLLMRuntime,
        *,
        max_output_tokens: int,
        temperature: float,
        max_validation_attempts: int,
        retry_delay_seconds: float,
    ):
        if max_output_tokens <= 0 or max_validation_attempts <= 0:
            raise ValueError("SkillX LLM limits must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("SkillX retry delay cannot be negative")
        self.runtime = runtime
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_validation_attempts = max_validation_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.usage = SkillXAdapterUsage()

    async def ainvoke(
        self,
        messages: Sequence,
        regex_pattern: str | None = None,
        regex_extractor: Callable[[str], Any] | None = None,
        **kwargs,
    ) -> str:
        converted = [_message_record(message) for message in messages]
        system_prompt = next(
            (message["content"] for message in converted if message["role"] == "system"),
            "",
        )
        cache_key = f"skillx:{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"
        for attempt in range(self.max_validation_attempts):
            result = await self.runtime.chat(
                ModelRole.RENDERER,
                converted,
                temperature=self.temperature,
                max_output_tokens=int(kwargs.get("max_tokens", self.max_output_tokens)),
                prompt_cache_key=cache_key,
            )
            self.usage = SkillXAdapterUsage(
                calls=self.usage.calls + 1,
                input_tokens=self.usage.input_tokens + (result.usage.input_tokens or 0),
                output_tokens=self.usage.output_tokens + (result.usage.output_tokens or 0),
                cached_input_tokens=self.usage.cached_input_tokens
                + (result.usage.cached_input_tokens or 0),
            )
            content = result.content
            if content is None:
                valid = False
            elif regex_extractor is not None:
                valid = regex_extractor(content) is not None
            elif regex_pattern is not None:
                valid = re.search(regex_pattern, content) is not None
            else:
                valid = True
            if valid:
                return content
            if attempt + 1 < self.max_validation_attempts:
                await asyncio.sleep(self.retry_delay_seconds)
        raise ValueError("SkillX output failed its official parser validation")


def _message_record(message) -> dict[str, str]:
    if isinstance(message, tuple):
        role, content = message
    else:
        role = getattr(message, "type", getattr(message, "role", None))
        content = getattr(message, "content", None)
    role_map = {
        "human": "user",
        "user": "user",
        "ai": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    if role not in role_map or not isinstance(content, str):
        raise ValueError("unsupported SkillX message type")
    return {"role": role_map[role], "content": content}
