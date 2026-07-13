from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SkillXAdapterUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    validation_failures: int = 0
    transport_failures: int = 0


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
        self.validation_diagnostics: list[dict[str, Any]] = []
        self.transport_diagnostics: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: Sequence,
        regex_pattern: str | None = None,
        regex_extractor: Callable[[str], Any] | None = None,
        **kwargs,
    ) -> str:
        if self.transport_diagnostics:
            raise RuntimeError("SkillX transport circuit breaker is open")
        converted = [_message_record(message) for message in messages]
        system_prompt = next(
            (message["content"] for message in converted if message["role"] == "system"),
            "",
        )
        cache_key = f"skillx:{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"
        for attempt in range(self.max_validation_attempts):
            try:
                result = await self.runtime.chat(
                    ModelRole.RENDERER,
                    converted,
                    temperature=self.temperature,
                    max_output_tokens=int(
                        kwargs.get("max_tokens", self.max_output_tokens)
                    ),
                    prompt_cache_key=cache_key,
                )
            except Exception as exc:
                diagnostic = {
                    "prompt_sha256": hashlib.sha256(
                        system_prompt.encode()
                    ).hexdigest(),
                    "attempt": attempt + 1,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                }
                self.transport_diagnostics.append(diagnostic)
                self.usage = SkillXAdapterUsage(
                    calls=self.usage.calls,
                    input_tokens=self.usage.input_tokens,
                    output_tokens=self.usage.output_tokens,
                    cached_input_tokens=self.usage.cached_input_tokens,
                    validation_failures=self.usage.validation_failures,
                    transport_failures=self.usage.transport_failures + 1,
                )
                logger.error("SkillX transport failure: %s", diagnostic)
                raise
            self.usage = SkillXAdapterUsage(
                calls=self.usage.calls + 1,
                input_tokens=self.usage.input_tokens + (result.usage.input_tokens or 0),
                output_tokens=self.usage.output_tokens + (result.usage.output_tokens or 0),
                cached_input_tokens=self.usage.cached_input_tokens
                + (result.usage.cached_input_tokens or 0),
                validation_failures=self.usage.validation_failures,
                transport_failures=self.usage.transport_failures,
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
            text = content or ""
            self.validation_diagnostics.append(
                {
                    "prompt_sha256": hashlib.sha256(
                        system_prompt.encode()
                    ).hexdigest(),
                    "messages_sha256": hashlib.sha256(
                        json.dumps(
                            converted,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "attempt": attempt + 1,
                    "max_output_tokens": int(
                        kwargs.get("max_tokens", self.max_output_tokens)
                    ),
                    "finish_reason": result.finish_reason,
                    "output_tokens": result.usage.output_tokens,
                    "content_chars": len(text),
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "has_skill_open_tag": "<skill>" in text,
                    "has_skill_close_tag": "</skill>" in text,
                    "preview_start": text[:500],
                    "preview_end": text[-500:],
                    "messages": converted,
                    "content": text,
                }
            )
            if attempt + 1 < self.max_validation_attempts:
                await asyncio.sleep(self.retry_delay_seconds)
        self.usage = SkillXAdapterUsage(
            calls=self.usage.calls,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            cached_input_tokens=self.usage.cached_input_tokens,
            validation_failures=self.usage.validation_failures + 1,
            transport_failures=self.usage.transport_failures,
        )
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
