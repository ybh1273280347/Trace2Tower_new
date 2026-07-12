from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole


async def probe() -> None:
    load_dotenv()
    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=2,
        timeout_seconds=60,
        retry_base_seconds=1,
    )
    try:
        result = await runtime.chat(
            ModelRole.AGENT,
            [{"role": "user", "content": "Call take_action with action set to look."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "take_action",
                        "description": "Take one environment action.",
                        "parameters": {
                            "type": "object",
                            "properties": {"action": {"type": "string"}},
                            "required": ["action"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            tool_choice="required",
            temperature=0,
            max_output_tokens=32,
        )
    finally:
        await runtime.close()

    if len(result.tool_calls) != 1 or result.tool_calls[0].name != "take_action":
        raise RuntimeError(f"unexpected tool calls: {result.tool_calls}")
    arguments = json.loads(result.tool_calls[0].arguments)
    if arguments != {"action": "look"}:
        raise RuntimeError(f"unexpected tool arguments: {arguments}")
    print(
        json.dumps(
            {
                "tool": result.tool_calls[0].name,
                "arguments": arguments,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "latency_ms": result.latency_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(probe())
