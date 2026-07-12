from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def run_probe(name: str, request: Callable[[], Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        value = request()
        return {
            "name": name,
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "result": value,
        }
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> int:
    load_dotenv()

    embedding_client = OpenAI(
        base_url=setting("EMBEDDING_BASE_URL"),
        api_key=setting("EMBEDDING_API_KEY"),
    )
    agent_client = OpenAI(
        base_url=setting("AGENT_BASE_URL"),
        api_key=setting("AGENT_API_KEY"),
    )
    renderer_client = OpenAI(
        base_url=setting("RENDERER_BASE_URL"),
        api_key=setting("RENDERER_API_KEY"),
    )

    probes = [
        run_probe(
            "embedding",
            lambda: len(
                embedding_client.embeddings.create(
                    model=setting("EMBEDDING_MODEL"),
                    input=["Trace2Tower endpoint probe"],
                ).data[0].embedding
            ),
        ),
        run_probe(
            "agent",
            lambda: agent_client.chat.completions.create(
                model=setting("AGENT_MODEL"),
                messages=[{"role": "user", "content": "Reply with OK only."}],
                max_tokens=8,
                temperature=0,
                extra_body={"thinking": {"type": setting("AGENT_THINKING")}},
            ).choices[0].message.content,
        ),
        run_probe(
            "renderer",
            lambda: renderer_client.chat.completions.create(
                model=setting("RENDERER_MODEL"),
                messages=[{"role": "user", "content": "Reply with OK only."}],
                max_completion_tokens=8,
                reasoning_effort=setting("RENDERER_REASONING_EFFORT"),
            ).choices[0].message.content,
        ),
    ]
    print(json.dumps(probes, ensure_ascii=False, indent=2))
    return 0 if all(probe["ok"] for probe in probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

