from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.semantic_index import SkillEmbeddingIndex


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    base = json.loads(options.base_index.read_text(encoding="utf-8"))
    structure = json.loads(options.structure.read_text(encoding="utf-8"))
    communities = structure["discovery"]["communities"]
    skill_ids = tuple(item["community_id"] for item in communities)
    texts = tuple(item["canonical_goal"] for item in communities)
    vectors = []
    usages = []
    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=2,
    )
    try:
        for start in range(0, len(texts), options.batch_size):
            result = await runtime.embed(texts[start : start + options.batch_size])
            vectors.extend(result.vectors)
            usages.append(asdict(result.usage))
    finally:
        await runtime.close()

    high_index = SkillEmbeddingIndex(
        skill_ids=skill_ids,
        vectors=tuple(vectors),
        text_hashes=tuple(
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ),
    )
    payload = {
        "mid_index": base["mid_index"],
        "high_index": high_index.to_record(),
        "report": {
            "contract": "canonical_goal_retrieval_key_v1",
            "high_count": len(skill_ids),
            "batch_size": options.batch_size,
            "usage": usages,
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["report"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))

