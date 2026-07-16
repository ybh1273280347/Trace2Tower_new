from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.task_conditioning import TaskConditionProfile
from trace2tower.semantic_index import SkillEmbeddingIndex


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    base = json.loads(options.base_index.read_text(encoding="utf-8"))
    profile = TaskConditionProfile.from_record(
        json.loads(options.profile.read_text(encoding="utf-8"))
    )
    skill_ids = tuple(item.skill_id for item in profile.skills)
    conditions = tuple(item.condition for item in profile.skills)
    texts = tuple(condition.retrieval_text for condition in conditions)
    text_hashes = tuple(
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    )
    hash_by_skill = dict(zip(skill_ids, text_hashes, strict=True))
    reusable_vectors = {}
    if options.output.exists():
        existing = SkillEmbeddingIndex.from_record(
            json.loads(options.output.read_text(encoding="utf-8"))["high_index"]
        )
        reusable_vectors = {
            skill_id: vector
            for skill_id, vector, text_hash in zip(
                existing.skill_ids,
                existing.vectors,
                existing.text_hashes,
                strict=True,
            )
            if hash_by_skill.get(skill_id) == text_hash
        }
    missing = tuple(
        (skill_id, text)
        for skill_id, text in zip(skill_ids, texts, strict=True)
        if skill_id not in reusable_vectors
    )
    new_vectors = {}
    usages = []
    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=2,
    )
    try:
        for start in range(0, len(missing), options.batch_size):
            batch = missing[start : start + options.batch_size]
            result = await runtime.embed([text for _, text in batch])
            new_vectors.update(
                zip(
                    (skill_id for skill_id, _ in batch),
                    result.vectors,
                    strict=True,
                )
            )
            usages.append(asdict(result.usage))
    finally:
        await runtime.close()

    high_index = SkillEmbeddingIndex(
        skill_ids=skill_ids,
        vectors=tuple(
            (reusable_vectors | new_vectors)[skill_id] for skill_id in skill_ids
        ),
        text_hashes=text_hashes,
    )
    payload = {
        "mid_index": base["mid_index"],
        "high_index": high_index.to_record(),
        "report": {
            "contract": "domain_task_condition_retrieval_v1",
            "high_count": len(skill_ids),
            "reused_high_embeddings": len(reusable_vectors),
            "new_high_embeddings": len(new_vectors),
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
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
