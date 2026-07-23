from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.components.llm_runtime import CommonLLMRuntime, EmbeddingResult, LLMUsage
from trace2tower.methods.trace2tower.core.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.inference.formatting import high_card_text, mid_card_text


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    Trace2TowerConfig.from_record(load_yaml(options.config))
    payload = json.loads(options.cards.read_text(encoding="utf-8"))
    mid_cards = tuple(MidSkillCard.from_record(item) for item in payload["mid_cards"])
    high_cards = tuple(HighSkillCard.from_record(item) for item in payload["high_cards"])
    if not mid_cards:
        raise ValueError("retrieval index requires at least one Mid card")
    mid_texts = {card.skill_id: mid_card_text(card) for card in mid_cards}
    high_texts = {card.skill_id: high_card_text(card) for card in high_cards}
    mid_hashes = {
        skill_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for skill_id, text in mid_texts.items()
    }
    high_hashes = {
        skill_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for skill_id, text in high_texts.items()
    }
    reusable_mid_vectors = {}
    reusable_high_vectors = {}
    embedding_checkpoint_path = options.output.with_name(
        f"{options.output.stem}.embedding-checkpoint.json"
    )
    embedding_checkpoint = (
        json.loads(embedding_checkpoint_path.read_text(encoding="utf-8")).get(
            "vectors_by_text_hash", {}
        )
        if embedding_checkpoint_path.exists()
        else {}
    )
    if options.output.exists():
        existing = json.loads(options.output.read_text(encoding="utf-8"))
        existing_mid = SkillEmbeddingIndex.from_record(existing["mid_index"])
        existing_high = SkillEmbeddingIndex.from_record(existing["high_index"])
        if existing_mid.text_hashes:
            reusable_mid_vectors = {
                skill_id: vector
                for skill_id, vector, text_hash in zip(
                    existing_mid.skill_ids,
                    existing_mid.vectors,
                    existing_mid.text_hashes,
                    strict=True,
                )
                if mid_hashes.get(skill_id) == text_hash
            }
        if existing_high.text_hashes:
            reusable_high_vectors = {
                skill_id: vector
                for skill_id, vector, text_hash in zip(
                    existing_high.skill_ids,
                    existing_high.vectors,
                    existing_high.text_hashes,
                    strict=True,
                )
                if high_hashes.get(skill_id) == text_hash
            }
    reusable_mid_vectors |= {
        skill_id: tuple(embedding_checkpoint[text_hash])
        for skill_id, text_hash in mid_hashes.items()
        if skill_id not in reusable_mid_vectors and text_hash in embedding_checkpoint
    }
    reusable_high_vectors |= {
        skill_id: tuple(embedding_checkpoint[text_hash])
        for skill_id, text_hash in high_hashes.items()
        if skill_id not in reusable_high_vectors and text_hash in embedding_checkpoint
    }
    missing_mid_ids = tuple(
        skill_id for skill_id in mid_texts if skill_id not in reusable_mid_vectors
    )
    missing_high_ids = tuple(
        skill_id for skill_id in high_texts if skill_id not in reusable_high_vectors
    )
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    try:
        mid_result = (
            await _embed_batches(
                runtime,
                [mid_texts[skill_id] for skill_id in missing_mid_ids],
                [mid_hashes[skill_id] for skill_id in missing_mid_ids],
                options.embedding_batch_size,
                options.embedding_batch_delay_seconds,
                embedding_checkpoint,
                embedding_checkpoint_path,
            )
            if missing_mid_ids
            else None
        )
        high_result = (
            await _embed_batches(
                runtime,
                [high_texts[skill_id] for skill_id in missing_high_ids],
                [high_hashes[skill_id] for skill_id in missing_high_ids],
                options.embedding_batch_size,
                options.embedding_batch_delay_seconds,
                embedding_checkpoint,
                embedding_checkpoint_path,
            )
            if missing_high_ids
            else None
        )
        new_mid_vectors = dict(
            zip(missing_mid_ids, mid_result.vectors if mid_result else (), strict=True)
        )
        new_high_vectors = dict(
            zip(missing_high_ids, high_result.vectors if high_result else (), strict=True)
        )
        all_mid_vectors = reusable_mid_vectors | new_mid_vectors
        all_high_vectors = reusable_high_vectors | new_high_vectors
        mid_index = SkillEmbeddingIndex(
            tuple(mid_texts),
            tuple(all_mid_vectors[skill_id] for skill_id in mid_texts),
            tuple(mid_hashes.values()),
        )
        high_index = SkillEmbeddingIndex(
            tuple(high_texts),
            tuple(all_high_vectors[skill_id] for skill_id in high_texts),
            tuple(high_hashes.values()),
        )
    finally:
        await runtime.close()

    report = {
        "cards": options.cards.as_posix(),
        "mid_count": len(mid_cards),
        "high_count": len(high_cards),
        "embedding_dimension": len(mid_index.vectors[0]),
        "reused_mid_embeddings": len(reusable_mid_vectors),
        "new_mid_embeddings": len(missing_mid_ids),
        "reused_high_embeddings": len(reusable_high_vectors),
        "new_high_embeddings": len(missing_high_ids),
        "mid_embedding_usage": asdict(mid_result.usage) if mid_result else None,
        "high_embedding_usage": asdict(high_result.usage) if high_result else None,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        options.output,
        {
            "mid_index": mid_index.to_record(),
            "high_index": high_index.to_record(),
            "report": report,
        },
    )
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return 0


async def _embed_batches(
    runtime: CommonLLMRuntime,
    texts: list[str],
    text_hashes: list[str],
    batch_size: int,
    delay_seconds: float,
    checkpoint: dict[str, list[float]],
    checkpoint_path: Path,
) -> EmbeddingResult:
    if batch_size <= 0 or delay_seconds < 0:
        raise ValueError("embedding batch settings are invalid")
    results = []
    if len(texts) != len(text_hashes):
        raise ValueError("embedding texts and hashes must align")
    for start in range(0, len(texts), batch_size):
        if start and delay_seconds:
            await asyncio.sleep(delay_seconds)
        result = await runtime.embed(texts[start : start + batch_size])
        results.append(result)
        for text_hash, vector in zip(
            text_hashes[start : start + batch_size], result.vectors, strict=True
        ):
            checkpoint[text_hash] = list(vector)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(checkpoint_path, {"vectors_by_text_hash": checkpoint})
    return EmbeddingResult(
        vectors=tuple(vector for result in results for vector in result.vectors),
        usage=LLMUsage(
            input_tokens=_sum_usage(result.usage.input_tokens for result in results),
            output_tokens=_sum_usage(result.usage.output_tokens for result in results),
            billable_tokens=_sum_usage(result.usage.billable_tokens for result in results),
            cached_input_tokens=_sum_usage(result.usage.cached_input_tokens for result in results),
            cache_write_input_tokens=_sum_usage(
                result.usage.cache_write_input_tokens for result in results
            ),
        ),
        latency_ms=sum(result.latency_ms for result in results),
    )


def _sum_usage(values) -> int | None:
    counts = tuple(value for value in values if value is not None)
    return sum(counts) if counts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-delay-seconds", type=float, default=0.0)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
