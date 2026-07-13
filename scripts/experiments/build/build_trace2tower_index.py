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

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.retrieval import (
    SkillEmbeddingIndex,
    high_card_text,
    mid_card_text,
    retrieve_tower,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    config_record = load_yaml(options.config)
    config = Trace2TowerConfig.from_record(config_record)
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
            await runtime.embed([mid_texts[skill_id] for skill_id in missing_mid_ids])
            if missing_mid_ids
            else None
        )
        high_result = (
            await runtime.embed([high_texts[skill_id] for skill_id in missing_high_ids])
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
        retrieval = None
        query_usage = None
        if options.query_goal is not None:
            if options.query_observation is None:
                raise ValueError("query goal requires --query-observation")
            query_result = await runtime.embed(
                [
                    options.query_goal,
                    f"{options.query_goal}\n{options.query_observation}",
                ]
            )
            selected = retrieve_tower(
                query_result.vectors[0],
                query_result.vectors[1],
                high_index,
                mid_index,
                {card.skill_id: card for card in high_cards},
                {card.skill_id: card for card in mid_cards},
                high_top_k=config.high_top_k,
                direct_mid_top_k=config.direct_mid_top_k,
                high_similarity_threshold=options.high_similarity_threshold,
            )
            retrieval = {
                "skill_ids": selected.skill_ids,
                "high_candidate": asdict(selected.high_candidate)
                if selected.high_candidate
                else None,
                "high_match": asdict(selected.high_match)
                if selected.high_match
                else None,
                "direct_mid_matches": [
                    asdict(match) for match in selected.direct_mid_matches
                ],
                "context": selected.context,
                "context_chars": len(selected.context),
            }
            query_usage = asdict(query_result.usage)
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
        "query_embedding_usage": query_usage,
        "retrieval": retrieval,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-goal")
    parser.add_argument("--query-observation")
    parser.add_argument("--high-similarity-threshold", type=float, default=-1.0)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
