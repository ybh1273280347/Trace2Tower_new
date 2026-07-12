from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rollout_no_skill_train import load_yaml, write_json

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
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    try:
        mid_result = await runtime.embed([mid_card_text(card) for card in mid_cards])
        high_result = (
            await runtime.embed([high_card_text(card) for card in high_cards])
            if high_cards
            else None
        )
        mid_index = SkillEmbeddingIndex(
            tuple(card.skill_id for card in mid_cards), mid_result.vectors
        )
        high_index = SkillEmbeddingIndex(
            tuple(card.skill_id for card in high_cards),
            high_result.vectors if high_result else (),
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
            )
            retrieval = {
                "skill_ids": selected.skill_ids,
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
        "mid_embedding_usage": asdict(mid_result.usage),
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
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
