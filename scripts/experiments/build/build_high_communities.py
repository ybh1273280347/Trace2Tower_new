from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from scripts.experiments.build.build_trace2tower_skills import (
    build_trajectory_render_contexts,
    load_skill_records,
)
from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.high_communities import discover_high_communities
from trace2tower.methods.trace2tower.high_paths import trajectory_mid_sequences
from trace2tower.methods.trace2tower.models import HighPath, MidCluster
from trace2tower.methods.trace2tower.renderer import render_high_community_card
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard


def _representatives(contexts: list[dict], limit: int = 8) -> tuple[dict, ...]:
    by_sequence = {}
    for context in contexts:
        by_sequence.setdefault(tuple(context["ordered_mid_ids"]), context)
    selected = list(by_sequence.values())
    selected_ids = {context["trajectory_id"] for context in selected}
    selected.extend(
        context for context in contexts if context["trajectory_id"] not in selected_ids
    )
    return tuple(selected[:limit])


def _transition_summary(successful: list[dict], unsuccessful: list[dict]) -> dict:
    positive = Counter()
    negative = Counter()
    for context in successful:
        positive.update(zip(context["ordered_mid_ids"], context["ordered_mid_ids"][1:]))
    for context in unsuccessful:
        negative.update(zip(context["ordered_mid_ids"], context["ordered_mid_ids"][1:]))
    return {
        "transitions": [
            {
                "source": source,
                "target": target,
                "successful_count": positive[(source, target)],
                "unsuccessful_count": negative[(source, target)],
            }
            for source, target in sorted(set(positive) | set(negative))
        ]
    }


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    records = load_skill_records(options.input)
    clusters = tuple(
        MidCluster.from_record(item)
        for item in json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    )
    paths = tuple(
        HighPath.from_record(item)
        for item in json.loads(options.paths.read_text(encoding="utf-8"))["paths"]
    )
    card_payload = json.loads(options.cards.read_text(encoding="utf-8"))
    mid_cards = tuple(MidSkillCard.from_record(item) for item in card_payload["mid_cards"])
    mid_cards_by_id = {card.skill_id: card for card in mid_cards}
    discovery = discover_high_communities(
        records,
        clusters,
        paths,
        success_threshold=options.success_threshold,
    )
    contexts = build_trajectory_render_contexts(records, clusters)
    sequences = trajectory_mid_sequences(records, clusters)
    records_by_id = {str(record["trajectory_id"]): record for record in records}
    failed_ids = tuple(
        trajectory_id
        for trajectory_id, record in records_by_id.items()
        if float(record["primary_score"]) < options.success_threshold
    )
    reusable_cards = {}
    if options.output_cards.exists():
        reusable_cards = {
            card.skill_id: card
            for card in (
                HighSkillCard.from_record(item)
                for item in json.loads(options.output_cards.read_text(encoding="utf-8")).get(
                    "high_cards", ()
                )
            )
        }

    runtime = CommonLLMRuntime(
        max_concurrency=options.concurrency,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    high_cards = []
    usages = []
    try:
        for community in discovery.communities:
            successful = [
                {"trajectory_id": trajectory_id, **contexts[trajectory_id]}
                for trajectory_id in community.supporting_trajectory_ids
            ]
            member_transitions = {
                transition
                for context in successful
                for transition in zip(
                    context["ordered_mid_ids"], context["ordered_mid_ids"][1:]
                )
            }
            unsuccessful = sorted(
                (
                    {
                        "trajectory_id": trajectory_id,
                        "overlap": len(
                            member_transitions
                            & set(zip(sequences[trajectory_id], sequences[trajectory_id][1:]))
                        ),
                        **contexts[trajectory_id],
                    }
                    for trajectory_id in failed_ids
                ),
                key=lambda item: (-item["overlap"], item["trajectory_id"]),
            )
            card = reusable_cards.get(community.community_id)
            result = None
            if card is None:
                card, result = await render_high_community_card(
                    runtime,
                    options.benchmark,
                    community.community_id,
                    tuple(mid_cards_by_id[mid_id] for mid_id in community.member_mid_ids),
                    _representatives(successful),
                    _representatives(unsuccessful),
                    _transition_summary(successful, unsuccessful),
                )
            high_cards.append(card)
            usages.append(
                {
                    "community_id": community.community_id,
                    "successful_trajectory_count": len(successful),
                    "failure_candidate_count": len(unsuccessful),
                    "usage": asdict(result.usage) if result else None,
                }
            )
    finally:
        await runtime.close()

    write_json(
        options.output_structure,
        {
            "paths": [path.to_record() for path in paths],
            "communities": [community.to_record() for community in discovery.communities],
            "discovery": {
                "trajectory_ids": discovery.trajectory_ids,
                "labels": discovery.labels,
                "feature_count": discovery.feature_count,
                "graph_weight": discovery.graph_weight,
                "modularity": discovery.modularity,
                "rendering": usages,
            },
        },
    )
    write_json(
        options.output_cards,
        {
            "mid_cards": [card.to_record() for card in mid_cards],
            "high_cards": [card.to_record() for card in high_cards],
            "usage": [*card_payload.get("usage", ()), *usages],
        },
    )
    print(
        json.dumps(
            {
                "community_count": len(discovery.communities),
                "modularity": discovery.modularity,
                "communities": usages,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output-structure", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
