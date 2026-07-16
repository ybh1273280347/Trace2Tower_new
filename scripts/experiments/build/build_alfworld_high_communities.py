from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
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
from trace2tower.methods.trace2tower.alfworld_events import (
    ALFWORLD_EXCLUSIVE_PATH_EVENTS,
    alfworld_goal_events,
)
from trace2tower.methods.trace2tower.graph_retrieval import TowerGraphProfile
from trace2tower.methods.trace2tower.models import HighCommunity, HighPath, MidCluster
from trace2tower.methods.trace2tower.renderer import render_high_community_card
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard


def _representative_contexts(contexts: list[dict], limit: int = 8) -> list[dict]:
    representatives = {}
    for context in contexts:
        key = tuple(context["ordered_mid_ids"])
        representatives.setdefault(key, context)
    selected = list(representatives.values())[:limit]
    if len(selected) < limit:
        selected_ids = {context["trajectory_id"] for context in selected}
        selected.extend(
            context
            for context in contexts
            if context["trajectory_id"] not in selected_ids
        )
    return selected[:limit]


def _transition_summary(
    successful: list[dict],
    unsuccessful: list[dict],
) -> dict:
    positive = Counter()
    negative = Counter()
    for context in successful:
        mids = context["ordered_mid_ids"]
        positive.update(zip(mids, mids[1:]))
    for context in unsuccessful:
        mids = context["ordered_mid_ids"]
        negative.update(zip(mids, mids[1:]))
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
    profile = TowerGraphProfile.from_record(
        json.loads(options.profile.read_text(encoding="utf-8"))
    )
    contexts = build_trajectory_render_contexts(records, clusters)
    minimum_support = max(1, int(len(records) * options.min_support_ratio))

    distinctive_event_by_mid = {}
    for mid_id, counts in profile.mid_event_counts.items():
        event, count = max(counts.items(), key=lambda item: (item[1], item[0].value))
        if event in ALFWORLD_EXCLUSIVE_PATH_EVENTS and count / sum(counts.values()) >= 0.5:
            distinctive_event_by_mid[mid_id] = event

    successful_by_signature = defaultdict(list)
    failed_records = []
    for record in sorted(records, key=lambda item: item["trajectory_id"]):
        context = {
            "trajectory_id": record["trajectory_id"],
            **contexts[record["trajectory_id"]],
        }
        if float(record["primary_score"]) >= 0.999:
            signature = tuple(
                dict.fromkeys(
                    mid_id
                    for mid_id in context["ordered_mid_ids"]
                    if mid_id in distinctive_event_by_mid
                )
            )
            successful_by_signature[signature].append(context)
        else:
            failed_records.append(context)

    cards_by_id = {card.skill_id: card for card in mid_cards}
    reusable_high_cards = {}
    if options.output_cards.exists():
        existing_cards = json.loads(options.output_cards.read_text(encoding="utf-8"))
        reusable_high_cards = {
            card.skill_id: card
            for card in (
                HighSkillCard.from_record(item)
                for item in existing_cards.get("high_cards", ())
            )
        }
    paths_by_signature = defaultdict(set)
    for signature, successful in successful_by_signature.items():
        trajectory_ids = {context["trajectory_id"] for context in successful}
        for path in paths:
            if trajectory_ids.intersection(path.supporting_trajectory_ids):
                paths_by_signature[signature].add(path.path_id)

    runtime = CommonLLMRuntime(
        max_concurrency=options.concurrency,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    communities = []
    high_cards = []
    usages = []
    try:
        for signature in sorted(successful_by_signature):
            successful = successful_by_signature[signature]
            if len(successful) < minimum_support:
                continue
            required_events = frozenset(
                distinctive_event_by_mid[mid_id] for mid_id in signature
            )
            unsuccessful = [
                context
                for context in failed_records
                if alfworld_goal_events(context["goal"]) == required_events
            ]
            member_mid_ids = tuple(
                sorted(
                    {
                        mid_id
                        for context in successful
                        for mid_id in context["ordered_mid_ids"]
                    }
                )
            )
            digest_input = ",".join(signature or ("plain",))
            community_id = (
                "high_community_"
                + hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:12]
            )
            community = HighCommunity(
                community_id=community_id,
                member_mid_ids=member_mid_ids,
                member_path_ids=tuple(sorted(paths_by_signature[signature])),
                supporting_trajectory_ids=tuple(
                    sorted(context["trajectory_id"] for context in successful)
                ),
            )
            card = reusable_high_cards.get(community_id)
            if card is None:
                card, result = await render_high_community_card(
                    runtime,
                    Benchmark.ALFWORLD,
                    community_id,
                    tuple(cards_by_id[mid_id] for mid_id in member_mid_ids),
                    _representative_contexts(successful),
                    _representative_contexts(unsuccessful),
                    _transition_summary(successful, unsuccessful),
                )
            else:
                result = None
            communities.append(community)
            high_cards.append(card)
            usages.append(
                {
                    "community_id": community_id,
                    "signature_mid_ids": signature,
                    "signature_events": tuple(
                        distinctive_event_by_mid[mid_id].value for mid_id in signature
                    ),
                    "successful_context_count": len(successful),
                    "unsuccessful_context_count": len(unsuccessful),
                    "usage": asdict(result.usage) if result else None,
                }
            )
    finally:
        await runtime.close()

    write_json(
        options.output_structure,
        {
            "paths": [path.to_record() for path in paths],
            "communities": [community.to_record() for community in communities],
            "discovery": {
                "distinctive_event_by_mid": {
                    mid_id: event.value
                    for mid_id, event in sorted(distinctive_event_by_mid.items())
                },
                "minimum_support": minimum_support,
                "communities": usages,
            },
        },
    )
    write_json(
        options.output_cards,
        {
            "mid_cards": [card.to_record() for card in mid_cards],
            "high_cards": [card.to_record() for card in high_cards],
            "usage": usages,
        },
    )
    print(
        json.dumps(
            {
                "community_count": len(communities),
                "communities": usages,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-structure", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--min-support-ratio", type=float, default=0.02)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
