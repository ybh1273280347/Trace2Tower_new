from __future__ import annotations

import argparse
import asyncio
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
from trace2tower.methods.trace2tower.alfworld_events import alfworld_goal_events
from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.renderer import render_high_community_card
from trace2tower.methods.trace2tower.skills import MidSkillCard


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    records = load_skill_records(options.input)
    clusters = tuple(
        MidCluster.from_record(item)
        for item in json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    )
    contexts = build_trajectory_render_contexts(records, clusters)
    cards = json.loads(options.cards.read_text(encoding="utf-8"))
    mid_cards = tuple(MidSkillCard.from_record(item) for item in cards["mid_cards"])

    grouped = defaultdict(lambda: {"success": [], "failure": []})
    for record in sorted(records, key=lambda item: item["trajectory_id"]):
        goal = contexts[record["trajectory_id"]]["goal"]
        key = tuple(sorted(event.value for event in alfworld_goal_events(goal))) or ("plain",)
        outcome = "success" if float(record["primary_score"]) >= 0.999 else "failure"
        grouped[key][outcome].append(contexts[record["trajectory_id"]])

    successful = []
    unsuccessful = []
    for key in sorted(grouped):
        successful.extend(grouped[key]["success"][:4])
        unsuccessful.extend(grouped[key]["failure"][:4])

    positive = Counter()
    negative = Counter()
    for context in successful:
        positive.update(zip(context["ordered_mid_ids"], context["ordered_mid_ids"][1:]))
    for context in unsuccessful:
        negative.update(zip(context["ordered_mid_ids"], context["ordered_mid_ids"][1:]))
    transitions = sorted(set(positive) | set(negative))
    summary = {
        "transitions": [
            {
                "source": source,
                "target": target,
                "successful_count": positive[(source, target)],
                "unsuccessful_count": negative[(source, target)],
            }
            for source, target in transitions
        ]
    }

    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    try:
        card, result = await render_high_community_card(
            runtime,
            Benchmark.ALFWORLD,
            "high_community",
            mid_cards,
            successful,
            unsuccessful,
            summary,
        )
    finally:
        await runtime.close()

    write_json(
        options.output,
        {
            "card": card.to_record(),
            "successful_context_count": len(successful),
            "unsuccessful_context_count": len(unsuccessful),
            "usage": asdict(result.usage),
        },
    )
    lines = [
        f"# {card.name}",
        "",
        card.description,
        "",
        "## Procedure",
        "",
        *(f"{index}. {step}" for index, step in enumerate(card.procedure, 1)),
        "",
        "## Constraints",
        "",
        *(f"- {item}" for item in card.constraints),
        "",
    ]
    options.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(options.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
