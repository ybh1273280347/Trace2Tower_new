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
)
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.alfworld_task_adapter import AlfworldTaskAdapter
from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    extract_task_prototype,
)
from trace2tower.methods.trace2tower.models import (
    AlfworldEventType,
    HighCommunity,
    HighPath,
    MidCluster,
    PrimitiveAction,
)
from trace2tower.methods.trace2tower.renderer import (
    render_task_conditioned_high_card,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.task_conditioning import (
    SkillTaskCondition,
    TaskConditionProfile,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _transformation_event(transformation: str) -> AlfworldEventType | None:
    if transformation == "NONE":
        return None
    return {
        PrimitiveAction.CLEAN: AlfworldEventType.CLEAN_OBJECT,
        PrimitiveAction.HEAT: AlfworldEventType.HEAT_OBJECT,
        PrimitiveAction.COOL: AlfworldEventType.COOL_OBJECT,
        PrimitiveAction.TOGGLE: AlfworldEventType.TOGGLE_OBJECT,
        PrimitiveAction.SLICE: AlfworldEventType.SLICE_OBJECT,
    }[PrimitiveAction(transformation)]


def _event_community(structure: dict) -> dict[AlfworldEventType | None, dict]:
    communities = {
        item["community_id"]: item for item in structure["communities"]
    }
    by_event = {}
    for discovery in structure["discovery"]["communities"]:
        events = tuple(discovery["signature_events"])
        event = AlfworldEventType(events[0]) if events else None
        by_event[event] = communities[discovery["community_id"]]
    return by_event


def _select_graph_path(
    records: list[dict],
    parent: dict,
    paths_by_id: dict[str, HighPath],
) -> HighPath:
    trajectory_ids = {str(record["trajectory_id"]) for record in records}
    candidates = [paths_by_id[path_id] for path_id in parent["member_path_ids"]]
    if not candidates:
        raise ValueError(f"High community has no paths: {parent['community_id']}")
    return max(
        candidates,
        key=lambda path: (
            len(trajectory_ids.intersection(path.supporting_trajectory_ids)),
            path.contrastive_score,
            len(path.ordered_mid_ids),
            path.path_id,
        ),
    )


def _representative_contexts(
    records: list[dict],
    contexts: dict[str, dict],
    *,
    limit: int = 8,
) -> tuple[dict, ...]:
    selected = []
    seen_sequences = set()
    seen_samples = set()
    for record in sorted(records, key=lambda item: item["trajectory_id"]):
        context = contexts[record["trajectory_id"]]
        sequence = tuple(context["ordered_mid_ids"])
        sample_id = record["sample_id"]
        if sequence in seen_sequences or sample_id in seen_samples:
            continue
        selected.append(context)
        seen_sequences.add(sequence)
        seen_samples.add(sample_id)
        if len(selected) >= limit:
            return tuple(selected)
    for record in sorted(records, key=lambda item: item["trajectory_id"]):
        sample_id = record["sample_id"]
        if sample_id in seen_samples:
            continue
        selected.append(contexts[record["trajectory_id"]])
        seen_samples.add(sample_id)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _task_render_condition(records: list[dict]) -> dict:
    prototypes = [extract_task_prototype(record) for record in records]
    record = prototypes[0].to_record()
    source_counts = Counter(
        source
        for prototype in prototypes
        for source in set(prototype.source_receptacles)
    )
    record["source_candidates"] = [
        {
            "receptacle": source,
            "successful_trajectory_count": count,
            "successful_trajectory_ratio": count / len(prototypes),
        }
        for source, count in sorted(
            source_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return record


def _merge_task_specific_prior(
    card: HighSkillCard,
    prior: HighSkillCard | None,
) -> HighSkillCard:
    if prior is None:
        return card
    graph_guards = tuple(
        constraint
        for constraint in card.constraints
        if constraint not in prior.constraints
    )[:3]
    return HighSkillCard(
        card.skill_id,
        card.ordered_mid_ids,
        prior.name,
        prior.description,
        prior.procedure,
        tuple(dict.fromkeys((*prior.constraints, *graph_guards))),
        card.member_mid_ids,
    )


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    adapter = AlfworldTaskAdapter()
    groups: dict[str, list[dict]] = defaultdict(list)
    with options.trajectories.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prototype = extract_task_prototype(record)
            if prototype.canonical_goal:
                groups[prototype.canonical_goal].append(record)

    with options.preprocessed.open(encoding="utf-8") as source:
        preprocessed = [json.loads(line) for line in source if line.strip()]
    clusters = tuple(
        MidCluster.from_record(item)
        for item in json.loads(options.clusters.read_text(encoding="utf-8"))["clusters"]
    )
    contexts = build_trajectory_render_contexts(preprocessed, clusters)

    base_cards = json.loads(options.base_cards.read_text(encoding="utf-8"))
    structure = json.loads(options.base_structure.read_text(encoding="utf-8"))
    base_high_cards = {
        card.skill_id: card
        for card in (
            HighSkillCard.from_record(item) for item in base_cards["high_cards"]
        )
    }
    mid_cards = tuple(
        MidSkillCard.from_record(item) for item in base_cards["mid_cards"]
    )
    mid_cards_by_id = {card.skill_id: card for card in mid_cards}
    paths = tuple(HighPath.from_record(item) for item in structure["paths"])
    paths_by_id = {path.path_id: path for path in paths}
    event_communities = _event_community(structure)

    reusable_cards = {}
    if options.output_cards.exists():
        reusable_cards = {
            card.skill_id: card
            for card in (
                HighSkillCard.from_record(item)
                for item in json.loads(
                    options.output_cards.read_text(encoding="utf-8")
                ).get("high_cards", ())
            )
        }
    prior_cards = {}
    if options.prior_task_cards:
        prior_cards = {
            card.skill_id: card
            for card in (
                HighSkillCard.from_record(item)
                for item in json.loads(
                    options.prior_task_cards.read_text(encoding="utf-8")
                ).get("high_cards", ())
            )
        }

    specifications = []
    for goal, records in sorted(groups.items()):
        successful = [
            record
            for record in records
            if float(record.get("primary_score", 0.0)) >= options.success_threshold
        ]
        if len(successful) < options.min_support:
            continue
        unsuccessful = [record for record in records if record not in successful]
        prototype = extract_task_prototype(successful[0])
        event = _transformation_event(prototype.transformation)
        parent = event_communities.get(event)
        if parent is None:
            continue
        parent_card = base_high_cards[parent["community_id"]]
        selected_path = _select_graph_path(successful, parent, paths_by_id)
        community_id = (
            "high_task_" + hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]
        )
        specifications.append(
            {
                "community_id": community_id,
                "goal": goal,
                "prototype": prototype,
                "successful": successful,
                "unsuccessful": unsuccessful,
                "parent": parent,
                "parent_card": parent_card,
                "selected_path": selected_path,
            }
        )

    runtime = CommonLLMRuntime(
        max_concurrency=options.concurrency,
        max_attempts=3,
        timeout_seconds=180,
        retry_base_seconds=2,
    )
    try:
        pending = []
        for item in specifications:
            existing = reusable_cards.get(item["community_id"])
            if existing and existing.ordered_mid_ids == item["selected_path"].ordered_mid_ids:
                pending.append(None)
                continue
            pending.append(
                render_task_conditioned_high_card(
                    runtime,
                    Benchmark.ALFWORLD,
                    item["community_id"],
                    _task_render_condition(item["successful"]),
                    item["selected_path"],
                    item["parent_card"],
                    mid_cards_by_id,
                    item["parent"]["member_mid_ids"],
                    _representative_contexts(item["successful"], contexts),
                    _representative_contexts(item["unsuccessful"], contexts),
                )
            )
        rendered = await asyncio.gather(
            *(job for job in pending if job is not None)
        )
    finally:
        await runtime.close()

    rendered_iter = iter(rendered)
    high_cards = []
    communities = []
    discovery = []
    task_conditions = []
    usages = []
    for item, job in zip(specifications, pending, strict=True):
        if job is None:
            card = reusable_cards[item["community_id"]]
            usage = None
        else:
            card, result = next(rendered_iter)
            usage = asdict(result.usage)
        card = _merge_task_specific_prior(
            card,
            prior_cards.get(item["community_id"]),
        )
        successful = item["successful"]
        unsuccessful = item["unsuccessful"]
        parent = item["parent"]
        selected_path = item["selected_path"]
        prototype = item["prototype"]
        community_id = item["community_id"]
        high_cards.append(card)
        communities.append(
            HighCommunity(
                community_id=community_id,
                member_mid_ids=tuple(parent["member_mid_ids"]),
                member_path_ids=tuple(parent["member_path_ids"]),
                supporting_trajectory_ids=tuple(
                    sorted(record["trajectory_id"] for record in successful)
                ),
            )
        )
        task_conditions.append(
            SkillTaskCondition(
                community_id,
                adapter.profile_condition(prototype.to_record()),
            )
        )
        discovery.append(
            {
                "community_id": community_id,
                "parent_event_community_id": parent["community_id"],
                "canonical_goal": item["goal"],
                "successful_support": len(successful),
                "unsuccessful_support": len(unsuccessful),
                "prototype": prototype.to_record(),
                "selected_path_id": selected_path.path_id,
                "ordered_mid_ids": selected_path.ordered_mid_ids,
            }
        )
        usages.append({"community_id": community_id, "usage": usage})

    _write_json(
        options.output_cards,
        {
            "mid_cards": [card.to_record() for card in mid_cards],
            "high_cards": [card.to_record() for card in high_cards],
            "usage": usages,
        },
    )
    _write_json(
        options.output_structure,
        {
            "paths": [path.to_record() for path in paths],
            "communities": [community.to_record() for community in communities],
            "discovery": {
                "contract": "domain_task_condition_graph_render_v2",
                "min_support": options.min_support,
                "community_count": len(communities),
                "communities": discovery,
            },
        },
    )
    _write_json(
        options.output_profile,
        TaskConditionProfile(adapter.domain, tuple(task_conditions)).to_record(),
    )
    print(
        json.dumps(
            {
                "task_goal_count": len(groups),
                "community_count": len(communities),
                "covered_successful_trajectories": sum(
                    len(community.supporting_trajectory_ids)
                    for community in communities
                ),
                "rendered_card_count": sum(job is not None for job in pending),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--base-cards", type=Path, required=True)
    parser.add_argument("--base-structure", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--output-structure", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--prior-task-cards", type=Path)
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
