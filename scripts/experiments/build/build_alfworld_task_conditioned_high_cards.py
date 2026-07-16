from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    extract_task_prototype,
)
from trace2tower.methods.trace2tower.alfworld_task_adapter import AlfworldTaskAdapter
from trace2tower.methods.trace2tower.models import (
    AlfworldEventType,
    HighCommunity,
    PrimitiveAction,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard
from trace2tower.methods.trace2tower.task_conditioning import (
    SkillTaskCondition,
    TaskConditionProfile,
)


def _normalized_action(action: str) -> str:
    words = [word for word in action.casefold().strip().split() if not word.isdigit()]
    return " ".join(words)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _successful_actions(record: dict) -> tuple[str, ...]:
    actions = []
    for step in record["steps"]:
        arguments = step.get("action_arguments", {})
        action = arguments.get("action") if isinstance(arguments, dict) else None
        if not isinstance(action, str) or not step.get("valid_action", True):
            continue
        normalized = _normalized_action(action)
        if normalized and (not actions or actions[-1] != normalized):
            actions.append(normalized)
    return tuple(actions)


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


def main(options: argparse.Namespace) -> int:
    adapter = AlfworldTaskAdapter()
    groups: dict[str, list[dict]] = defaultdict(list)
    with options.trajectories.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if float(record.get("primary_score", 0.0)) < options.success_threshold:
                continue
            prototype = extract_task_prototype(record)
            if prototype.canonical_goal:
                groups[prototype.canonical_goal].append(record)

    base_cards = json.loads(options.base_cards.read_text(encoding="utf-8"))
    structure = json.loads(options.base_structure.read_text(encoding="utf-8"))
    base_high_cards = {
        card.skill_id: card
        for card in (
            HighSkillCard.from_record(item) for item in base_cards["high_cards"]
        )
    }
    event_communities = _event_community(structure)
    high_cards = []
    communities = []
    discovery = []
    task_conditions = []
    for goal, records in sorted(groups.items()):
        if len(records) < options.min_support:
            continue
        representative = min(records, key=lambda record: (len(_successful_actions(record)), record["trajectory_id"]))
        prototype = extract_task_prototype(representative)
        event = _transformation_event(prototype.transformation)
        parent = event_communities.get(event)
        if parent is None:
            continue
        parent_card = base_high_cards[parent["community_id"]]
        community_id = "high_task_" + hashlib.sha256(goal.encode("utf-8")).hexdigest()[:12]
        actions = _successful_actions(representative)
        high_cards.append(
            HighSkillCard(
                skill_id=community_id,
                ordered_mid_ids=(),
                name=goal.rstrip(".").capitalize(),
                description=f"Concrete task prototype: {goal}",
                procedure=tuple(f"Execute `{action}` when its objects are visible." for action in actions),
                constraints=parent_card.constraints,
                member_mid_ids=tuple(parent["member_mid_ids"]),
            )
        )
        communities.append(
            HighCommunity(
                community_id=community_id,
                member_mid_ids=tuple(parent["member_mid_ids"]),
                member_path_ids=tuple(parent["member_path_ids"]),
                supporting_trajectory_ids=tuple(
                    sorted(record["trajectory_id"] for record in records)
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
                "canonical_goal": goal,
                "support": len(records),
                "prototype": prototype.to_record(),
                "representative_trajectory_id": representative["trajectory_id"],
                "normalized_actions": actions,
            }
        )

    _write_json(
        options.output_cards,
        {
            "mid_cards": [
                MidSkillCard.from_record(item).to_record()
                for item in base_cards["mid_cards"]
            ],
            "high_cards": [card.to_record() for card in high_cards],
            "usage": [],
        },
    )
    _write_json(
        options.output_structure,
        {
            "paths": structure["paths"],
            "communities": [community.to_record() for community in communities],
            "discovery": {
                "contract": "domain_task_condition_v1",
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
                    len(community.supporting_trajectory_ids) for community in communities
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--base-cards", type=Path, required=True)
    parser.add_argument("--base-structure", type=Path, required=True)
    parser.add_argument("--output-cards", type=Path, required=True)
    parser.add_argument("--output-structure", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--min-support", type=int, default=4)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    raise SystemExit(main(parser.parse_args()))
