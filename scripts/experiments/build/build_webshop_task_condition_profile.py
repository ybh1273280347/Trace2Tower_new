from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2tower.methods.trace2tower.retrieval import high_card_text
from trace2tower.methods.trace2tower.task_conditioning import (
    SkillTaskCondition,
    TaskConditionProfile,
)
from trace2tower.methods.trace2tower.skills import HighSkillCard
from trace2tower.methods.trace2tower.webshop_task_adapter import WebshopTaskAdapter


def main(options: argparse.Namespace) -> int:
    tower = json.loads(options.tower.read_text(encoding="utf-8"))
    high_cards = tuple(
        HighSkillCard.from_record(record) for record in tower["high_cards"]
    )
    adapter = WebshopTaskAdapter()
    profile = TaskConditionProfile(
        adapter.domain,
        tuple(
            SkillTaskCondition(
                card.skill_id,
                adapter.profile_condition({"task_text": high_card_text(card)}),
            )
            for card in high_cards
        )
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(profile.to_record(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tower_snapshot_id": tower["snapshot_id"],
                "condition_count": len(profile.skills),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
