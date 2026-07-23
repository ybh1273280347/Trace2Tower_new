from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.methods.trace2tower.artifacts.tower import (
    TowerSnapshot,
    TowerSourceHashes,
    build_tower_snapshot,
    sha256_file,
)
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard


def add_completion_guards(
    base_cards: tuple[HighSkillCard, ...],
    stateful_cards: tuple[HighSkillCard, ...],
) -> tuple[HighSkillCard, ...]:
    stateful_by_id = {card.skill_id: card for card in stateful_cards}
    if {card.skill_id for card in base_cards} != set(stateful_by_id):
        raise ValueError("base and stateful High card IDs differ")

    patched = []
    for card in base_cards:
        stateful_card = stateful_by_id[card.skill_id]
        if card.ordered_mid_ids != stateful_card.ordered_mid_ids:
            raise ValueError(f"High path changed: {card.skill_id}")
        completion_guard = stateful_card.procedure[-1].strip()
        if not completion_guard or not any(
            token in completion_guard.casefold() for token in ("buy", "purchase")
        ):
            raise ValueError(f"High card has no purchase completion guard: {card.skill_id}")
        if completion_guard in card.constraints:
            raise ValueError(f"completion guard already exists: {card.skill_id}")
        patched.append(
            replace(card, constraints=(*card.constraints, completion_guard))
        )
    return tuple(patched)


def main(options: argparse.Namespace) -> int:
    base = TowerSnapshot.from_record(
        json.loads(options.base_tower.read_text(encoding="utf-8"))
    )
    stateful = TowerSnapshot.from_record(
        json.loads(options.stateful_tower.read_text(encoding="utf-8"))
    )
    if (
        base.benchmark != stateful.benchmark
        or base.version != stateful.version
        or base.config != stateful.config
        or base.training_trajectory_ids != stateful.training_trajectory_ids
        or base.mid_clusters != stateful.mid_clusters
        or base.high_paths != stateful.high_paths
        or base.mid_cards != stateful.mid_cards
    ):
        raise ValueError("base and stateful snapshots differ outside rendered High cards")

    high_cards = add_completion_guards(base.high_cards, stateful.high_cards)
    options.output_dir.mkdir(parents=True, exist_ok=False)
    cards_path = options.output_dir / "rendered-cards.json"
    write_json(
        cards_path,
        {
            "mid_cards": [card.to_record() for card in base.mid_cards],
            "high_cards": [card.to_record() for card in high_cards],
        },
    )
    snapshot = build_tower_snapshot(
        version=base.version,
        benchmark=base.benchmark,
        config=base.config,
        training_trajectory_ids=base.training_trajectory_ids,
        source_hashes=TowerSourceHashes(
            preprocessed_trajectories=base.source_hashes.preprocessed_trajectories,
            clusters=base.source_hashes.clusters,
            high_paths=base.source_hashes.high_paths,
            rendered_cards=sha256_file(cards_path),
            retrieval_index=base.source_hashes.retrieval_index,
        ),
        low_skills=base.low_skills,
        mid_clusters=base.mid_clusters,
        high_paths=base.high_paths,
        mid_cards=base.mid_cards,
        high_cards=high_cards,
        mid_index=base.mid_index,
        high_index=base.high_index,
        high_communities=base.high_communities,
    )
    tower_path = options.output_dir / "tower.json"
    write_json(tower_path, snapshot.to_record())
    write_json(
        options.output_dir / "provenance.json",
        {
            "evidence_role": "diagnostic",
            "base_tower": {
                "path": options.base_tower.as_posix(),
                "snapshot_id": base.snapshot_id,
                "sha256": sha256_file(options.base_tower),
            },
            "stateful_guard_source": {
                "path": options.stateful_tower.as_posix(),
                "snapshot_id": stateful.snapshot_id,
                "sha256": sha256_file(options.stateful_tower),
                "field": "high_cards[*].procedure[-1]",
            },
            "derived_tower": {
                "path": tower_path.as_posix(),
                "snapshot_id": snapshot.snapshot_id,
                "sha256": sha256_file(tower_path),
            },
            "changed_fields": ["high_cards[*].constraints"],
            "preserved_from_base": [
                "training_trajectory_ids",
                "low_skills",
                "mid_clusters",
                "high_paths",
                "mid_cards",
                "mid_index",
                "high_index",
                "high_card_name",
                "high_card_description",
                "high_card_procedure",
                "high_card_existing_constraints",
                "high_card_retrieval_condition",
            ],
            "completion_guard_count": len(high_cards),
        },
    )
    print(
        json.dumps(
            {
                "snapshot_id": snapshot.snapshot_id,
                "tower_sha256": sha256_file(tower_path),
                "completion_guard_count": len(high_cards),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tower", type=Path, required=True)
    parser.add_argument("--stateful-tower", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
