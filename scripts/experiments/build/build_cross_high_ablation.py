from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from trace2tower.methods.trace2tower.retrieval import SkillEmbeddingIndex
from trace2tower.methods.trace2tower.tower import TowerSnapshot, TowerSourceHashes, build_tower_snapshot


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as output:
        temporary = Path(output.name)
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main(options: argparse.Namespace) -> int:
    base = TowerSnapshot.from_record(json.loads(options.base.read_text(encoding="utf-8")))
    donor = TowerSnapshot.from_record(json.loads(options.donor.read_text(encoding="utf-8")))
    skill_id = options.skill_id
    donor_cards = {card.skill_id: card for card in donor.high_cards}
    if skill_id not in donor_cards or skill_id not in {card.skill_id for card in base.high_cards}:
        raise ValueError("selected High must exist in both snapshots")
    base_path = next(path for path in base.high_paths if path.path_id == skill_id)
    donor_path = next(path for path in donor.high_paths if path.path_id == skill_id)
    if base_path != donor_path:
        raise ValueError("cross-High ablation requires identical structural evidence")

    donor_index = donor.high_index.skill_ids.index(skill_id)
    base_index = base.high_index.skill_ids.index(skill_id)
    high_cards = tuple(
        donor_cards[skill_id] if card.skill_id == skill_id else card
        for card in base.high_cards
    )
    vectors = list(base.high_index.vectors)
    text_hashes = list(base.high_index.text_hashes)
    vectors[base_index] = donor.high_index.vectors[donor_index]
    text_hashes[base_index] = donor.high_index.text_hashes[donor_index]
    high_index = SkillEmbeddingIndex(
        base.high_index.skill_ids,
        tuple(vectors),
        tuple(text_hashes),
    )
    source_hashes = replace(
        base.source_hashes,
        rendered_cards=digest(
            {
                "mid_cards": [card.to_record() for card in base.mid_cards],
                "high_cards": [card.to_record() for card in high_cards],
            }
        ),
        retrieval_index=digest(
            {
                "mid_index": base.mid_index.to_record(),
                "high_index": high_index.to_record(),
            }
        ),
    )
    snapshot = build_tower_snapshot(
        version=base.version,
        benchmark=base.benchmark,
        config=base.config,
        training_trajectory_ids=base.training_trajectory_ids,
        source_hashes=TowerSourceHashes.from_record(asdict(source_hashes)),
        low_skills=base.low_skills,
        mid_clusters=base.mid_clusters,
        high_paths=base.high_paths,
        mid_cards=base.mid_cards,
        high_cards=high_cards,
        mid_index=base.mid_index,
        high_index=high_index,
    )
    snapshot.require_complete()
    write_json(options.output, snapshot.to_record())
    write_json(
        options.report,
        {
            "snapshot_id": snapshot.snapshot_id,
            "base_snapshot_id": base.snapshot_id,
            "donor_snapshot_id": donor.snapshot_id,
            "replaced_high_skill_id": skill_id,
            "structural_evidence_identical": True,
            "base_path": options.base.as_posix(),
            "donor_path": options.donor.as_posix(),
            "output_path": options.output.as_posix(),
            "source_hashes": asdict(source_hashes),
        },
    )
    print(snapshot.snapshot_id)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
