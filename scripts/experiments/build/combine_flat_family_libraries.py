from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from scripts.experiments.data.prepare_alfworld_protocol import FAMILIES
from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.manifests import Benchmark
from trace2tower.methods.flat_skill_summary.models import (
    FlatSkillLibrary,
    build_flat_library,
)
from trace2tower.semantic_index import SkillEmbeddingIndex


async def main(options: argparse.Namespace) -> int:
    libraries = {
        family: FlatSkillLibrary.from_record(
            json.loads(
                (options.family_root / family / "library.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        for family in FAMILIES
    }
    prompt_hashes = {library.prompt_sha256 for library in libraries.values()}
    if len(prompt_hashes) != 1 or any(
        library.benchmark is not Benchmark.ALFWORLD for library in libraries.values()
    ):
        raise ValueError("Flat family libraries must share one ALFWorld prompt contract")

    cards = []
    vectors = {}
    text_hashes = {}
    for family in FAMILIES:
        library = libraries[family]
        source_vectors = dict(
            zip(library.index.skill_ids, library.index.vectors, strict=True)
        )
        source_hashes = dict(
            zip(library.index.skill_ids, library.index.text_hashes, strict=True)
        )
        for card in library.cards:
            skill_id = f"flat_{family}_{card.skill_id.removeprefix('flat_')}"
            cards.append(
                replace(card, skill_id=skill_id)
            )
            vectors[skill_id] = source_vectors[card.skill_id]
            text_hashes[skill_id] = source_hashes[card.skill_id]
    cards = tuple(sorted(cards, key=lambda card: card.skill_id))
    ids = tuple(card.skill_id for card in cards)
    index = SkillEmbeddingIndex(
        ids,
        tuple(vectors[skill_id] for skill_id in ids),
        tuple(text_hashes[skill_id] for skill_id in ids),
    )
    library = build_flat_library(
        Benchmark.ALFWORLD,
        next(iter(prompt_hashes)),
        cards,
        index,
    )
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(options.output_dir / "library.json", library.to_record())
    report = {
        "library_id": library.library_id,
        "card_count": len(cards),
        "family_card_counts": {
            family: len(libraries[family].cards) for family in FAMILIES
        },
        "reused_embedding_count": len(cards),
        "new_embedding_count": 0,
        "embedding_input_tokens": 0,
    }
    write_json(options.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
