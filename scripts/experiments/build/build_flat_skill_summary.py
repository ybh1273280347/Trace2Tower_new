from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.methods.flat_skill_summary import prompt as prompt_module
from trace2tower.methods.flat_skill_summary.models import (
    FlatSkillCard,
    build_flat_library,
)
from trace2tower.methods.flat_skill_summary.renderer import (
    flat_card_text,
    render_flat_skill,
)
from trace2tower.results import MethodName
from trace2tower.semantic_index import SkillEmbeddingIndex
from trace2tower.trajectory import TrajectoryReader


def write_card_checkpoint(
    path: Path,
    prompt_sha256: str,
    cards: dict[str, FlatSkillCard],
    usage: list[dict],
) -> None:
    write_json(
        path,
        {
            "prompt_sha256": prompt_sha256,
            "cards": [cards[skill_id].to_record() for skill_id in sorted(cards)],
            "usage": usage,
        },
    )


async def main(options: argparse.Namespace) -> int:
    config = load_yaml(options.config)
    if config["method"] != MethodName.FLAT_SKILL_SUMMARY:
        raise ValueError("Flat builder requires the Flat Skill Summary config")
    paths = tuple(Path(path) for path in sorted(glob.glob(options.trajectory_glob)))
    if not paths:
        raise FileNotFoundError(f"no trajectories match: {options.trajectory_glob}")
    trajectories = tuple(
        trajectory
        for path in paths
        for trajectory in TrajectoryReader.read_jsonl(path)
        if trajectory.benchmark is options.benchmark
    )
    if any(
        trajectory.split is not ExperimentSplit.TRAIN
        or trajectory.method is not MethodName.NO_SKILL
        for trajectory in trajectories
    ):
        raise ValueError("Flat library requires shared No-Skill training trajectories")
    successful = tuple(
        sorted(
            (
                trajectory
                for trajectory in trajectories
                if trajectory.primary_score >= float(config["success_threshold"])
            ),
            key=lambda trajectory: trajectory.trajectory_id,
        )
    )
    if not successful:
        raise ValueError("Flat library requires at least one fully successful trajectory")
    prompt_path = Path(prompt_module.__file__)
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.output_dir / "cards.json"
    cards: dict[str, FlatSkillCard] = {}
    usage = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["prompt_sha256"] != prompt_sha256:
            raise ValueError("Flat prompt changed; use a new output directory")
        cards = {
            card.source_trajectory_id: card
            for card in (
                FlatSkillCard.from_record(item) for item in checkpoint["cards"]
            )
        }
        usage = list(checkpoint.get("usage", ()))
    successful_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in successful
    }
    if set(cards) - set(successful_by_id):
        raise ValueError("Flat checkpoint references a non-selected trajectory")
    reused_card_count = len(cards)
    missing = tuple(
        trajectory
        for trajectory in successful
        if trajectory.trajectory_id not in cards
    )

    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    try:
        for trajectory in missing:
            card, result = await render_flat_skill(runtime, trajectory)
            cards[trajectory.trajectory_id] = card
            usage.append(
                {
                    "skill_id": card.skill_id,
                    **asdict(result.usage),
                    "latency_ms": result.latency_ms,
                }
            )
            write_card_checkpoint(checkpoint_path, prompt_sha256, cards, usage)

        ordered_cards = tuple(sorted(cards.values(), key=lambda card: card.skill_id))
        texts = tuple(flat_card_text(card) for card in ordered_cards)
        text_hashes = tuple(
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        )
        hash_by_id = dict(
            zip((card.skill_id for card in ordered_cards), text_hashes, strict=True)
        )
        reusable_vectors = {}
        library_path = options.output_dir / "library.json"
        if library_path.exists():
            existing_library = json.loads(library_path.read_text(encoding="utf-8"))
            existing_index = SkillEmbeddingIndex.from_record(existing_library["index"])
            if existing_index.text_hashes:
                reusable_vectors = {
                    skill_id: vector
                    for skill_id, vector, text_hash in zip(
                        existing_index.skill_ids,
                        existing_index.vectors,
                        existing_index.text_hashes,
                        strict=True,
                    )
                    if hash_by_id.get(skill_id) == text_hash
                }
        missing_embedding_ids = tuple(
            card.skill_id
            for card in ordered_cards
            if card.skill_id not in reusable_vectors
        )
        text_by_id = dict(zip((card.skill_id for card in ordered_cards), texts, strict=True))
        embedding_results = []
        batch_size = int(common["embedding_batch_size"])
        for offset in range(0, len(missing_embedding_ids), batch_size):
            batch_ids = missing_embedding_ids[offset : offset + batch_size]
            embedding_results.append(
                await runtime.embed([text_by_id[skill_id] for skill_id in batch_ids])
            )
    finally:
        await runtime.close()
    new_vectors = dict(
        zip(
            missing_embedding_ids,
            (
                vector
                for result in embedding_results
                for vector in result.vectors
            ),
            strict=True,
        )
    )
    vectors = reusable_vectors | new_vectors
    index = SkillEmbeddingIndex(
        tuple(card.skill_id for card in ordered_cards),
        tuple(vectors[card.skill_id] for card in ordered_cards),
        text_hashes,
    )
    library = build_flat_library(
        options.benchmark,
        prompt_sha256,
        ordered_cards,
        index,
    )
    write_json(options.output_dir / "library.json", library.to_record())
    report = {
        "benchmark": options.benchmark.value,
        "trajectory_count": len(trajectories),
        "successful_trajectory_count": len(successful),
        "reused_card_count": reused_card_count,
        "new_card_count": len(missing),
        "card_count": len(cards),
        "library_id": library.library_id,
        "prompt_sha256": prompt_sha256,
        "renderer_model": os.environ["RENDERER_MODEL"],
        "reused_embedding_count": len(reusable_vectors),
        "new_embedding_count": len(missing_embedding_ids),
        "embedding_input_tokens": (
            sum(result.usage.input_tokens or 0 for result in embedding_results)
            if embedding_results
            else None
        ),
    }
    write_json(options.output_dir / "report.json", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--trajectory-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/flat_skill_summary.yaml"),
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
