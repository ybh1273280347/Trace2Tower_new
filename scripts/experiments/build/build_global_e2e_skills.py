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
from trace2tower.methods.global_e2e import end_to_end_prompt
from trace2tower.methods.global_e2e.models import (
    GlobalE2ESkillCard,
    build_global_e2e_library,
)
from trace2tower.methods.global_e2e.renderer import (
    format_trajectory_corpus,
    global_e2e_card_text,
    induce_global_e2e_skills,
)
from trace2tower.results import MethodName
from trace2tower.semantic_index import SkillEmbeddingIndex
from trace2tower.trajectory import TrajectoryReader


async def main(options: argparse.Namespace) -> int:
    config = load_yaml(options.config)
    method = MethodName(config["method"])
    if method is not MethodName.GLOBAL_E2E_GPT:
        raise ValueError("corpus induction requires global_e2e_gpt")
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
        raise ValueError("Global E2E requires shared No-Skill training trajectories")
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
        raise ValueError("Global E2E requires at least one fully successful trajectory")

    prompt_path = Path(end_to_end_prompt.__file__)
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    corpus = format_trajectory_corpus(successful)
    corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / "corpus.json").write_text(corpus, encoding="utf-8")

    load_dotenv(options.env)
    if os.environ.get("RENDERER_MODEL") != "gpt-5.4":
        raise ValueError("Global E2E induction is frozen to RENDERER_MODEL=gpt-5.4")
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=1,
        timeout_seconds=options.timeout_seconds,
        retry_base_seconds=common["retry_base_seconds"],
    )
    checkpoint_path = options.output_dir / "induction.json"
    result_usage = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint["prompt_sha256"] != prompt_sha256
            or checkpoint["corpus_sha256"] != corpus_sha256
            or checkpoint["renderer_model"] != os.environ["RENDERER_MODEL"]
        ):
            raise ValueError("Global E2E checkpoint belongs to different inputs")
        cards = tuple(
            GlobalE2ESkillCard.from_record(item) for item in checkpoint["cards"]
        )
        result_usage = checkpoint["builder_chat_usage"]
    else:
        cards, rendered_corpus, result = await induce_global_e2e_skills(
            runtime,
            successful,
        )
        if rendered_corpus != corpus:
            raise ValueError("Global E2E renderer did not receive the frozen corpus")
        result_usage = asdict(result.usage)
        write_json(
            checkpoint_path,
            {
                "prompt_sha256": prompt_sha256,
                "corpus_sha256": corpus_sha256,
                "renderer_model": os.environ["RENDERER_MODEL"],
                "builder_chat_usage": result_usage,
                "cards": [card.to_record() for card in cards],
            },
        )

    ordered_cards = tuple(sorted(cards, key=lambda card: card.skill_id))
    texts = tuple(global_e2e_card_text(card) for card in ordered_cards)
    text_hashes = tuple(
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    )
    try:
        embedding = await runtime.embed(texts)
    finally:
        await runtime.close()
    index = SkillEmbeddingIndex(
        tuple(card.skill_id for card in ordered_cards),
        embedding.vectors,
        text_hashes,
    )
    library = build_global_e2e_library(
        options.benchmark,
        prompt_sha256,
        corpus_sha256,
        tuple(trajectory.trajectory_id for trajectory in successful),
        ordered_cards,
        index,
    )
    write_json(options.output_dir / "library.json", library.to_record())
    report = {
        "benchmark": options.benchmark.value,
        "source_trajectory_count": len(trajectories),
        "successful_trajectory_count": len(successful),
        "corpus_characters": len(corpus),
        "corpus_sha256": corpus_sha256,
        "prompt_sha256": prompt_sha256,
        "renderer_model": os.environ["RENDERER_MODEL"],
        "builder_chat_input_tokens": result_usage["input_tokens"],
        "builder_chat_output_tokens": result_usage["output_tokens"],
        "cached_chat_input_tokens": result_usage.get("cached_input_tokens"),
        "skill_count": len(cards),
        "library_id": library.library_id,
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
        default=Path("configs/experiments/webshop_global_e2e.yaml"),
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--timeout-seconds", type=float, default=600)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
