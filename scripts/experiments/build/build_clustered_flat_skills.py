from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from trace2tower.benchmarks.models import ClickableKind
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark, ExperimentSplit
from trace2tower.methods.flat_skill_summary import clustered_prompt
from trace2tower.methods.flat_skill_summary.clustered_renderer import (
    render_clustered_flat_skill,
)
from trace2tower.methods.flat_skill_summary.corpus_renderer import (
    corpus_flat_card_text,
    format_trajectory_corpus,
)
from trace2tower.methods.flat_skill_summary.models import (
    CorpusFlatSkillCard,
    build_corpus_flat_library,
)
from trace2tower.results import MethodName
from trace2tower.semantic_index import SkillEmbeddingIndex
from trace2tower.trajectory import EpisodeTrajectory, StepRecord, TrajectoryReader

RANDOM_STATE = 20260715


def step_event(step: StepRecord) -> str:
    if not step.valid_action:
        return "INVALID"
    if step.action_name == "search_action":
        return "SEARCH"
    if step.action_name != "click_action" or not step.action_arguments:
        return step.action_name.upper() if step.action_name else "NO_ACTION"
    value = str(step.action_arguments.get("value", ""))
    if value == "Buy Now":
        return "BUY"
    if value == "Back to Search":
        return "REJECT_TO_SEARCH"
    if value == "next >":
        return "NEXT_RESULTS"
    if value == "< prev":
        return "RETURN_PREVIOUS"
    if value in {"Description", "Features", "Attributes", "Reviews"}:
        return f"INSPECT_{value.upper()}"
    kind = step.clickable_types.get(value)
    if kind is ClickableKind.PRODUCT_LINK:
        return "OPEN_PRODUCT"
    if kind is ClickableKind.OPTION:
        return "SELECT_OPTION"
    return "CLICK_CONTROL"


def task_profile(
    sample_id: str,
    trajectories: tuple[EpisodeTrajectory, ...],
) -> dict:
    goals = {trajectory.task_goal for trajectory in trajectories}
    if len(goals) != 1:
        raise ValueError(f"task repeats disagree on goal: {sample_id}")
    patterns = tuple(
        sorted(
            {
                " > ".join(step_event(step) for step in trajectory.steps)
                for trajectory in trajectories
            }
        )
    )
    return {
        "sample_id": sample_id,
        "task_goal": next(iter(goals)),
        "successful_patterns": patterns,
        "successful_trajectory_ids": tuple(
            sorted(trajectory.trajectory_id for trajectory in trajectories)
        ),
    }


def profile_text(profile: dict) -> str:
    patterns = "\n".join(f"- {pattern}" for pattern in profile["successful_patterns"])
    return (
        "Successful execution patterns:\n"
        f"{patterns}\n\n"
        "Goal-observable requirements:\n"
        f"{profile['task_goal']}"
    )


def choose_clustering(vectors: np.ndarray, task_ids: tuple[str, ...]) -> dict:
    candidates = []
    labels_by_k = {}
    for cluster_count in range(2, min(6, len(task_ids) - 1) + 1):
        labels = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=300,
        ).fit_predict(vectors)
        sizes = tuple(int(np.sum(labels == label)) for label in range(cluster_count))
        score = float(silhouette_score(vectors, labels, metric="cosine"))
        candidates.append(
            {
                "cluster_count": cluster_count,
                "silhouette_cosine": score,
                "cluster_sizes": sizes,
                "eligible": min(sizes) >= 3,
            }
        )
        labels_by_k[cluster_count] = labels
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("no clustering candidate has at least three tasks per cluster")
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["silhouette_cosine"],
            -candidate["cluster_count"],
        ),
    )
    raw_labels = labels_by_k[selected["cluster_count"]]
    members = {
        label: tuple(
            task_ids[index]
            for index, current in enumerate(raw_labels)
            if current == label
        )
        for label in range(selected["cluster_count"])
    }
    ordered_labels = sorted(members, key=lambda label: min(members[label]))
    canonical = {label: index for index, label in enumerate(ordered_labels)}
    labels = tuple(canonical[int(label)] for label in raw_labels)
    return {
        "random_state": RANDOM_STATE,
        "selection_rule": "maximum cosine silhouette with at least 3 tasks per cluster",
        "candidates": candidates,
        "selected_cluster_count": selected["cluster_count"],
        "labels": labels,
    }


async def main(options: argparse.Namespace) -> int:
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
        raise ValueError("clustered Flat requires shared No-Skill training trajectories")
    successful = tuple(
        sorted(
            (trajectory for trajectory in trajectories if trajectory.primary_score >= 0.999),
            key=lambda trajectory: trajectory.trajectory_id,
        )
    )
    grouped = defaultdict(list)
    for trajectory in successful:
        grouped[trajectory.sample_id].append(trajectory)
    profiles = tuple(
        task_profile(sample_id, tuple(grouped[sample_id]))
        for sample_id in sorted(grouped)
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(options.env)
    if os.environ.get("RENDERER_MODEL") != "gpt-5.4":
        raise ValueError("clustered Flat rendering is frozen to RENDERER_MODEL=gpt-5.4")
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=4,
        max_attempts=1,
        timeout_seconds=options.timeout_seconds,
        retry_base_seconds=common["retry_base_seconds"],
    )
    profile_embedding = await runtime.embed([profile_text(profile) for profile in profiles])
    vectors = normalize(np.asarray(profile_embedding.vectors, dtype=np.float64))
    clustering = choose_clustering(
        vectors,
        tuple(profile["sample_id"] for profile in profiles),
    )
    profile_by_id = {profile["sample_id"]: profile for profile in profiles}
    clusters = []
    for cluster_index in range(clustering["selected_cluster_count"]):
        task_ids = tuple(
            profile["sample_id"]
            for profile, label in zip(profiles, clustering["labels"], strict=True)
            if label == cluster_index
        )
        cluster_trajectories = tuple(
            trajectory
            for task_id in task_ids
            for trajectory in grouped[task_id]
        )
        clusters.append(
            {
                "cluster_id": f"task_cluster_{cluster_index:02d}",
                "task_ids": task_ids,
                "trajectory_ids": tuple(
                    trajectory.trajectory_id for trajectory in cluster_trajectories
                ),
            }
        )
    cluster_record = {
        **clustering,
        "task_count": len(profiles),
        "successful_trajectory_count": len(successful),
        "profiles": profiles,
        "clusters": clusters,
    }
    write_json(options.output_dir / "clusters.json", cluster_record)

    checkpoint_path = options.output_dir / "cards.json"
    existing_cards = {}
    usage = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        existing_cards = {
            item["cluster_id"]: CorpusFlatSkillCard.from_record(item["card"])
            for item in checkpoint["cards"]
        }
        usage = list(checkpoint["usage"])

    async def render(cluster: dict):
        cluster_id = cluster["cluster_id"]
        if cluster_id in existing_cards:
            return cluster_id, existing_cards[cluster_id], None
        task_ids = cluster["task_ids"]
        cluster_trajectories = tuple(
            trajectory for task_id in task_ids for trajectory in grouped[task_id]
        )
        card, result = await render_clustered_flat_skill(
            runtime,
            cluster_id,
            tuple(profile_by_id[task_id] for task_id in task_ids),
            cluster_trajectories,
        )
        return cluster_id, card, result

    rendered = await asyncio.gather(*(render(cluster) for cluster in clusters))
    cards_by_cluster = dict(existing_cards)
    for cluster_id, card, result in rendered:
        cards_by_cluster[cluster_id] = card
        if result is not None:
            usage.append({"cluster_id": cluster_id, **asdict(result.usage)})
    write_json(
        checkpoint_path,
        {
            "cards": [
                {"cluster_id": cluster_id, "card": cards_by_cluster[cluster_id].to_record()}
                for cluster_id in sorted(cards_by_cluster)
            ],
            "usage": usage,
        },
    )

    ordered_cards = tuple(
        cards_by_cluster[cluster["cluster_id"]] for cluster in clusters
    )
    texts = tuple(corpus_flat_card_text(card) for card in ordered_cards)
    card_embedding = await runtime.embed(texts)
    await runtime.close()
    text_hashes = tuple(
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    )
    index = SkillEmbeddingIndex(
        tuple(card.skill_id for card in ordered_cards),
        card_embedding.vectors,
        text_hashes,
    )
    prompt_sha256 = hashlib.sha256(
        Path(clustered_prompt.__file__).read_bytes()
    ).hexdigest()
    corpus = format_trajectory_corpus(successful)
    corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    library = build_corpus_flat_library(
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
        "successful_task_count": len(profiles),
        "selected_cluster_count": clustering["selected_cluster_count"],
        "cluster_sizes": [len(cluster["task_ids"]) for cluster in clusters],
        "renderer_model": os.environ["RENDERER_MODEL"],
        "builder_chat_input_tokens": sum(item["input_tokens"] or 0 for item in usage),
        "builder_chat_output_tokens": sum(item["output_tokens"] or 0 for item in usage),
        "skill_count": len(ordered_cards),
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
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--timeout-seconds", type=float, default=600)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
