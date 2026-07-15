from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.graph_retrieval import (
    TowerGraphProfile,
    retrieve_tower_graph,
)
from trace2tower.methods.trace2tower.provider import Trace2TowerSkillProvider
from trace2tower.methods.trace2tower.retrieval import retrieve_tower
from trace2tower.methods.trace2tower.webshop_events import (
    infer_webshop_page_type,
    webshop_applicable_events,
)


DEFAULT_RUNS = (
    (
        "v0-cap8",
        Path("artifacts/runs/webshop-original-concept-v1-test-flash-p100-full-cap8-r1"),
    ),
    (
        "pareto-cap8",
        Path("artifacts/runs/webshop-original-concept-v1-test-a-flash-pareto-v1-cap8-r1"),
    ),
    (
        "pareto-cap3",
        Path("artifacts/runs/webshop-original-concept-v1-test-a-flash-pareto-v1-cap3-r1"),
    ),
)
DEFAULT_SAMPLE_IDS = ("webshop:170", "webshop:873", "webshop:969")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "experiments/webshop/original-concept-v1/refinement/"
            "retrieval-diagnostic.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "experiments/webshop/original-concept-v1/refinement/"
            "RETRIEVAL_DIAGNOSTIC.md"
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser.parse_args()


def read_result(run_dir: Path, sample_id: str) -> dict:
    matches = []
    for result_path in run_dir.rglob("results.jsonl"):
        for line in result_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["sample_id"] == sample_id:
                matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"expected one result for {sample_id} in {run_dir}")
    return matches[0]


def read_trajectory(run_dir: Path, sample_id: str) -> dict:
    matches = []
    for trajectory_path in run_dir.rglob("trajectories/*.json"):
        record = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if record["sample_id"] == sample_id:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"expected one trajectory for {sample_id} in {run_dir}")
    return matches[0]


def load_provider(
    runtime: CommonLLMRuntime, run_dir: Path
) -> Trace2TowerSkillProvider:
    resolved = yaml.safe_load(
        (run_dir / "resolved-config.yaml").read_text(encoding="utf-8")
    )
    method = resolved["method"]
    diverse = method.get("retrieval_strategy", "legacy") == "diverse"
    lifecycle_path = method.get("lifecycle_report")
    return Trace2TowerSkillProvider.from_path(
        runtime,
        Path(resolved["artifacts"]["webshop"]["path"]),
        lifecycle_report_path=Path(lifecycle_path) if lifecycle_path else None,
        include_high=bool(method["include_high"]),
        direct_mid_top_k=int(method["direct_mid_top_k"]),
        high_similarity_threshold=float(method["high_similarity_threshold"]),
        include_high_child_context=bool(method["include_high_child_context"]),
        direct_mid_candidate_top_k=(
            int(method["direct_mid_candidate_top_k"]) if diverse else None
        ),
        direct_mid_similarity_threshold=float(
            method.get("direct_mid_similarity_threshold", 0.45)
        ),
        direct_mid_relative_margin=float(
            method.get("direct_mid_relative_margin", 0.08)
        ),
        direct_mid_dedup_similarity_threshold=float(
            method.get("direct_mid_dedup_similarity_threshold", 0.95)
        ),
        direct_mid_mmr_lambda=float(method.get("direct_mid_mmr_lambda", 0.75)),
        status_tie_epsilon=float(method.get("status_tie_epsilon", 0.0)),
    )


async def embed_batches(
    runtime: CommonLLMRuntime, texts: list[str], batch_size: int
) -> list[tuple[float, ...]]:
    vectors = []
    for start in range(0, len(texts), batch_size):
        result = await runtime.embed(texts[start : start + batch_size])
        vectors.extend(result.vectors)
    return vectors


async def diagnose(options: argparse.Namespace) -> list[dict]:
    load_dotenv(options.env)
    runtime = CommonLLMRuntime(
        max_concurrency=4,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=1,
    )
    sample_ids = tuple(options.sample_id) or DEFAULT_SAMPLE_IDS
    records = []
    try:
        for label, run_dir in DEFAULT_RUNS:
            provider = load_provider(runtime, run_dir)
            profile_path = (
                Path(
                    yaml.safe_load(
                        (run_dir / "resolved-config.yaml").read_text(encoding="utf-8")
                    )["artifacts"]["webshop"]["path"]
                ).parent
                / "graph-retrieval-profile.json"
            )
            graph_profile = (
                TowerGraphProfile.from_record(
                    json.loads(profile_path.read_text(encoding="utf-8"))
                )
                if profile_path.exists()
                else None
            )
            high_cards = provider.high_cards
            mid_cards = provider.mid_cards
            for sample_id in sample_ids:
                result = read_result(run_dir, sample_id)
                trajectory = read_trajectory(run_dir, sample_id)
                texts = []
                for step in trajectory["steps"]:
                    texts.extend(
                        (
                            trajectory["task_goal"],
                            f"{trajectory['task_goal']}\n{step['observation']}",
                            step["observation"],
                        )
                    )
                vectors = await embed_batches(
                    runtime, texts, options.embedding_batch_size
                )
                steps = []
                for index, step in enumerate(trajectory["steps"]):
                    retrieval = retrieve_tower(
                        vectors[index * 3],
                        vectors[index * 3 + 1],
                        provider.snapshot.high_index,
                        provider.snapshot.mid_index,
                        high_cards,
                        mid_cards,
                        high_top_k=1 if provider.include_high else 0,
                        direct_mid_top_k=provider.direct_mid_top_k,
                        high_similarity_threshold=provider.high_similarity_threshold,
                        include_high_child_context=provider.include_high_child_context,
                        direct_mid_candidate_top_k=provider.direct_mid_candidate_top_k,
                        direct_mid_similarity_threshold=provider.direct_mid_similarity_threshold,
                        direct_mid_relative_margin=provider.direct_mid_relative_margin,
                        direct_mid_dedup_similarity_threshold=(
                            provider.direct_mid_dedup_similarity_threshold
                        ),
                        direct_mid_mmr_lambda=provider.direct_mid_mmr_lambda,
                        downweighted_skill_ids=provider.downweighted_skill_ids,
                        status_tie_epsilon=provider.status_tie_epsilon,
                    )
                    state_only_matches = provider.snapshot.mid_index.search(
                        vectors[index * 3 + 2], 8
                    )
                    graph_result = (
                        retrieve_tower_graph(
                            vectors[index * 3],
                            vectors[index * 3 + 2],
                            provider.snapshot.high_index,
                            provider.snapshot.mid_index,
                            high_cards,
                            mid_cards,
                            {
                                path.path_id: path
                                for path in provider.snapshot.high_paths
                            },
                            graph_profile,
                            webshop_applicable_events(
                                infer_webshop_page_type(step["observation"])
                            ),
                            mid_context_budget=3,
                            downweighted_skill_ids=provider.downweighted_skill_ids,
                            status_tie_epsilon=provider.status_tie_epsilon,
                        )
                        if graph_profile
                        else None
                    )
                    high_card = retrieval.high_card
                    steps.append(
                        {
                            "step_index": step["step_index"],
                            "observation": step["observation"].splitlines()[0][:120],
                            "action": step["action_name"],
                            "high": (
                                {
                                    "skill_id": high_card.skill_id,
                                    "name": high_card.name,
                                    "score": retrieval.high_match.cosine_similarity,
                                    "children": list(high_card.ordered_mid_ids),
                                }
                                if high_card
                                else None
                            ),
                            "direct_mid": [
                                {
                                    "skill_id": match.skill_id,
                                    "name": mid_cards[match.skill_id].name,
                                    "score": match.cosine_similarity,
                                }
                                for match in retrieval.direct_mid_matches
                            ],
                            "state_only_mid": [
                                {
                                    "skill_id": match.skill_id,
                                    "name": mid_cards[match.skill_id].name,
                                    "score": match.cosine_similarity,
                                }
                                for match in state_only_matches
                            ],
                            "graph_preview": (
                                {
                                    "high": {
                                        "skill_id": graph_result.graph_high_match.skill_id,
                                        "name": graph_result.retrieval.high_card.name,
                                        "graph_score": graph_result.graph_high_match.score,
                                        "goal_score": (
                                            graph_result.graph_high_match.goal_similarity
                                        ),
                                        "state_score": (
                                            graph_result.graph_high_match.state_similarity
                                        ),
                                        "event_compatibility": (
                                            graph_result.graph_high_match.event_compatibility
                                        ),
                                        "path_quality": (
                                            graph_result.graph_high_match.path_quality
                                        ),
                                    },
                                    "mid_ids": list(
                                        graph_result.retrieval.context_skill_ids[1:]
                                    ),
                                    "mid_names": [
                                        card.name for card in graph_result.retrieval.mid_cards
                                    ],
                                }
                                if graph_result and graph_result.graph_high_match
                                else None
                            ),
                            "context_skill_ids": list(retrieval.context_skill_ids),
                            "context_skill_count": len(retrieval.context_skill_ids),
                        }
                    )
                records.append(
                    {
                        "run": label,
                        "sample_id": sample_id,
                        "task_goal": trajectory["task_goal"],
                        "primary_score": result["primary_score"],
                        "trajectory_skill_ids": result["skill_ids"],
                        "steps": steps,
                    }
                )
    finally:
        await runtime.close()
    return records


def render_markdown(records: list[dict]) -> str:
    lines = [
        "# Test-A Tower Retrieval Diagnostic",
        "",
        "Each section replays the frozen retriever on the observations from that run. "
        "The trajectory-level skill list is a union; the table below is the actual "
        "per-step selection reconstructed from the frozen index.",
        "",
    ]
    for record in records:
        lines.extend(
            (
                f"## {record['run']} / {record['sample_id']}",
                "",
                f"Goal: {record['task_goal']}",
                "",
                f"Reward: {record['primary_score']:.4f}",
                "",
                "| Step | State | Legacy High | Goal+state Mid | State-only Mid | Graph preview (High; active/next/fill) | Context count |",
                "|---:|---|---|---|---|---|---:|",
            )
        )
        for step in record["steps"]:
            high = step["high"]
            high_text = (
                f"{high['name']} ({high['score']:.3f}); "
                f"children={','.join(high['children'])}"
                if high
                else "none"
            )
            mid_text = "<br>".join(
                f"{item['name']} ({item['score']:.3f})"
                for item in step["direct_mid"]
            )
            state_mid_text = "<br>".join(
                f"{item['name']} ({item['score']:.3f})"
                for item in step["state_only_mid"][:4]
            )
            graph = step["graph_preview"]
            graph_text = (
                f"{graph['high']['name']} ({graph['high']['graph_score']:.3f}); "
                + " -> ".join(graph["mid_names"])
                if graph
                else "profile unavailable"
            )
            state = step["observation"].replace("|", "\\|")
            lines.append(
                f"| {step['step_index']} | {state} | {high_text} | "
                f"{mid_text} | {state_mid_text} | {graph_text} | "
                f"{step['context_skill_count']} |"
            )
        lines.append("")

    lines.extend(("## Aggregate", ""))
    lines.append("| Run | Steps | Mean context skills | Dominant High | High share |")
    lines.append("|---|---:|---:|---|---:|")
    for run in dict.fromkeys(record["run"] for record in records):
        steps = [
            step
            for record in records
            if record["run"] == run
            for step in record["steps"]
        ]
        high_counts = Counter(
            step["high"]["skill_id"] for step in steps if step["high"]
        )
        dominant_high, count = high_counts.most_common(1)[0]
        mean_context = sum(step["context_skill_count"] for step in steps) / len(steps)
        lines.append(
            f"| {run} | {len(steps)} | {mean_context:.2f} | {dominant_high} | "
            f"{count / len(steps):.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    options = parse_args()
    records = await diagnose(options)
    options.output_json.parent.mkdir(parents=True, exist_ok=True)
    options.output_json.write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    options.output_md.write_text(render_markdown(records), encoding="utf-8")
    print(f"wrote {options.output_json}")
    print(f"wrote {options.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
