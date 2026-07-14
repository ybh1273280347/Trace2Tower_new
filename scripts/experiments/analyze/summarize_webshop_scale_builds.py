from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from scripts.experiments.run.rollout_no_skill_train import write_json


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def usage_totals(records: list[dict]) -> tuple[int, int]:
    return (
        sum(int(item.get("input_tokens") or 0) for item in records),
        sum(int(item.get("output_tokens") or 0) for item in records),
    )


def flat_summary(pool: str) -> dict:
    root = Path("artifacts/flat_skill_summary/webshop-scale-v1") / pool
    report = load(root / "report.json")
    cards = load(root / "cards.json")
    input_tokens, output_tokens = usage_totals(cards["usage"])
    return {
        "builder_chat_input_tokens": input_tokens,
        "builder_chat_output_tokens": output_tokens,
        "embedding_input_tokens": int(report["embedding_input_tokens"] or 0),
        "final_skill_count": int(report["card_count"]),
        "skill_counts": {"flat": int(report["card_count"])},
    }


def skillx_summary(pool: str) -> dict:
    root = Path("artifacts/skillx/webshop-scale-v1") / pool
    upstream = load(root / "upstream-parallel4-recoverable/report.json")
    execution = load(root / "execution/report.json")
    usage = upstream["llm_usage"]
    plan_count = int(execution["plan_count"])
    executable_count = int(execution["skill_count"])
    return {
        "builder_chat_input_tokens": int(usage["input_tokens"]),
        "builder_chat_output_tokens": int(usage["output_tokens"]),
        "embedding_input_tokens": int(upstream["embedding_input_tokens"])
        + int(execution["embedding_input_tokens"]),
        "final_skill_count": plan_count + executable_count,
        "skill_counts": {
            "planning": plan_count,
            "executable": executable_count,
        },
    }


def tower_summary(pool: str, policy: str) -> dict:
    root = Path("artifacts/trace2tower/scale-v1") / pool / policy
    skills = load(root / "skills/rendered-cards.json")
    graph = load(root / "graph/report.json")
    preprocessing = load(root / "preprocessed.metadata.json")
    retrieval = load(root / "skills/retrieval-index.json")["report"]
    snapshot = load(
        Path("artifacts/trace2tower/towers")
        / f"webshop-scale-v1-{pool}-{policy}.json"
    )
    high_paths = load(root / "skills/high-paths.json")["paths"]
    input_tokens, output_tokens = usage_totals(skills["usage"])
    embedding_usage = (
        retrieval.get("mid_embedding_usage"),
        retrieval.get("high_embedding_usage"),
    )
    embedding_tokens = int(preprocessing["embedding_input_tokens"]) + sum(
        int(item.get("input_tokens") or 0) for item in embedding_usage if item
    )
    cluster_sizes = [int(value) for value in graph["cluster_sizes"].values()]
    supporting_task_counts = [
        len(
            {
                trajectory_id.rsplit(":", 1)[0]
                for trajectory_id in path["supporting_trajectory_ids"]
            }
        )
        for path in high_paths
    ]
    low_count = len(snapshot["low_skills"])
    mid_count = len(snapshot["mid_cards"])
    high_count = len(snapshot["high_cards"])
    return {
        "builder_chat_input_tokens": input_tokens,
        "builder_chat_output_tokens": output_tokens,
        "embedding_input_tokens": embedding_tokens,
        "final_skill_count": low_count + mid_count + high_count,
        "skill_counts": {
            "low": low_count,
            "mid": mid_count,
            "high": high_count,
        },
        "graph": {
            "cluster_count": int(graph["cluster_count"]),
            "cluster_size_min": min(cluster_sizes),
            "cluster_size_max": max(cluster_sizes),
            "cluster_size_mean": fmean(cluster_sizes),
            "edge_count": int(graph["edge_count"]),
        },
        "high_support": {
            "path_count": len(high_paths),
            "mean_positive_support": (
                fmean(float(path["positive_support"]) for path in high_paths)
                if high_paths
                else None
            ),
            "mean_supporting_task_count": (
                fmean(supporting_task_counts) if supporting_task_counts else None
            ),
        },
    }


def main(options: argparse.Namespace) -> int:
    pools = options.pool or ["p50", "p100", "p200"]
    output = {
        "protocol_id": "webshop-scale-v1",
        "pools": {
            pool: {
                "flat": flat_summary(pool),
                "skillx": skillx_summary(pool),
                "tower_success": tower_summary(pool, "success"),
                "tower_mixed": tower_summary(pool, "mixed"),
            }
            for pool in pools
        },
    }
    write_json(options.output, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", action="append", choices=("p50", "p100", "p200"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/experiments/webshop-scale-v1/build-summary.json"
        ),
    )
    raise SystemExit(main(parser.parse_args()))
