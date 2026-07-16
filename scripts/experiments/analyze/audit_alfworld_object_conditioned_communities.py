from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    AlfworldTaskPrototype,
    extract_task_prototype,
)


def _key_text(key: tuple[Any, ...]) -> str:
    return " | ".join(str(part) for part in key)


def main(options: argparse.Namespace) -> int:
    successful: list[tuple[str, AlfworldTaskPrototype]] = []
    failed: list[tuple[str, AlfworldTaskPrototype]] = []
    with options.input.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prototype = extract_task_prototype(record)
            item = (str(record.get("trajectory_id", "")), prototype)
            if float(record.get("primary_score", 0.0)) >= options.success_threshold:
                successful.append(item)
            else:
                failed.append(item)

    exact = Counter(prototype.exact_key for _, prototype in successful)
    relaxed = Counter(prototype.relaxed_key for _, prototype in successful)
    object_communities = Counter(
        (prototype.transformation, prototype.target_object)
        for _, prototype in successful
    )
    transform = Counter(prototype.transformation for _, prototype in successful)
    minimum_support = max(
        1, int((len(successful) + len(failed)) * options.min_support_ratio)
    )

    def top(counter: Counter, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {"key": _key_text(key), "support": count, "ratio": count / len(successful)}
            for key, count in counter.most_common(limit)
        ]

    prototype_by_key: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for trajectory_id, prototype in successful:
        prototype_by_key[prototype.relaxed_key].append(trajectory_id)
    supported_relaxed = {
        _key_text(key): ids
        for key, ids in prototype_by_key.items()
        if len(ids) >= minimum_support
    }
    community_conditions: dict[str, dict[str, Any]] = {}
    for key, support in object_communities.items():
        if support < minimum_support:
            continue
        members = [
            prototype
            for _, prototype in successful
            if (prototype.transformation, prototype.target_object) == key
        ]
        sources = Counter(source for item in members for source in item.source_receptacles)
        devices = Counter(device for item in members for device in item.transformation_devices)
        destinations = Counter(
            destination for item in members for destination in item.destination_receptacles
        )
        community_conditions[_key_text(key)] = {
            "support": support,
            "ratio": support / len(successful),
            "sources": sources,
            "devices": devices,
            "destinations": destinations,
            "canonical_goal_examples": list(
                dict.fromkeys(item.canonical_goal for item in members)
            )[:8],
        }

    output = {
        "input": str(options.input),
        "trajectory_count": len(successful) + len(failed),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "success_threshold": options.success_threshold,
        "min_support_ratio": options.min_support_ratio,
        "minimum_support": minimum_support,
        "successful_by_transformation": transform,
        "top_exact_object_conditioned_prototypes": top(exact),
        "top_relaxed_object_conditioned_prototypes": top(relaxed),
        "top_object_communities": top(object_communities),
        "supported_relaxed_prototypes": supported_relaxed,
        "supported_object_communities": community_conditions,
        "distinct_exact_prototypes": len(exact),
        "distinct_relaxed_prototypes": len(relaxed),
        "failure_examples": [
            {"trajectory_id": trajectory_id, **prototype.to_record()}
            for trajectory_id, prototype in failed[: options.failure_examples]
        ],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "trajectory_count": output["trajectory_count"],
        "successful_count": output["successful_count"],
        "minimum_support": minimum_support,
        "distinct_exact_prototypes": len(exact),
        "distinct_relaxed_prototypes": len(relaxed),
        "supported_relaxed_prototypes": len(supported_relaxed),
        "supported_object_communities": len(community_conditions),
        "top_relaxed": output["top_relaxed_object_conditioned_prototypes"][:10],
        "top_object_communities": output["top_object_communities"][:15],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=0.999)
    parser.add_argument("--min-support-ratio", type=float, default=0.02)
    parser.add_argument("--failure-examples", type=int, default=20)
    raise SystemExit(main(parser.parse_args()))
