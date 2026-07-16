from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.methods.trace2tower.alfworld_task_prototypes import (
    canonical_goal,
    goal_destination,
    goal_target_object,
    goal_transformation,
)
from trace2tower.semantic_index import SkillEmbeddingIndex


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    labels = json.loads(options.labels.read_text(encoding="utf-8"))["tasks"]
    label_by_goal = {item["task_goal"]: item for item in labels}
    trajectories = {}
    for path in options.run_dir.rglob("*.json"):
        if "trajectories" not in path.parts:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("task_goal") in label_by_goal:
            trajectories[record["task_goal"]] = record

    structure = json.loads(options.structure.read_text(encoding="utf-8"))
    goal_by_skill = {
        item["community_id"]: item["canonical_goal"]
        for item in structure["discovery"]["communities"]
    }
    prototype_by_skill = {
        item["community_id"]: item["prototype"]
        for item in structure["discovery"]["communities"]
    }
    index_payload = json.loads(options.index.read_text(encoding="utf-8"))
    index = SkillEmbeddingIndex.from_record(index_payload["high_index"])
    queries = []
    pairs = []
    for label in labels:
        external = label["task_goal"]
        record = trajectories[external]
        canonical = canonical_goal(record["steps"][0]["observation"])
        pairs.append((label, record, external, canonical))
        queries.extend((external, canonical))

    runtime = CommonLLMRuntime(
        max_concurrency=1,
        max_attempts=3,
        timeout_seconds=120,
        retry_base_seconds=2,
    )
    try:
        result = await runtime.embed(queries)
    finally:
        await runtime.close()

    rows = []
    external_matches = 0
    canonical_matches = 0
    structured_matches = 0
    for offset, (label, record, external, canonical) in enumerate(pairs):
        external_match = index.search(result.vectors[offset * 2], 1)[0]
        canonical_match = index.search(result.vectors[offset * 2 + 1], 1)[0]
        query_target = goal_target_object(canonical)
        query_transformation = goal_transformation(canonical)
        query_destination = goal_destination(canonical)
        ranked = index.search(result.vectors[offset * 2 + 1], len(index.skill_ids))
        structured_match = max(
            ranked,
            key=lambda match: (
                prototype_by_skill[match.skill_id]["target_object"] == query_target,
                prototype_by_skill[match.skill_id]["transformation"]
                == query_transformation,
                query_destination
                in prototype_by_skill[match.skill_id]["destination_receptacles"],
                match.cosine_similarity,
            ),
        )
        external_goal = goal_by_skill[external_match.skill_id]
        canonical_goal_match = goal_by_skill[canonical_match.skill_id]
        external_exact = external_goal.casefold() == canonical.casefold()
        canonical_exact = canonical_goal_match.casefold() == canonical.casefold()
        structured_prototype = prototype_by_skill[structured_match.skill_id]
        structured_exact = (
            structured_prototype["target_object"] == query_target
            and structured_prototype["transformation"] == query_transformation
            and query_destination in structured_prototype["destination_receptacles"]
        )
        external_matches += external_exact
        canonical_matches += canonical_exact
        structured_matches += structured_exact
        rows.append(
            {
                "sample_id": record["sample_id"],
                "external_goal": external,
                "canonical_goal": canonical,
                "expected_target": label["target_action_name"],
                "expected_destination": label["destination_action_name"],
                "query_structure": {
                    "target_object": query_target,
                    "transformation": query_transformation,
                    "destination": query_destination,
                },
                "external_query_retrieval": {
                    "skill_id": external_match.skill_id,
                    "score": external_match.cosine_similarity,
                    "canonical_goal": external_goal,
                    "exact_canonical_match": external_exact,
                },
                "canonical_query_retrieval": {
                    "skill_id": canonical_match.skill_id,
                    "score": canonical_match.cosine_similarity,
                    "canonical_goal": canonical_goal_match,
                    "exact_canonical_match": canonical_exact,
                },
                "structured_canonical_retrieval": {
                    "skill_id": structured_match.skill_id,
                    "score": structured_match.cosine_similarity,
                    "canonical_goal": goal_by_skill[structured_match.skill_id],
                    "prototype": structured_prototype,
                    "exact_structure_match": structured_exact,
                },
            }
        )

    output = {
        "sample_count": len(rows),
        "external_query_exact_matches": external_matches,
        "canonical_query_exact_matches": canonical_matches,
        "structured_query_exact_matches": structured_matches,
        "rows": rows,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
