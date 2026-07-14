from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import yaml
from scripts.experiments.analyze.check_skillx_upstream import inspect_skillx
from dotenv import load_dotenv
from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json

from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.skillx.models import (
    SkillXCard,
    SkillXExecutionLibrary,
    SkillXPlan,
    build_execution_library,
    digest,
    plan_text,
    skill_text,
)
from trace2tower.semantic_index import SkillEmbeddingIndex


def parse_source_library(
    payload: dict,
    benchmark: Benchmark,
) -> tuple[tuple[SkillXPlan, ...], tuple[SkillXCard, ...]]:
    if payload.get("benchmark") != benchmark.value:
        raise ValueError("SkillX source library benchmark does not match")
    source_skills = payload.get("skills")
    if not isinstance(source_skills, dict):
        raise ValueError("SkillX source library has no skills object")
    plans = []
    for task, record in source_skills.get("planning", {}).items():
        if record.get("task") != task or not record.get("plan"):
            raise ValueError("SkillX planning record does not match its task key")
        semantic = {"task": task, "plan": record["plan"]}
        plans.append(
            SkillXPlan(
                plan_id=f"skillx_plan_{digest(semantic)[:12]}",
                source_sha256=digest(record),
                task=task,
                plan=str(record["plan"]),
            )
        )
    skills = []
    for skill_type in ("functional", "atomic"):
        for record in source_skills.get(skill_type, []):
            content = str(record.get("content", ""))
            tools = list(record.get("tools", []))
            if benchmark is Benchmark.ALFWORLD:
                content = content.replace("apis.alfworld.take_action", "take_action")
                tools = [
                    "take_action" if tool == "apis.alfworld.take_action" else tool
                    for tool in tools
                ]
            semantic = {
                "name": record.get("name"),
                "document": record.get("document"),
                "content": content,
                "tools": tools,
                "skill_type": skill_type,
            }
            required_text = ("name", "document", "content")
            if any(
                not isinstance(semantic[field], str) or not semantic[field]
                for field in required_text
            ):
                raise ValueError("SkillX execution skill requires name, document, and content")
            if not isinstance(semantic["tools"], list) or any(
                not isinstance(tool, str) or not tool for tool in semantic["tools"]
            ):
                raise ValueError("SkillX execution skill has invalid tools")
            skills.append(
                SkillXCard(
                    skill_id=f"skillx_{digest(semantic)[:12]}",
                    source_sha256=digest(record),
                    name=semantic["name"],
                    document=semantic["document"],
                    content=semantic["content"],
                    tools=tuple(semantic["tools"]),
                    skill_type=skill_type,
                )
            )
    if not plans and not skills:
        raise ValueError("SkillX source library is empty")
    return tuple(plans), tuple(skills)


def reusable_vectors(
    output_path: Path,
    text_hash_by_id: dict[str, str],
) -> dict[str, tuple[float, ...]]:
    if not output_path.exists():
        return {}
    existing = SkillXExecutionLibrary.from_record(
        json.loads(output_path.read_text(encoding="utf-8"))
    )
    reusable = {}
    for index in (existing.plan_index, existing.skill_index):
        reusable.update(
            {
                skill_id: vector
                for skill_id, vector, text_hash in zip(
                    index.skill_ids,
                    index.vectors,
                    index.text_hashes,
                    strict=True,
                )
                if text_hash_by_id.get(skill_id) == text_hash
            }
        )
    return reusable


async def main(options: argparse.Namespace) -> int:
    source_bytes = options.source_library.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    payload = json.loads(source_bytes)
    plans, skills = parse_source_library(payload, options.benchmark)
    upstream = inspect_skillx(options.skillx_root)
    plan_text_by_id = {plan.plan_id: plan_text(plan) for plan in plans}
    skill_text_by_id = {skill.skill_id: skill_text(skill) for skill in skills}
    text_by_id = plan_text_by_id | skill_text_by_id
    text_hash_by_id = {
        skill_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for skill_id, text in text_by_id.items()
    }
    output_path = options.output_dir / "library.json"
    reused = reusable_vectors(output_path, text_hash_by_id)
    output_reused_count = len(reused)
    checkpoint_path = options.output_dir / "embedding-checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("source_library_sha256") == source_sha256:
            reused.update(
                {
                    skill_id: tuple(item["vector"])
                    for skill_id, item in checkpoint.get("vectors", {}).items()
                    if text_hash_by_id.get(skill_id) == item.get("text_hash")
                }
            )
    checkpoint_reused_count = len(reused) - output_reused_count
    missing_ids = tuple(sorted(set(text_by_id) - set(reused)))

    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    new_vectors = {}
    embedding_results = []
    batch_size = int(common["embedding_batch_size"])
    try:
        for offset in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[offset : offset + batch_size]
            result = await runtime.embed(
                [text_by_id[skill_id] for skill_id in batch_ids]
            )
            embedding_results.append(result)
            new_vectors.update(zip(batch_ids, result.vectors, strict=True))
            write_json(
                checkpoint_path,
                {
                    "source_library_sha256": source_sha256,
                    "vectors": {
                        skill_id: {
                            "text_hash": text_hash_by_id[skill_id],
                            "vector": vector,
                        }
                        for skill_id, vector in sorted((reused | new_vectors).items())
                    },
                },
            )
    finally:
        await runtime.close()
    vectors = reused | new_vectors

    def build_index(ids: tuple[str, ...]) -> SkillEmbeddingIndex:
        return SkillEmbeddingIndex(
            ids,
            tuple(vectors[skill_id] for skill_id in ids),
            tuple(text_hash_by_id[skill_id] for skill_id in ids),
        )

    plan_ids = tuple(plan.plan_id for plan in sorted(plans, key=lambda item: item.plan_id))
    skill_ids = tuple(skill.skill_id for skill in sorted(skills, key=lambda item: item.skill_id))
    library = build_execution_library(
        options.benchmark,
        source_sha256,
        upstream["commit"],
        plans,
        skills,
        build_index(plan_ids),
        build_index(skill_ids),
    )
    write_json(output_path, library.to_record())
    report = {
        "benchmark": options.benchmark.value,
        "library_id": library.library_id,
        "source_library_sha256": source_sha256,
        "skillx_commit": upstream["commit"],
        "plan_count": len(plans),
        "skill_count": len(skills),
        "reused_embedding_count": len(reused),
        "reused_library_embedding_count": output_reused_count,
        "reused_checkpoint_embedding_count": checkpoint_reused_count,
        "new_embedding_count": len(missing_ids),
        "embedding_input_tokens": sum(
            result.usage.input_tokens or 0 for result in embedding_results
        ),
    }
    write_json(options.output_dir / "report.json", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Benchmark, choices=tuple(Benchmark), required=True)
    parser.add_argument("--source-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--skillx-root", type=Path, default=Path("third_party/SkillX"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
