from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import write_json
from trace2tower.components.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.core.manifests import Benchmark
from trace2tower.methods.trace2skill.models import build_artifact


ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_trace_patches",
        "description": "Submit concise trajectory-local skill patches.",
        "parameters": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "instance_id": {"type": "string"},
                            "outcome": {"type": "string", "enum": ["success", "error"]},
                            "items": {
                                "type": "array",
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "guidance": {"type": "string"},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["title", "guidance", "evidence"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["instance_id", "outcome", "items"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        },
    },
}

MERGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_merged_patches",
        "description": "Merge trajectory-local patches into reusable non-redundant guidance.",
        "parameters": {
            "type": "object",
            "properties": {
                "patches": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "guidance": {"type": "string"},
                            "when_to_apply": {"type": "string"},
                            "cautions": {"type": "string"},
                        },
                        "required": ["title", "guidance", "when_to_apply", "cautions"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["patches"],
            "additionalProperties": False,
        },
    },
}

SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_skill",
        "description": "Submit the complete portable SKILL.md.",
        "parameters": {
            "type": "object",
            "properties": {"skill_markdown": {"type": "string"}},
            "required": ["skill_markdown"],
            "additionalProperties": False,
        },
    },
}


def load_trajectories(path: Path, *, max_trajectories: int | None) -> list[dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repeat_zero = sorted(
        (record for record in records if int(record["repeat_id"]) == 0),
        key=lambda record: record["sample_id"],
    )
    return repeat_zero[:max_trajectories] if max_trajectories else repeat_zero


def trajectory_text(record: dict, *, max_chars: int) -> str:
    sections = [
        f"INSTANCE: {record['sample_id']}",
        f"TASK: {record['task_goal']}",
        f"FINAL SCORE: {float(record['primary_score']):.4f}",
        f"FINISH REASON: {record['finish_reason']}",
        "TRACE:",
    ]
    for step in record["steps"]:
        action = json.dumps(
            {"name": step.get("action_name"), "arguments": step.get("action_arguments")},
            ensure_ascii=False,
            sort_keys=True,
        )
        observation = str(step.get("observation", "")).replace("\n", " ")[:500]
        next_observation = str(step.get("next_observation", "")).replace("\n", " ")[:500]
        sections.append(
            f"{step['step_index']}. OBS={observation}\nACTION={action}\n"
            f"NEXT={next_observation}\nVALID={step.get('valid_action')}"
        )
    return "\n".join(sections)[:max_chars]


def chunked(items: list, size: int) -> list[list]:
    return [items[offset : offset + size] for offset in range(0, len(items), size)]


def tool_payload(result, name: str) -> dict:
    calls = tuple(call for call in result.tool_calls if call.name == name)
    if len(calls) != 1:
        raise ValueError(f"Trace2Skill call must return exactly one {name} tool call")
    return json.loads(calls[0].arguments)


def domain_description(benchmark: Benchmark) -> str:
    if benchmark is Benchmark.ALFWORLD:
        return (
            "ALFWorld household tasks. The agent observes text states and must use only currently "
            "available navigation and manipulation actions to locate objects, satisfy prerequisites, "
            "perform requested state changes, and place or inspect the correct objects."
        )
    return (
        "WebShop product-purchase tasks. The agent searches a catalog, compares evidence across "
        "products and detail tabs, binds all required options, respects the budget, and purchases "
        "only after verifying the complete conjunction of user constraints."
    )


async def create_initial_skill(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    checkpoint: Path,
    model_role: ModelRole,
) -> str:
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf-8"))["skill_markdown"]
    result = await runtime.chat(
        model_role,
        [
            {
                "role": "system",
                "content": (
                    "Create a weak but usable initial agent skill from parametric domain knowledge. "
                    "This is the Trace2Skill skill-creation starting point before trajectory patches. "
                    "Write concise general procedures and safeguards, not benchmark answers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Domain: {domain_description(benchmark)}\n\n"
                    "Return a complete SKILL.md with a title, scope, core workflow, recovery rules, "
                    "and completion checks. Do not mention Trace2Skill or training examples."
                ),
            },
        ],
        tools=[SKILL_TOOL],
        tool_choice="required",
        temperature=0.2,
        max_output_tokens=2200,
        prompt_cache_key=f"trace2skill:{benchmark}:initial:v1",
    )
    payload = tool_payload(result, "submit_skill")
    write_json(checkpoint, payload)
    return str(payload["skill_markdown"]).strip()


async def analyze_batch(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    initial_skill: str,
    batch: list[dict],
    output_path: Path,
    *,
    max_trace_chars: int,
    model_role: ModelRole,
) -> list[dict]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))["records"]
    expected = {record["sample_id"] for record in batch}
    result = await runtime.chat(
        model_role,
        [
            {
                "role": "system",
                "content": (
                    "You are the parallel patch-proposal stage of Trace2Skill. Analyze each frozen "
                    "trajectory independently against the initial skill. For score 1, preserve the "
                    "minimal successful workflow and infer at most three reusable success patches. "
                    "For score below 1, identify only causal behaviors supported by the trace and "
                    "propose at most three prevention patches. A partial reward is still an error "
                    "trajectory: preserve what worked but patch the missing requirement. Never copy "
                    "specific object names, product IDs, task answers, or benchmark-instance facts. "
                    "Each guidance item must be a concise operational instruction usable on unseen tasks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DOMAIN\n{domain_description(benchmark)}\n\nINITIAL SKILL\n{initial_skill}\n\n"
                    "TRAJECTORIES\n\n"
                    + "\n\n---\n\n".join(
                        trajectory_text(record, max_chars=max_trace_chars) for record in batch
                    )
                ),
            },
        ],
        tools=[ANALYSIS_TOOL],
        tool_choice="required",
        temperature=0.2,
        max_output_tokens=2600,
        prompt_cache_key=f"trace2skill:{benchmark}:trajectory-patch:v1",
    )
    payload = tool_payload(result, "submit_trace_patches")
    actual = {record["instance_id"] for record in payload["records"]}
    unknown = actual - expected
    if unknown:
        raise ValueError(f"Trace2Skill analysis returned unknown instance IDs: {unknown}")
    outcome_by_id = {
        record["sample_id"]: "success" if float(record["primary_score"]) >= 1.0 else "error"
        for record in batch
    }
    by_id = {record["instance_id"]: record for record in payload["records"]}
    normalized = [
        {
            "instance_id": instance_id,
            "outcome": outcome_by_id[instance_id],
            "items": by_id.get(instance_id, {}).get("items", []),
        }
        for instance_id in sorted(expected)
    ]
    payload["records"] = normalized
    write_json(output_path, payload)
    return normalized


async def merge_group(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    patches: list[dict],
    output_path: Path,
    model_role: ModelRole,
) -> list[dict]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))["patches"]
    result = await runtime.chat(
        model_role,
        [
            {
                "role": "system",
                "content": (
                    "You are Trace2Skill's many-to-one patch consolidation stage. Deduplicate "
                    "semantically overlapping guidance, resolve conflicts in favor of repeated and "
                    "causally grounded evidence, preserve distinct success workflows and failure "
                    "safeguards, and remove instance-specific facts. Patch value may be combinatorial, "
                    "so do not drop complementary prerequisites merely because each appears rarely. "
                    "Return a concise portable set of operational patches."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DOMAIN\n{domain_description(benchmark)}\n\nPATCHES\n"
                    + json.dumps(patches, ensure_ascii=False, sort_keys=True)
                ),
            },
        ],
        tools=[MERGE_TOOL],
        tool_choice="required",
        temperature=0.2,
        max_output_tokens=3200,
        prompt_cache_key=f"trace2skill:{benchmark}:hierarchical-merge:v1",
    )
    payload = tool_payload(result, "submit_merged_patches")
    if not payload["patches"]:
        raise ValueError("Trace2Skill merge produced no patches")
    write_json(output_path, payload)
    return payload["patches"]


async def render_final_skill(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    initial_skill: str,
    patches: list[dict],
    output_path: Path,
    model_role: ModelRole,
) -> str:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))["skill_markdown"]
    result = await runtime.chat(
        model_role,
        [
            {
                "role": "system",
                "content": (
                    "Materialize the final Trace2Skill skill-creation artifact. Integrate all merged "
                    "patches into one coherent static SKILL.md. Preserve useful initial guidance, "
                    "remove redundancy and conflicts, and keep the skill under 500 lines. It will be "
                    "preloaded unchanged for unseen tasks, with no test-time retrieval."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DOMAIN\n{domain_description(benchmark)}\n\nINITIAL SKILL\n{initial_skill}\n\n"
                    f"MERGED PATCHES\n{json.dumps(patches, ensure_ascii=False, sort_keys=True)}\n\n"
                    "Return only the complete operational SKILL.md through the required tool. Do not "
                    "mention training trajectories, scores, model names, or specific task instances."
                ),
            },
        ],
        tools=[SKILL_TOOL],
        tool_choice="required",
        temperature=0.2,
        max_output_tokens=5000,
        prompt_cache_key=f"trace2skill:{benchmark}:final-skill:v1",
    )
    payload = tool_payload(result, "submit_skill")
    skill = str(payload["skill_markdown"]).strip()
    if len(skill.splitlines()) > 500:
        raise ValueError("Trace2Skill final skill exceeds the 500-line upstream limit")
    write_json(output_path, payload)
    return skill


async def main(options: argparse.Namespace) -> int:
    load_dotenv(options.env)
    benchmark = Benchmark(options.benchmark)
    model_role = ModelRole(options.model_role)
    if model_role is ModelRole.EMBEDDING:
        raise ValueError("Trace2Skill construction requires a chat model role")
    trajectories = load_trajectories(
        options.trajectory_pool,
        max_trajectories=options.max_trajectories,
    )
    if not trajectories:
        raise ValueError("Trace2Skill build selected no repeat-0 trajectories")
    if {record["benchmark"] for record in trajectories} != {benchmark.value}:
        raise ValueError("Trace2Skill trajectory pool benchmark mismatch")
    work_dir = options.output.parent / f"{options.output.stem}-build"
    analysis_dir = options.analysis_cache_dir or work_dir / "analysis"
    merge_dir = work_dir / "merge"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)
    runtime = CommonLLMRuntime(
        max_concurrency=options.max_concurrency,
        max_attempts=4,
        timeout_seconds=180,
        retry_base_seconds=1,
    )
    try:
        initial_skill_path = options.initial_skill_cache or work_dir / "initial-skill.json"
        initial_skill = await create_initial_skill(
            runtime,
            benchmark,
            initial_skill_path,
            model_role,
        )
        batches = chunked(trajectories, options.analysis_batch_size)
        batch_records = await asyncio.gather(
            *(
                analyze_batch(
                    runtime,
                    benchmark,
                    initial_skill,
                    batch,
                    analysis_dir / f"batch-{index:04d}.json",
                    max_trace_chars=options.max_trace_chars,
                    model_role=model_role,
                )
                for index, batch in enumerate(batches)
            )
        )
        patches = [
            {
                "instance_id": record["instance_id"],
                "outcome": record["outcome"],
                **item,
            }
            for records in batch_records
            for record in records
            for item in record["items"]
            if options.signal == "combined" or record["outcome"] == options.signal
        ]
        if not patches:
            raise ValueError("Trace2Skill analysis produced no trajectory-local patches")
        level = 0
        while len(patches) > options.merge_batch_size:
            groups = chunked(patches, options.merge_batch_size)
            merged = await asyncio.gather(
                *(
                    merge_group(
                        runtime,
                        benchmark,
                        group,
                        merge_dir / f"level-{level:02d}-batch-{index:04d}.json",
                        model_role,
                    )
                    for index, group in enumerate(groups)
                )
            )
            patches = [patch for group in merged for patch in group]
            level += 1
        final_patches = await merge_group(
            runtime,
            benchmark,
            patches,
            merge_dir / f"level-{level:02d}-final.json",
            model_role,
        )
        skill = await render_final_skill(
            runtime,
            benchmark,
            initial_skill,
            final_patches,
            work_dir / "final-skill.json",
            model_role,
        )
    finally:
        await runtime.close()

    selected_trajectories = [
        record
        for record in trajectories
        if options.signal == "combined"
        or (
            options.signal == "success"
            and float(record["primary_score"]) >= 1.0
        )
        or (
            options.signal == "error"
            and float(record["primary_score"]) < 1.0
        )
    ]
    success_count = sum(
        float(record["primary_score"]) >= 1.0 for record in selected_trajectories
    )
    artifact = build_artifact(
        benchmark=benchmark,
        source_pool_sha256=hashlib.sha256(options.trajectory_pool.read_bytes()).hexdigest(),
        author_model=options.author_model,
        trajectory_count=len(selected_trajectories),
        success_count=success_count,
        failure_count=len(selected_trajectories) - success_count,
        skill_markdown=skill,
        evolution_signal=options.signal,
    )
    write_json(options.output, artifact.to_record())
    write_json(
        work_dir / "build-report.json",
        {
            "artifact_id": artifact.artifact_id,
            "benchmark": benchmark.value,
            "trajectory_pool": options.trajectory_pool.as_posix(),
            "evolution_signal": options.signal,
            "trajectory_count": len(selected_trajectories),
            "success_count": success_count,
            "failure_count": len(selected_trajectories) - success_count,
            "analysis_batch_count": len(batches),
            "final_patch_count": len(final_patches),
            "skill_lines": len(skill.splitlines()),
            "skill_chars": len(skill),
        },
    )
    print(json.dumps(artifact.to_record(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--trajectory-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author-model", default="deepseek-v4-flash")
    parser.add_argument(
        "--model-role",
        choices=(ModelRole.AGENT.value, ModelRole.RENDERER.value),
        default=ModelRole.AGENT.value,
    )
    parser.add_argument("--analysis-batch-size", type=int, default=4)
    parser.add_argument("--merge-batch-size", type=int, default=32)
    parser.add_argument("--max-trace-chars", type=int, default=9000)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--max-concurrency", type=int, default=30)
    parser.add_argument(
        "--signal",
        choices=("combined", "error", "success"),
        default="combined",
    )
    parser.add_argument("--analysis-cache-dir", type=Path)
    parser.add_argument("--initial-skill-cache", type=Path)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
