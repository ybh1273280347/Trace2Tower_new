from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.algorithms.semantic_index import SkillEmbeddingIndex
from trace2tower.components.llm_runtime import CommonLLMRuntime, ModelRole
from trace2tower.core.manifests import Benchmark
from trace2tower.core.results import MethodName
from trace2tower.core.trajectory import EpisodeTrajectory, TrajectoryReader
from trace2tower.methods.expel.induction import (
    RuleOperation,
    RuleOperationName,
    apply_rule_operations,
    comparison_messages,
    parse_rule_operations,
    render_trajectory,
    success_messages,
)
from trace2tower.methods.expel.models import ExpeLEpisode, build_execution_library
from trace2tower.methods.expel.retrieval import task_scope


RULE_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_expel_rules",
        "description": "Apply up to four ExpeL rule-set operations derived from the evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": [item.value for item in RuleOperationName],
                            },
                            "rule_number": {"type": "integer", "minimum": 1},
                            "text": {"type": "string"},
                        },
                        "required": ["name", "rule_number", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["operations"],
            "additionalProperties": False,
        },
    },
}


def stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def completed_rule_log(
    path: Path,
    selected: dict[str, tuple[EpisodeTrajectory, ...]],
    *,
    max_comparison_updates: int,
    max_success_updates: int,
    success_chunk_size: int,
) -> tuple[tuple[str, ...], list[dict]] | None:
    if not path.exists():
        return None
    comparison_count = min(
        max_comparison_updates,
        sum(
            item.primary_score < 1.0
            for trajectories in selected.values()
            for item in trajectories
        ),
    )
    success_count = min(
        max_success_updates,
        (len(selected) + success_chunk_size - 1) // success_chunk_size,
    )
    update_log = json.loads(path.read_text(encoding="utf-8"))
    expected_count = comparison_count + success_count
    if len(update_log) < expected_count:
        return None
    return tuple(update_log[-1]["rules_after"][:20]), update_log


def select_training_tasks(
    trajectories: tuple[EpisodeTrajectory, ...],
    benchmark: Benchmark,
    *,
    max_tasks: int,
    seed: int,
) -> dict[str, tuple[EpisodeTrajectory, ...]]:
    grouped = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.sample_id].append(trajectory)

    by_scope = defaultdict(list)
    for sample_id, task_trajectories in grouped.items():
        successes = [item for item in task_trajectories if item.primary_score == 1.0]
        if not successes:
            continue
        has_failure = any(item.primary_score < 1.0 for item in task_trajectories)
        scope = task_scope(benchmark, successes[0].task_goal)
        by_scope[scope].append((not has_failure, stable_rank(seed, sample_id), sample_id))
    for candidates in by_scope.values():
        candidates.sort()

    selected = []
    offsets = {scope: 0 for scope in by_scope}
    while len(selected) < max_tasks:
        added = False
        for scope in sorted(by_scope):
            offset = offsets[scope]
            if offset >= len(by_scope[scope]):
                continue
            selected.append(by_scope[scope][offset][2])
            offsets[scope] += 1
            added = True
            if len(selected) == max_tasks:
                break
        if not added:
            break
    if not selected:
        raise ValueError("ExpeL build pool has no fully successful training tasks")
    return {
        sample_id: tuple(sorted(grouped[sample_id], key=lambda item: item.repeat_id))
        for sample_id in selected
    }


async def extract_rules(
    runtime: CommonLLMRuntime,
    benchmark: Benchmark,
    selected: dict[str, tuple[EpisodeTrajectory, ...]],
    rendered_successes: dict[str, str],
    *,
    max_comparison_updates: int,
    max_success_updates: int,
    success_chunk_size: int,
    max_output_tokens: int,
    checkpoint_path: Path,
) -> tuple[tuple[str, ...], list[dict], list]:
    weighted_rules: tuple[tuple[str, int], ...] = ()
    update_log = []
    usages = []
    comparison_pairs = [
        (sample_id, failure)
        for sample_id, trajectories in selected.items()
        for failure in trajectories
        if failure.primary_score < 1.0
    ][:max_comparison_updates]
    for update_index, (sample_id, failure) in enumerate(comparison_pairs, 1):
        rules_before = tuple(rule for rule, _ in weighted_rules)
        result = await runtime.chat(
            ModelRole.RENDERER,
            comparison_messages(
                benchmark,
                failure.task_goal,
                rendered_successes[sample_id],
                render_trajectory(failure, max_chars=9000),
                rules_before,
            ),
            temperature=0,
            max_output_tokens=max_output_tokens,
            prompt_cache_key=f"expel:{benchmark}:comparison:e41ec9a",
            tools=[RULE_UPDATE_TOOL],
            tool_choice="required",
        )
        operations = rule_operations(result)
        weighted_rules = apply_rule_operations(weighted_rules, operations)
        usages.append(result.usage)
        update_log.append(
            {
                "stage": "compare",
                "sample_id": sample_id,
                "operations": [
                    {
                        "name": operation.name.value,
                        "rule_number": operation.rule_number,
                        "text": operation.text,
                    }
                    for operation in operations
                ],
                "rules_after": [rule for rule, _ in weighted_rules],
            }
        )
        write_json(checkpoint_path, update_log)
        print(
            f"comparison update {update_index}/{len(comparison_pairs)}: "
            f"rules={len(weighted_rules)}",
            flush=True,
        )

    success_histories = [rendered_successes[sample_id] for sample_id in selected]
    chunks = [
        tuple(success_histories[offset : offset + success_chunk_size])
        for offset in range(0, len(success_histories), success_chunk_size)
    ][:max_success_updates]
    for chunk_index, chunk in enumerate(chunks):
        rules_before = tuple(rule for rule, _ in weighted_rules)
        result = await runtime.chat(
            ModelRole.RENDERER,
            success_messages(benchmark, chunk, rules_before),
            temperature=0,
            max_output_tokens=max_output_tokens,
            prompt_cache_key=f"expel:{benchmark}:all-success:e41ec9a",
            tools=[RULE_UPDATE_TOOL],
            tool_choice="required",
        )
        operations = rule_operations(result)
        weighted_rules = apply_rule_operations(weighted_rules, operations)
        usages.append(result.usage)
        update_log.append(
            {
                "stage": "all_success",
                "chunk_index": chunk_index,
                "operations": [
                    {
                        "name": operation.name.value,
                        "rule_number": operation.rule_number,
                        "text": operation.text,
                    }
                    for operation in operations
                ],
                "rules_after": [rule for rule, _ in weighted_rules],
            }
        )
        write_json(checkpoint_path, update_log)
        print(
            f"success update {chunk_index + 1}/{len(chunks)}: rules={len(weighted_rules)}",
            flush=True,
        )
    rules = tuple(rule for rule, _ in weighted_rules[:20])
    if not rules:
        raise ValueError("ExpeL rule extraction produced no valid rules")
    return rules, update_log, usages


async def embed_with_fallback(
    runtime: CommonLLMRuntime,
    sample_ids: tuple[str, ...],
    task_goals: tuple[str, ...],
) -> tuple[dict[str, tuple[float, ...]], list]:
    try:
        result = await runtime.embed(task_goals)
        return dict(zip(sample_ids, result.vectors, strict=True)), [result]
    except Exception:
        if len(sample_ids) == 1:
            raise
        midpoint = len(sample_ids) // 2
        left_vectors, left_results = await embed_with_fallback(
            runtime,
            sample_ids[:midpoint],
            task_goals[:midpoint],
        )
        right_vectors, right_results = await embed_with_fallback(
            runtime,
            sample_ids[midpoint:],
            task_goals[midpoint:],
        )
        return left_vectors | right_vectors, left_results + right_results


def rule_operations(result) -> tuple[RuleOperation, ...]:
    if len(result.tool_calls) == 1 and result.tool_calls[0].name == "update_expel_rules":
        payload = json.loads(result.tool_calls[0].arguments)
        operations = []
        for item in payload["operations"][:4]:
            text = str(item["text"]).strip()
            if text and not text.endswith("."):
                text += "."
            operations.append(
                RuleOperation(
                    RuleOperationName(item["name"]),
                    int(item["rule_number"]),
                    text,
                )
            )
        return tuple(operation for operation in operations if operation.text)
    return parse_rule_operations(result.content or "")


async def main(options: argparse.Namespace) -> int:
    source_bytes = options.pool.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    trajectories = TrajectoryReader.read_jsonl(options.pool)
    if any(
        trajectory.benchmark is not options.benchmark
        or trajectory.method is not MethodName.NO_SKILL
        for trajectory in trajectories
    ):
        raise ValueError("ExpeL requires one-benchmark No-Skill training trajectories")
    selected = select_training_tasks(
        trajectories,
        options.benchmark,
        max_tasks=options.max_training_tasks,
        seed=options.seed,
    )
    successes = {
        sample_id: min(
            (item for item in task_trajectories if item.primary_score == 1.0),
            key=lambda item: (len(item.steps), item.repeat_id),
        )
        for sample_id, task_trajectories in selected.items()
    }
    rendered_successes = {
        sample_id: render_trajectory(trajectory, max_chars=options.max_trajectory_chars)
        for sample_id, trajectory in successes.items()
    }

    options.output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(options.env)
    common = load_yaml(options.config_root / "common.yaml")
    runtime = CommonLLMRuntime(
        max_concurrency=common["global_api_concurrency"],
        max_attempts=common["provider_max_attempts"],
        timeout_seconds=common["provider_timeout_seconds"],
        retry_base_seconds=common["retry_base_seconds"],
    )
    try:
        reused_rules = completed_rule_log(
            options.output_dir / "rule-update-log.partial.json",
            selected,
            max_comparison_updates=options.max_comparison_updates,
            max_success_updates=options.max_success_updates,
            success_chunk_size=options.success_chunk_size,
        )
        if reused_rules is None:
            rules, update_log, rule_usages = await extract_rules(
                runtime,
                options.benchmark,
                selected,
                rendered_successes,
                max_comparison_updates=options.max_comparison_updates,
                max_success_updates=options.max_success_updates,
                success_chunk_size=options.success_chunk_size,
                max_output_tokens=options.rule_max_output_tokens,
                checkpoint_path=options.output_dir / "rule-update-log.partial.json",
            )
        else:
            rules, update_log = reused_rules
            rule_usages = []
            print("reusing completed ExpeL rule-update checkpoint", flush=True)
        ordered_ids = tuple(sorted(selected))
        embedding_results = []
        embedding_by_sample_id = {}
        embedding_checkpoint_path = options.output_dir / "embedding-checkpoint.json"
        if embedding_checkpoint_path.exists():
            checkpoint = json.loads(embedding_checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("source_pool_sha256") == source_sha256:
                embedding_by_sample_id = {
                    str(sample_id): tuple(float(value) for value in vector)
                    for sample_id, vector in checkpoint.get("vectors", {}).items()
                }
        embedding_batch_size = int(common.get("embedding_batch_size", 16))
        missing_ids = tuple(sample_id for sample_id in ordered_ids if sample_id not in embedding_by_sample_id)
        for offset in range(0, len(missing_ids), embedding_batch_size):
            batch_ids = missing_ids[offset : offset + embedding_batch_size]
            batch_vectors, batch_results = await embed_with_fallback(
                runtime,
                batch_ids,
                tuple(successes[sample_id].task_goal for sample_id in batch_ids),
            )
            embedding_by_sample_id.update(batch_vectors)
            embedding_results.extend(batch_results)
            write_json(
                embedding_checkpoint_path,
                {
                    "source_pool_sha256": source_sha256,
                    "vectors": {
                        sample_id: list(vector)
                        for sample_id, vector in sorted(embedding_by_sample_id.items())
                    },
                },
            )
            print(
                f"embedding batch {min(offset + embedding_batch_size, len(missing_ids))}/"
                f"{len(missing_ids)}",
                flush=True,
            )
        embedding_vectors = tuple(embedding_by_sample_id[sample_id] for sample_id in ordered_ids)
    finally:
        await runtime.close()

    episodes = tuple(
        ExpeLEpisode(
            episode_id=f"expel_episode_{hashlib.sha256(sample_id.encode()).hexdigest()[:12]}",
            sample_id=sample_id,
            task_goal=successes[sample_id].task_goal,
            task_scope=task_scope(options.benchmark, successes[sample_id].task_goal),
            trajectory=rendered_successes[sample_id],
        )
        for sample_id in ordered_ids
    )
    episode_index = SkillEmbeddingIndex(
        tuple(episode.episode_id for episode in episodes),
        tuple(embedding_vectors),
        tuple(hashlib.sha256(episode.task_goal.encode()).hexdigest() for episode in episodes),
    )
    library = build_execution_library(
        options.benchmark,
        source_sha256,
        rules,
        episodes,
        episode_index,
    )
    write_json(options.output_dir / "library.json", library.to_record())
    write_json(options.output_dir / "rule-update-log.json", update_log)
    report = {
        "benchmark": options.benchmark.value,
        "library_id": library.library_id,
        "source_pool": options.pool.as_posix(),
        "source_pool_sha256": source_sha256,
        "selected_training_tasks": len(selected),
        "selected_sample_ids": list(ordered_ids),
        "comparison_rule_updates": sum(item["stage"] == "compare" for item in update_log),
        "success_rule_updates": sum(item["stage"] == "all_success" for item in update_log),
        "rule_count": len(rules),
        "episode_count": len(episodes),
        "rule_input_tokens": sum(usage.input_tokens or 0 for usage in rule_usages),
        "rule_output_tokens": sum(usage.output_tokens or 0 for usage in rule_usages),
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
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-training-tasks", type=int, default=20)
    parser.add_argument("--max-comparison-updates", type=int, default=6)
    parser.add_argument("--max-success-updates", type=int, default=2)
    parser.add_argument("--success-chunk-size", type=int, default=4)
    parser.add_argument("--max-trajectory-chars", type=int, default=9000)
    parser.add_argument("--rule-max-output-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-root", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    raise SystemExit(asyncio.run(main(parser.parse_args())))
