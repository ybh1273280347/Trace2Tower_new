from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import yaml

from trace2tower.evaluation import (
    ConstructionCost,
    aggregate_method,
    paired_bootstrap,
    unresolved_failures,
)
from trace2tower.manifests import (
    Benchmark,
    ExperimentSplit,
    expand_manifest_repeats,
    read_manifest,
)
from trace2tower.results import EpisodeResult, MethodName


def assignments(values: list[str]) -> dict[MethodName, list[Path]]:
    parsed = defaultdict(list)
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not raw_path:
            raise ValueError(f"expected METHOD=PATH assignment: {value}")
        parsed[MethodName(name)].append(Path(raw_path))
    return dict(parsed)


def read_results(paths: list[Path]) -> tuple[EpisodeResult, ...]:
    return tuple(
        EpisodeResult.from_record(json.loads(line))
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def read_jsonl(paths: list[Path]) -> tuple[dict, ...]:
    return tuple(
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def sha256_files(paths: list[Path]) -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as output_file:
        temporary_path = Path(output_file.name)
        output_file.write(content)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def main(options: argparse.Namespace) -> int:
    config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    if config != {
        "baseline_method": "no_skill",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 42,
        "confidence_level": 0.95,
    }:
        raise ValueError("evaluation config changed outside the frozen contract")
    benchmark = Benchmark(options.benchmark)
    split = ExperimentSplit(options.split)
    entries = read_manifest(options.manifest)
    entries = [
        entry
        for entry in entries
        if entry.benchmark is benchmark
        and entry.split is split
        and (not options.sample_id or entry.sample_id in set(options.sample_id))
    ]
    if options.sample_id and {entry.sample_id for entry in entries} != set(options.sample_id):
        raise ValueError("one or more selected sample IDs are absent from the manifest")
    entries = expand_manifest_repeats(entries, options.repeat_id)
    if not entries:
        raise ValueError("evaluation selection is empty")
    selected_keys = {
        (entry.benchmark, entry.split, entry.sample_id, entry.repeat_id)
        for entry in entries
    }

    result_paths = assignments(options.results)
    error_paths = assignments(options.errors)
    construction_paths = assignments(options.construction_cost)
    if any(len(paths) != 1 for paths in construction_paths.values()):
        raise ValueError("each method accepts at most one construction cost record")
    results_by_method = {}
    for method, paths in result_paths.items():
        loaded = read_results(paths)
        results_by_method[method] = tuple(
            result
            for result in loaded
            if (
                result.benchmark,
                result.split,
                result.sample_id,
                result.repeat_id,
            )
            in selected_keys
        )
    costs = {
        method: ConstructionCost.from_record(
            json.loads(paths[0].read_text(encoding="utf-8"))
        )
        for method, paths in construction_paths.items()
    }
    if not set(costs) <= set(results_by_method):
        raise ValueError("construction cost was provided for an unevaluated method")
    audits = {}
    aggregates = {}
    for method, results in results_by_method.items():
        audit, aggregate = aggregate_method(
            entries,
            results,
            benchmark=benchmark,
            split=split,
            method=method,
            construction_cost=costs.get(method),
        )
        audits[method.value] = audit.to_record()
        aggregates[method.value] = aggregate.to_record()

    baseline_method = MethodName(config["baseline_method"])
    pairwise = []
    if baseline_method in results_by_method:
        for method in sorted(results_by_method, key=lambda item: item.value):
            if method is baseline_method:
                continue
            pairwise.append(
                paired_bootstrap(
                    results_by_method[baseline_method],
                    results_by_method[method],
                    benchmark=benchmark,
                    split=split,
                    baseline_method=baseline_method,
                    candidate_method=method,
                    bootstrap_samples=int(config["bootstrap_samples"]),
                    bootstrap_seed=int(config["bootstrap_seed"]),
                    confidence_level=float(config["confidence_level"]),
                ).to_record()
            )

    all_results = tuple(
        result for results in results_by_method.values() for result in results
    )
    if not set(error_paths) <= set(results_by_method):
        raise ValueError("errors were provided for an unevaluated method")
    all_errors = []
    for method, paths in error_paths.items():
        records = read_jsonl(paths)
        if any(record.get("method") != method.value for record in records):
            raise ValueError("error file method does not match its assignment")
        all_errors.extend(
            record
            for record in records
            if (
                Benchmark(record["benchmark"]),
                ExperimentSplit(record["split"]),
                str(record["sample_id"]),
                int(record["repeat_id"]),
            )
            in selected_keys
        )
    failures = unresolved_failures(all_errors, all_results)
    source_hashes = {
        method.value: sha256_files(paths) for method, paths in result_paths.items()
    }
    report = {
        "benchmark": benchmark.value,
        "split": split.value,
        "manifest_sha256": hashlib.sha256(options.manifest.read_bytes()).hexdigest(),
        "evaluation_config": config,
        "evaluation_config_sha256": hashlib.sha256(
            options.config.read_bytes()
        ).hexdigest(),
        "selected_sample_ids": sorted({entry.sample_id for entry in entries}),
        "selected_repeat_ids": sorted({entry.repeat_id for entry in entries}),
        "expected_episode_count": len(entries),
        "result_source_hashes": source_hashes,
        "error_source_hashes": {
            method.value: sha256_files(paths)
            for method, paths in error_paths.items()
        },
        "construction_cost_source_hashes": {
            method.value: sha256_files(paths)
            for method, paths in construction_paths.items()
        },
        "audits": audits,
        "methods": aggregates,
        "unresolved_failure_count": len(failures),
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(options.output_dir / "aggregate.json", report)
    write_json(options.output_dir / "pairwise.json", {"comparisons": pairwise})
    write_text(
        options.output_dir / "failures.jsonl",
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in failures),
    )
    write_text(options.output_dir / "aggregate.md", render_markdown(report, pairwise))
    print(yaml.safe_dump({**report, "pairwise": pairwise}, sort_keys=False))
    return 0


def render_markdown(report: dict, pairwise: list[dict]) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        f"Benchmark: `{report['benchmark']}`  ",
        f"Split: `{report['split']}`  ",
        f"Expected episodes per method: {report['expected_episode_count']}",
        "",
        "| Method | Primary metric | Mean | Steps | Invalid rate | Billable coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, aggregate in sorted(report["methods"].items()):
        lines.append(
            f"| {method} | {aggregate['primary_metric']} | "
            f"{aggregate['primary_metric_mean']:.6f} | "
            f"{aggregate['mean_steps']:.3f} | "
            f"{aggregate['invalid_action_rate']:.6f} | "
            f"{aggregate['billable_token_coverage']:.3f} |"
        )
    if pairwise:
        lines.extend(
            (
                "",
                "| Candidate vs No-Skill | Mean difference | 95% CI | Episodes | Tasks |",
                "|---|---:|---:|---:|---:|",
            )
        )
        for comparison in pairwise:
            lower, upper = comparison["confidence_interval"]
            lines.append(
                f"| {comparison['candidate_method']} | "
                f"{comparison['mean_difference']:.6f} | "
                f"[{lower:.6f}, {upper:.6f}] | {comparison['pair_count']} | "
                f"{comparison['task_count']} |"
            )
    lines.extend(("", f"Unresolved failures: {report['unresolved_failure_count']}", ""))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(Benchmark), required=True)
    parser.add_argument("--split", choices=tuple(ExperimentSplit), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", action="append", default=[], required=True)
    parser.add_argument("--errors", action="append", default=[])
    parser.add_argument("--construction-cost", action="append", default=[])
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--repeat-id", action="append", type=int, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/evaluation.yaml"),
    )
    raise SystemExit(main(parser.parse_args()))
