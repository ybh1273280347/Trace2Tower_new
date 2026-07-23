from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import fmean
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "deliverables" / "trace2tower-main-report-data-20260721"

RESULT_FIELDS = (
    "run_id",
    "benchmark",
    "split",
    "method",
    "sample_id",
    "repeat_id",
    "primary_score",
    "success",
    "steps",
    "invalid_actions",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "skill_ids",
    "skill_context_chars",
    "context_skill_ids",
    "skill_context_sha256",
    "chat_input_tokens",
    "chat_output_tokens",
    "error",
)

RUNS = {
    # Main ALFWorld table.
    "alfworld-test-v1-flash-noskill-r0": ("main", 134),
    "alfworld-test-v1-flash-manual-event-policy-r0": ("main", 134),
    "trace2skill-gpt54-p310-alfworld-test-r0": ("main", 134),
    "trace2skill-gpt54-p310-error-alfworld-test-r0": ("main", 134),
    "alfworld-test-v1-flash-skillx-global-p310-r0": ("main", 134),
    "alfworld-test-expel-p310-flash-r0": ("main", 134),
    "alfworld-ablation-v1-plan-rewrite-high-only-flash-r0": ("main", 134),
    # Main WebShop table.
    "webshop-original-concept-v1-validation-flash-noskill-r1": ("main", 100),
    "webshop-skillx-native-inference-p100-validation-flash-r1": ("main", 100),
    "webshop-validation-expel-p100-flash-r0": ("main", 100),
    "trace2skill-gpt54-p100-webshop-validation-r0": ("main", 100),
    "trace2skill-gpt54-p100-error-webshop-validation-r0": ("main", 100),
    "webshop-expert-crafted-skills-validation-flash-r0": ("main", 100),
    "webshop-alfworld-v17-replication-p100-validation-r0": ("main", 100),
    # ALFWorld single-run structural case study. It is excluded from the main table.
    "alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0": ("case_study", 134),
    "alfworld-ablation-v3-formal-no-transition-flash-r0": ("ablation", 134),
    "alfworld-ablation-v3-formal-no-outcome-flash-r0": ("ablation", 134),
    "alfworld-ablation-v3-formal-no-contrastive-flash-r0": ("ablation", 134),
    # Cross-model figures. Flash runs above are reused rather than duplicated.
    "cross-dsflash-render-gpt54-agent-alfworld-tower-r0": ("cross_model", 134),
    "cross-gpt54-agent-alfworld-noskill-r0": ("cross_model", 134),
    "cross-dsflash-render-gpt54-agent-webshop-tower-r0": ("cross_model", 100),
    "cross-gpt54-agent-webshop-noskill-r0": ("cross_model", 100),
    "generalize-gpt54-render-dspro-agent-alfworld-tower-r0": ("cross_model", 134),
    "generalize-dspro-agent-alfworld-noskill-r0": ("cross_model", 134),
    "generalize-gpt54-render-dspro-agent-webshop-tower-r0": ("cross_model", 100),
    "generalize-dspro-agent-webshop-noskill-r0": ("cross_model", 100),
}

EXPORT_NAMES = {
    "alfworld-test-v1-flash-noskill-r0": "alfworld-main-noskill.jsonl",
    "alfworld-test-v1-flash-manual-event-policy-r0": "alfworld-main-expert-crafted.jsonl",
    "trace2skill-gpt54-p310-alfworld-test-r0": "alfworld-main-trace2skill-combined.jsonl",
    "trace2skill-gpt54-p310-error-alfworld-test-r0": "alfworld-main-trace2skill-error.jsonl",
    "alfworld-test-v1-flash-skillx-global-p310-r0": "alfworld-main-skillx-no-rewrite.jsonl",
    "alfworld-test-expel-p310-flash-r0": "alfworld-main-expel.jsonl",
    "alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0": "alfworld-full-mid-case-study.jsonl",
    "webshop-original-concept-v1-validation-flash-noskill-r1": "webshop-main-noskill.jsonl",
    "webshop-skillx-native-inference-p100-validation-flash-r1": "webshop-main-skillx.jsonl",
    "webshop-validation-expel-p100-flash-r0": "webshop-main-expel.jsonl",
    "trace2skill-gpt54-p100-webshop-validation-r0": "webshop-main-trace2skill-combined.jsonl",
    "trace2skill-gpt54-p100-error-webshop-validation-r0": "webshop-main-trace2skill-error.jsonl",
    "webshop-expert-crafted-skills-validation-flash-r0": "webshop-main-expert-crafted.jsonl",
    "webshop-alfworld-v17-replication-p100-validation-r0": "webshop-main-trace2tower.jsonl",
    "alfworld-ablation-v1-plan-rewrite-high-only-flash-r0": "alfworld-main-trace2tower-high-only.jsonl",
    "alfworld-ablation-v3-formal-no-transition-flash-r0": "alfworld-ablation-no-transition.jsonl",
    "alfworld-ablation-v3-formal-no-outcome-flash-r0": "alfworld-ablation-no-outcome.jsonl",
    "alfworld-ablation-v3-formal-no-contrastive-flash-r0": "alfworld-ablation-no-contrastive.jsonl",
    "cross-dsflash-render-gpt54-agent-alfworld-tower-r0": "alfworld-cross-gpt54-trace2tower.jsonl",
    "cross-gpt54-agent-alfworld-noskill-r0": "alfworld-cross-gpt54-noskill.jsonl",
    "cross-dsflash-render-gpt54-agent-webshop-tower-r0": "webshop-cross-gpt54-trace2tower.jsonl",
    "cross-gpt54-agent-webshop-noskill-r0": "webshop-cross-gpt54-noskill.jsonl",
    "generalize-gpt54-render-dspro-agent-alfworld-tower-r0": "alfworld-cross-deepseek-pro-trace2tower.jsonl",
    "generalize-dspro-agent-alfworld-noskill-r0": "alfworld-cross-deepseek-pro-noskill.jsonl",
    "generalize-gpt54-render-dspro-agent-webshop-tower-r0": "webshop-cross-deepseek-pro-trace2tower.jsonl",
    "generalize-dspro-agent-webshop-noskill-r0": "webshop-cross-deepseek-pro-noskill.jsonl",
}

MAIN_TABLES = {
    "alfworld": {
        "No-Skill": "alfworld-test-v1-flash-noskill-r0",
        "Expert-Crafted Skills": "alfworld-test-v1-flash-manual-event-policy-r0",
        "Trace2Skill +Combined": "trace2skill-gpt54-p310-alfworld-test-r0",
        "Trace2Skill +Error": "trace2skill-gpt54-p310-error-alfworld-test-r0",
        "SkillX no-rewrite": "alfworld-test-v1-flash-skillx-global-p310-r0",
        "ExpeL": "alfworld-test-expel-p310-flash-r0",
        "Trace2Tower High-only": "alfworld-ablation-v1-plan-rewrite-high-only-flash-r0",
    },
    "webshop": {
        "No-Skill": "webshop-original-concept-v1-validation-flash-noskill-r1",
        "SkillX P100": "webshop-skillx-native-inference-p100-validation-flash-r1",
        "ExpeL P100": "webshop-validation-expel-p100-flash-r0",
        "Trace2Skill +Combined": "trace2skill-gpt54-p100-webshop-validation-r0",
        "Trace2Skill +Error": "trace2skill-gpt54-p100-error-webshop-validation-r0",
        "Expert-Crafted Skills": "webshop-expert-crafted-skills-validation-flash-r0",
        "Trace2Tower P100": "webshop-alfworld-v17-replication-p100-validation-r0",
    },
}

COPY_FILES = {
    "alfworld-main-trace2tower-snapshot.json": (
        "artifacts/trace2tower/alfworld/original-concept-v17/p310/tower.json"
    ),
}

EXPECTED_MAIN = {
    "alfworld-test-v1-flash-noskill-r0": {"score": 0.5298507463},
    "alfworld-test-v1-flash-manual-event-policy-r0": {"score": 0.7611940299},
    "trace2skill-gpt54-p310-alfworld-test-r0": {"score": 0.5895522388},
    "trace2skill-gpt54-p310-error-alfworld-test-r0": {"score": 0.6194029851},
    "alfworld-test-v1-flash-skillx-global-p310-r0": {"score": 0.8134328358},
    "alfworld-test-expel-p310-flash-r0": {"score": 0.8059701493},
    "alfworld-ablation-v1-plan-rewrite-high-only-flash-r0": {
        "score": 0.8582089552
    },
    "webshop-original-concept-v1-validation-flash-noskill-r1": {"score": 0.65235},
    "webshop-skillx-native-inference-p100-validation-flash-r1": {
        "score": 0.6842666667
    },
    "webshop-validation-expel-p100-flash-r0": {"score": 0.63348333},
    "trace2skill-gpt54-p100-webshop-validation-r0": {"score": 0.59685},
    "trace2skill-gpt54-p100-error-webshop-validation-r0": {"score": 0.6283333333},
    "webshop-expert-crafted-skills-validation-flash-r0": {"score": 0.70085},
    "webshop-alfworld-v17-replication-p100-validation-r0": {
        "score": 0.7147666667
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(run_id: str, expected_count: int) -> list[dict]:
    run_root = ROOT / "artifacts" / "runs" / run_id
    paths = sorted(run_root.rglob("results.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no results.jsonl under {run_root}")
    rows = [row for path in paths for row in read_jsonl(path)]
    keys = [(row["sample_id"], int(row["repeat_id"])) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task keys in {run_id}: {duplicates[:5]}")
    if len(rows) != expected_count:
        raise ValueError(f"{run_id}: got {len(rows)} rows, expected {expected_count}")
    if any(row.get("run_id") != run_id for row in rows):
        raise ValueError(f"{run_id}: row-level run_id mismatch")
    if any(row.get("error") for row in rows):
        raise ValueError(f"{run_id}: formal results contain unresolved errors")
    return sorted(rows, key=lambda row: (row["sample_id"], int(row["repeat_id"])))


def sanitize(row: dict) -> dict:
    return {field: row.get(field) for field in RESULT_FIELDS if field in row}


def summarize(rows: list[dict]) -> dict:
    scores = [float(row["primary_score"]) for row in rows]
    full_success = [score >= 1.0 - 1e-12 for score in scores]
    return {
        "n": len(rows),
        "mean_primary_score": fmean(scores),
        "full_success_count": sum(full_success),
        "full_success_rate": fmean(full_success),
        "mean_steps": fmean(float(row["steps"]) for row in rows),
        "mean_invalid_actions": fmean(float(row["invalid_actions"]) for row in rows),
        "mean_input_tokens": fmean(float(row["input_tokens"]) for row in rows),
        "mean_skill_context_chars": fmean(
            float(row.get("skill_context_chars") or 0) for row in rows
        ),
        "finish_reason_counts": dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
    }


def export_runs(output: Path) -> dict[str, list[dict]]:
    loaded = {}
    for run_id, (role, expected_count) in RUNS.items():
        rows = load_run(run_id, expected_count)
        loaded[run_id] = rows
        destination = output / EXPORT_NAMES[run_id]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(
                json.dumps(sanitize(row), ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    return loaded


def export_main_tables(output: Path, loaded: dict[str, list[dict]]) -> dict:
    tables = {}
    for benchmark, methods in MAIN_TABLES.items():
        records = []
        for method, run_id in methods.items():
            summary = summarize(loaded[run_id])
            records.append({"method": method, "run_id": run_id, **summary})
        tables[benchmark] = records
        write_json(output / f"{benchmark}-main-table.json", records)
    write_json(output / "main-tables.json", tables)
    return tables


def export_task_family_map(output: Path) -> int:
    source = ROOT / "clean_artifacts" / "alfworld" / "manifests" / "alfworld_test.jsonl"
    rows = read_jsonl(source)
    sanitized = [
        {"sample_id": row["sample_id"], "task_family": row["task_family"]}
        for row in rows
    ]
    write_json(output / "alfworld-task-family-map.json", sanitized)
    return len(sanitized)


def copy_sources(output: Path) -> None:
    for destination, source in COPY_FILES.items():
        source_path = ROOT / source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination_path = output / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def verify_main(catalog: dict) -> dict:
    checks = []
    for run_id, expectation in EXPECTED_MAIN.items():
        observed = catalog[run_id]["summary"]["mean_primary_score"]
        expected = expectation["score"]
        difference = observed - expected
        checks.append(
            {
                "run_id": run_id,
                "expected_mean_primary_score": expected,
                "observed_mean_primary_score": observed,
                "absolute_difference": abs(difference),
                "passed": abs(difference) <= 1e-8,
            }
        )
    return {
        "all_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def source_map() -> dict:
    return {
        "publication_boundary": {
            "included": [
                "official aggregate results",
                "whitelisted per-task metrics",
                "ALFWorld task-family labels without task text",
                "the ALFWorld Tower snapshot used by structural figures",
                "plotting scripts and rendered figures",
            ],
            "excluded": [
                "training trajectories",
                "evaluation trajectories and observations",
                "task/goal text",
                "private manifests",
                "credentials and environment files",
                "deprecated, diagnostic, exploratory, or invalid runs",
            ],
        },
        "tables": {
            "ALFWorld main": {
                "aggregate": "tables/alfworld-main-table.json",
                "rows": list(MAIN_TABLES["alfworld"].values()),
            },
            "WebShop main": {
                "aggregate": "tables/webshop-main-table.json",
                "rows": list(MAIN_TABLES["webshop"].values()),
            },
            "ALFWorld task families": {
                "aggregate": "figure_data/report-data.json#alfworld_family_results",
                "family_map": "figure_data/alfworld-task-family-map.json",
                "rows": list(MAIN_TABLES["alfworld"].values()),
            },
            "ALFWorld structural case study": {
                "aggregate": "formal_results/alfworld-build-ablation-formal-rewrite-results.json",
                "rows": [
                    "alfworld-test-v1-flash-v18-budgeted-rewrite-gpt54-full-r0",
                    "alfworld-ablation-v3-formal-no-transition-flash-r0",
                    "alfworld-ablation-v3-formal-no-outcome-flash-r0",
                    "alfworld-ablation-v3-formal-no-contrastive-flash-r0",
                ],
                "publication_role": "single-run case study; excluded from the main effect claim",
                "semantic_only": "No compliant High was produced, so no rollout exists.",
            },
        },
        "figures": {
            "main-performance": ["figure_data/report-data.json", "tables/main-tables.json"],
            "quality-cost-tradeoff": ["figure_data/report-data.json", "tables/main-tables.json"],
            "tower-structure": ["tower/alfworld-p310-tower.json"],
            "tower-embedding-map": ["tower/alfworld-p310-tower.json"],
            "tower-compression-utilization": [
                "tower/alfworld-p310-tower.json",
                "figure_data/report-data.json",
            ],
            "alfworld-family-heatmap": [
                "figure_data/report-data.json",
                "figure_data/alfworld-task-family-map.json",
            ],
            "cross-model-tower-gains": ["figure_data/cross-model-analysis-data.json"],
            "alfworld-cross-model-family-spectrum": [
                "figure_data/cross-model-analysis-data.json",
                "figure_data/alfworld-task-family-map.json",
            ],
            "webshop-paired-reward-waterfalls": [
                "figure_data/cross-model-analysis-data.json",
                "task_level/run-catalog.json",
            ],
            "cross-model-success-set-agreement": [
                "figure_data/cross-model-analysis-data.json",
                "task_level/run-catalog.json",
            ],
        },
    }


def write_readme(output: Path, run_count: int, row_count: int) -> None:
    content = f"""# Trace2Tower 实验原始数据包

生成日期：2026-07-21

本目录包含论文主实验相关的机器可读数据，共 {run_count} 个正式运行、
{row_count:,} 条逐任务结果。下文给出了每张表、每组图与原始运行的对应关系。

所有文件平铺在本目录。`*.jsonl` 是逐任务结果，另外只有 ALFWorld Tower 快照和
任务族映射。

逐任务文件按 `sample_id` 对齐。ALFWorld Tower 只保留主实验结果，不包含补充探索结果。

## 实验对应关系

### ALFWorld 主表

| 方法 | 逐任务文件 |
|---|---|
| No-Skill | `alfworld-main-noskill.jsonl` |
| Expert-Crafted Skills | `alfworld-main-expert-crafted.jsonl` |
| Trace2Skill +Combined | `alfworld-main-trace2skill-combined.jsonl` |
| Trace2Skill +Error | `alfworld-main-trace2skill-error.jsonl` |
| SkillX no-rewrite | `alfworld-main-skillx-no-rewrite.jsonl` |
| ExpeL | `alfworld-main-expel.jsonl` |
| Trace2Tower High-only | `alfworld-main-trace2tower-high-only.jsonl` |

### WebShop 主表

| 方法 | 逐任务文件 |
|---|---|
| No-Skill | `webshop-main-noskill.jsonl` |
| SkillX P100 | `webshop-main-skillx.jsonl` |
| ExpeL P100 | `webshop-main-expel.jsonl` |
| Trace2Skill +Combined | `webshop-main-trace2skill-combined.jsonl` |
| Trace2Skill +Error | `webshop-main-trace2skill-error.jsonl` |
| Expert-Crafted Skills | `webshop-main-expert-crafted.jsonl` |
| Trace2Tower P100 | `webshop-main-trace2tower.jsonl` |

### ALFWorld 结构 case study

- Full Mid：`alfworld-full-mid-case-study.jsonl`。该单次运行不进入主表。
- No Transition：`alfworld-ablation-no-transition.jsonl`。
- No Outcome：`alfworld-ablation-no-outcome.jsonl`。
- No Contrastive：`alfworld-ablation-no-contrastive.jsonl`。
- Semantic-only 没有形成合规 High，因此没有在线结果。

### 其他数据

- Tower 完整快照：`alfworld-main-trace2tower-snapshot.json`。
- ALFWorld 任务族标签：`alfworld-task-family-map.json`。
- 跨模型逐任务结果：`alfworld-cross-*`、`webshop-cross-*` 以及 Flash 主运行 JSONL。
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def write_manifest(output: Path) -> dict:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "MANIFEST.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "bundle": output.name,
        "created_on": "2026-07-21",
        "file_count": len(files),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
        "files": files,
    }
    write_json(output / "MANIFEST.json", manifest)
    return manifest


def make_zip(output: Path) -> Path:
    destination = output.with_suffix(".zip")
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            archive.write(path, Path(output.name) / path.relative_to(output))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-zip", action="store_true")
    options = parser.parse_args()

    output = options.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    output.mkdir(parents=True)

    loaded = export_runs(output)
    export_task_family_map(output)
    copy_sources(output)
    row_count = sum(len(rows) for rows in loaded.values())
    write_readme(output, len(loaded), row_count)
    archive = None if options.no_zip else make_zip(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "archive": str(archive) if archive else None,
                "run_count": len(loaded),
                "row_count": row_count,
                "file_count": sum(1 for path in output.iterdir() if path.is_file()),
                "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
