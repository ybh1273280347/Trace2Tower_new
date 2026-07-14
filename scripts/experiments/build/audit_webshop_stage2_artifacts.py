from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.experiments.run.rollout_no_skill_train import load_yaml, write_json
from trace2tower.manifests import Benchmark
from trace2tower.methods.global_e2e.models import GlobalE2ESkillLibrary
from trace2tower.methods.global_e2e.renderer import format_trajectory_corpus
from trace2tower.methods.skillx.models import SkillXExecutionLibrary
from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.tower import TowerSnapshot
from trace2tower.results import MethodName
from trace2tower.trajectory import TrajectoryReader


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_blob(revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_global_e2e(
    artifact_dir: Path,
    successful_trajectories: tuple,
) -> dict:
    library_path = artifact_dir / "library.json"
    report_path = artifact_dir / "report.json"
    induction_path = artifact_dir / "induction.json"
    library = GlobalE2ESkillLibrary.from_record(read_json(library_path))
    report = read_json(report_path)
    induction = read_json(induction_path)
    corpus = format_trajectory_corpus(successful_trajectories)
    corpus_sha256 = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    legacy_prompt_revision = "db29a80"
    legacy_prompt_path = "src/trace2tower/methods/flat_skill_summary/end_to_end_prompt.py"
    legacy_prompt_sha256 = hashlib.sha256(
        git_blob(legacy_prompt_revision, legacy_prompt_path)
    ).hexdigest()
    expected_trajectory_ids = tuple(
        trajectory.trajectory_id for trajectory in successful_trajectories
    )
    if (
        library.benchmark is not Benchmark.WEBSHOP
        or library.training_trajectory_ids != expected_trajectory_ids
        or library.corpus_sha256 != corpus_sha256
        or report["corpus_sha256"] != corpus_sha256
        or report.get("induction_mode") != "end_to_end"
        or report["renderer_model"] != "gpt-5.4"
        or library.prompt_sha256 != legacy_prompt_sha256
        or induction["prompt_sha256"] != legacy_prompt_sha256
        or induction["corpus_sha256"] != corpus_sha256
        or induction["renderer_model"] != "gpt-5.4"
    ):
        raise ValueError("Global E2E artifact does not match its frozen provenance")

    return {
        "method": MethodName.GLOBAL_E2E_GPT.value,
        "reuse_policy": "stable_end_to_end_induction",
        "library": library_path.as_posix(),
        "library_id": library.library_id,
        "library_sha256": sha256_file(library_path),
        "skill_count": len(library.cards),
        "training_trajectory_count": len(library.training_trajectory_ids),
        "corpus_sha256": corpus_sha256,
        "legacy_prompt_revision": legacy_prompt_revision,
        "legacy_prompt_path": legacy_prompt_path,
        "legacy_prompt_sha256": legacy_prompt_sha256,
        "builder_chat_input_tokens": report["builder_chat_input_tokens"],
        "builder_chat_output_tokens": report["builder_chat_output_tokens"],
    }


def audit_skillx(
    artifact_dir: Path,
    source_library_path: Path,
    upstream_report_path: Path,
    runtime_config_path: Path,
    successful_trajectory_ids: tuple[str, ...],
) -> dict:
    library_path = artifact_dir / "library.json"
    report_path = artifact_dir / "report.json"
    library = SkillXExecutionLibrary.from_record(read_json(library_path))
    report = read_json(report_path)
    upstream_report = read_json(upstream_report_path)
    runtime_config = load_yaml(runtime_config_path)
    source_library_sha256 = sha256_file(source_library_path)
    skillx_head = subprocess.run(
        ["git", "-C", "third_party/SkillX", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        library.benchmark is not Benchmark.WEBSHOP
        or tuple(upstream_report["source_trajectory_ids"]) != successful_trajectory_ids
        or upstream_report["source_trajectory_count"] != len(successful_trajectory_ids)
        or library.source_library_sha256 != source_library_sha256
        or library.skillx_commit != skillx_head
        or report["skillx_commit"] != skillx_head
        or report["source_library_sha256"] != source_library_sha256
        or runtime_config["method"] != MethodName.SKILLX.value
        or runtime_config["max_skills"] != 8
    ):
        raise ValueError("SkillX artifact does not match its frozen provenance")

    return {
        "method": MethodName.SKILLX.value,
        "reuse_policy": "stable_official_skillx_library",
        "library": library_path.as_posix(),
        "library_id": library.library_id,
        "library_sha256": sha256_file(library_path),
        "source_library": source_library_path.as_posix(),
        "source_library_sha256": source_library_sha256,
        "upstream_reported_library_sha256": upstream_report["library_sha256"],
        "skillx_commit": skillx_head,
        "training_trajectory_count": len(successful_trajectory_ids),
        "plan_count": len(library.plans),
        "skill_count": len(library.skills),
        "runtime_max_skills": 8,
    }


def audit_tower(
    name: str,
    root: Path,
    method: MethodName,
    config_path: Path,
    preprocessed_path: Path,
) -> dict:
    snapshot_path = root / "tower.json"
    graph_dir = root / "graph"
    skills_dir = root / "skills"
    component_paths = {
        "preprocessed_trajectories": preprocessed_path,
        "clusters": graph_dir / "clusters.json",
        "high_paths": skills_dir / "high-paths.json",
        "rendered_cards": skills_dir / "rendered-cards.json",
        "retrieval_index": root / "index.json",
    }
    snapshot = TowerSnapshot.from_record(read_json(snapshot_path))
    snapshot.require_complete()
    config = Trace2TowerConfig.from_record(load_yaml(config_path))
    preprocessed_ids = tuple(
        json.loads(line)["trajectory_id"]
        for line in preprocessed_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    component_hashes = {field: sha256_file(path) for field, path in component_paths.items()}
    snapshot_hashes = {field: getattr(snapshot.source_hashes, field) for field in component_paths}
    if (
        snapshot.benchmark is not Benchmark.WEBSHOP
        or snapshot.config != config
        or snapshot.config.method is not method
        or snapshot.training_trajectory_ids != preprocessed_ids
        or snapshot_hashes != component_hashes
    ):
        raise ValueError(f"{name} snapshot does not match its frozen components")
    if method is MethodName.SEMANTIC_CLUSTERING and snapshot.high_cards:
        raise ValueError("Semantic Clustering baseline must not contain High cards")

    rendered = read_json(skills_dir / "rendered-cards.json")
    usages = rendered.get("usage", ())
    return {
        "name": name,
        "method": method.value,
        "snapshot": snapshot_path.as_posix(),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": sha256_file(snapshot_path),
        "config": config_path.as_posix(),
        "config_sha256": sha256_file(config_path),
        "training_trajectory_count": len(snapshot.training_trajectory_ids),
        "mid_count": len(snapshot.mid_cards),
        "high_count": len(snapshot.high_cards),
        "component_paths": {field: path.as_posix() for field, path in component_paths.items()},
        "component_sha256": component_hashes,
        "renderer_call_count": len(usages),
        "renderer_input_tokens": sum(item["input_tokens"] or 0 for item in usages),
        "renderer_output_tokens": sum(item["output_tokens"] or 0 for item in usages),
    }


def render_report(audit: dict) -> str:
    global_e2e = audit["baselines"]["global_e2e_gpt"]
    skillx = audit["baselines"]["skillx"]
    full = audit["snapshots"]["trace2tower"]
    semantic = audit["snapshots"]["semantic_clustering"]
    no_mixed = audit["snapshots"]["trace2tower_no_mixed"]
    global_row = (
        f"| Global E2E GPT | `{global_e2e['library_id']}` | "
        f"{global_e2e['training_trajectory_count']} | "
        f"{global_e2e['skill_count']} cards |"
    )
    skillx_row = (
        f"| SkillX | `{skillx['library_id']}` | "
        f"{skillx['training_trajectory_count']} | {skillx['plan_count']} plans + "
        f"{skillx['skill_count']} atomic skills |"
    )
    global_provenance = (
        "Global E2E 复用的是已完成的 GPT-5.4 end-to-end induction。审计把旧 "
        f"prompt blob 固定到 git revision `{global_e2e['legacy_prompt_revision']}`，"
        "并验证 prompt SHA、P50 的 94 条成功轨迹、corpus SHA、卡片和 embedding "
        "index。它作为 `global_e2e_gpt` 执行，不恢复旧方法名。"
    )
    skillx_provenance = (
        "SkillX 复用官方上游构建结果，绑定 commit "
        f"`{skillx['skillx_commit']}`、94 条成功轨迹和 source library SHA；"
        "运行时固定原生 `max_skills=8`。"
    )
    full_row = (
        f"| Full Trace2Tower | `{full['snapshot_id']}` | "
        f"{full['training_trajectory_count']} | {full['mid_count']} | "
        f"{full['high_count']} | true |"
    )
    semantic_row = (
        f"| Semantic Clustering | `{semantic['snapshot_id']}` | "
        f"{semantic['training_trajectory_count']} | {semantic['mid_count']} | "
        f"{semantic['high_count']} | false |"
    )
    no_mixed_row = (
        f"| No-mixed | `{no_mixed['snapshot_id']}` | "
        f"{no_mixed['training_trajectory_count']} | {no_mixed['mid_count']} | "
        f"{no_mixed['high_count']} | true |"
    )
    snapshot_summary = (
        "Full 与 Semantic Clustering 使用同一 173 条 mixed evidence；No-mixed "
        "只使用 94 条满分成功轨迹。Semantic Clustering 不包含 signed graph 或 "
        "High，因此只作为 baseline。No-event snapshot 延后到独立消融阶段构建。"
    )
    mid_only_summary = (
        "Mid-only 不生成新 artifact，执行时复用 Full snapshot "
        f"`{full['snapshot_id']}` 并设置 `include_high=false`。Manual Skill "
        f"SHA-256 为 `{audit['manual_skill']['sha256']}`。"
    )
    return f"""# Stage 2: P50 技能 Artifact 冻结

状态：`complete`
审计 ID：`{audit["audit_id"]}`

## Baseline artifacts

| 方法 | Artifact ID | 训练轨迹 | 内容 |
|---|---|---:|---:|
{global_row}
{skillx_row}

{global_provenance}

{skillx_provenance}

## 新构建 snapshots

| 方法 | Snapshot ID | 训练轨迹 | Mid | High | Event stratification |
|---|---|---:|---:|---:|---|
{full_row}
{semantic_row}
{no_mixed_row}

{snapshot_summary}

{mid_only_summary}

完整路径、组件哈希、配置哈希、renderer token 和 artifact SHA-256 见 `audit.json`。
"""


def main(options: argparse.Namespace) -> int:
    pool_audit = read_json(options.pool_audit)
    p50_path = Path(pool_audit["pools"]["p50"]["path"])
    if sha256_file(p50_path) != pool_audit["pools"]["p50"]["sha256"]:
        raise ValueError("P50 pool changed after stage 1")
    p50 = TrajectoryReader.read_jsonl(p50_path)
    successful = tuple(
        sorted(
            (trajectory for trajectory in p50 if trajectory.primary_score >= 0.999),
            key=lambda trajectory: trajectory.trajectory_id,
        )
    )
    successful_ids = tuple(trajectory.trajectory_id for trajectory in successful)
    mixed = TrajectoryReader.read_jsonl(options.mixed_evidence)
    success_only = TrajectoryReader.read_jsonl(options.success_evidence)
    p50_ids = {trajectory.trajectory_id for trajectory in p50}
    if (
        len(mixed) != 173
        or len(success_only) != 94
        or not {trajectory.trajectory_id for trajectory in mixed} <= p50_ids
        or tuple(sorted(trajectory.trajectory_id for trajectory in success_only)) != successful_ids
    ):
        raise ValueError("P50 evidence pools do not match the stage 1 trajectory pool")

    baselines = {
        "global_e2e_gpt": audit_global_e2e(
            options.global_e2e_dir,
            successful,
        ),
        "skillx": audit_skillx(
            options.skillx_dir,
            options.skillx_source_library,
            options.skillx_upstream_report,
            options.skillx_runtime_config,
            successful_ids,
        ),
    }
    snapshots = {
        "trace2tower": audit_tower(
            "full",
            options.tower_root / "full",
            MethodName.TRACE2TOWER,
            Path("configs/experiments/webshop_trace2tower.yaml"),
            options.tower_root / "mixed" / "preprocessed.jsonl",
        ),
        "semantic_clustering": audit_tower(
            "semantic-clustering",
            options.tower_root / "semantic-only",
            MethodName.SEMANTIC_CLUSTERING,
            Path("configs/experiments/webshop_semantic_clustering.yaml"),
            options.tower_root / "mixed" / "preprocessed.jsonl",
        ),
        "trace2tower_no_mixed": audit_tower(
            "no-mixed",
            options.tower_root / "no-mixed",
            MethodName.TRACE2TOWER_NO_MIXED,
            Path("configs/experiments/webshop_trace2tower_no_mixed.yaml"),
            options.tower_root / "success-only" / "preprocessed.jsonl",
        ),
    }
    audit = {
        "protocol_id": "webshop-event-tower-v2",
        "stage": 2,
        "status": "complete",
        "stage_1_audit_id": pool_audit["audit_id"],
        "inputs": {
            "p50_pool": p50_path.as_posix(),
            "p50_pool_sha256": sha256_file(p50_path),
            "mixed_evidence": options.mixed_evidence.as_posix(),
            "mixed_evidence_sha256": sha256_file(options.mixed_evidence),
            "mixed_trajectory_count": len(mixed),
            "success_evidence": options.success_evidence.as_posix(),
            "success_evidence_sha256": sha256_file(options.success_evidence),
            "success_trajectory_count": len(success_only),
        },
        "baselines": baselines,
        "snapshots": snapshots,
        "mid_only_binding": {
            "snapshot_id": snapshots["trace2tower"]["snapshot_id"],
            "include_high": False,
        },
        "manual_skill": {
            "path": options.manual_skill.as_posix(),
            "sha256": sha256_file(options.manual_skill),
        },
        "builder_code": {
            path.as_posix(): sha256_file(path)
            for path in (
                Path("src/trace2tower/methods/trace2tower/renderer.py"),
                Path("scripts/experiments/build/preprocess_trajectories.py"),
                Path("scripts/experiments/build/build_trace2tower_graph.py"),
                Path("scripts/experiments/build/build_trace2tower_skills.py"),
                Path("scripts/experiments/build/build_trace2tower_index.py"),
                Path("scripts/experiments/build/build_tower_snapshot.py"),
            )
        },
    }
    audit["audit_id"] = f"skillaudit_{canonical_hash(audit)[:16]}"
    write_json(options.output, audit)
    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(render_report(audit), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "audit_id": audit["audit_id"],
                "global_e2e": baselines["global_e2e_gpt"]["library_id"],
                "skillx": baselines["skillx"]["library_id"],
                "snapshots": {name: item["snapshot_id"] for name, item in snapshots.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-audit",
        type=Path,
        default=Path("experiments/webshop/event-tower-v2/stage-1-pools/audit.json"),
    )
    parser.add_argument(
        "--mixed-evidence",
        type=Path,
        default=Path(
            "artifacts/trajectories/webshop/evidence/webshop-flash50-repeat4-mixed-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--success-evidence",
        type=Path,
        default=Path(
            "artifacts/trajectories/webshop/evidence/webshop-flash50-repeat4-success-only-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--global-e2e-dir",
        type=Path,
        default=Path("artifacts/global_e2e/event-tower-v2/p50"),
    )
    parser.add_argument(
        "--skillx-dir",
        type=Path,
        default=Path("artifacts/skillx/event-tower-v2/p50/execution"),
    )
    parser.add_argument(
        "--skillx-source-library",
        type=Path,
        default=Path(
            "artifacts/skillx/webshop-success94-official-parallel2-recoverable-v6/library.json"
        ),
    )
    parser.add_argument(
        "--skillx-upstream-report",
        type=Path,
        default=Path(
            "artifacts/skillx/webshop-success94-official-parallel2-recoverable-v6/report.json"
        ),
    )
    parser.add_argument(
        "--skillx-runtime-config",
        type=Path,
        default=Path("configs/experiments/webshop_skillx.yaml"),
    )
    parser.add_argument(
        "--tower-root",
        type=Path,
        default=Path("artifacts/trace2tower/event-tower-v2/p50"),
    )
    parser.add_argument(
        "--manual-skill",
        type=Path,
        default=Path("experiments/webshop/event-tower-v2/manual-skill.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/webshop/event-tower-v2/stage-2-skills/audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/webshop/event-tower-v2/stage-2-skills/REPORT.md"),
    )
    raise SystemExit(main(parser.parse_args()))
