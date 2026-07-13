from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import yaml
from rollout_no_skill_train import write_json

EXPECTED_COMMIT = "36747f424a17ea041e476adf2ff976a206ec9c30"
PROTECTED_FILES = (
    "pipeline.py",
    "prompts/registry.py",
    "prompts/skill_prompts.py",
    "prompts/plan_prompts.py",
    "prompts/merge_prompts.py",
    "prompts/filter_prompts.py",
    "extraction/base.py",
    "extraction/skill_extractor.py",
    "extraction/plan_extractor.py",
    "extraction/tool_summary.py",
    "clustering/dbscan.py",
    "clustering/embedding.py",
    "clustering/merger.py",
    "filtering/base.py",
    "filtering/pipeline.py",
    "filtering/general_filter.py",
    "filtering/tool_filter.py",
    "inference/retriever.py",
    "inference/embedding_service.py",
)


def git(repository: Path, *arguments: str, text: bool = True):
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        text=text,
    )
    return result.stdout.strip() if text else result.stdout


def inspect_skillx(repository: Path) -> dict:
    commit = git(repository, "rev-parse", "HEAD")
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"SkillX commit changed: {commit}")
    changed = git(repository, "diff", "--name-only", "HEAD", "--", *PROTECTED_FILES)
    if changed:
        raise ValueError(f"protected SkillX files changed: {changed.splitlines()}")
    hashes = {
        path: hashlib.sha256(
            git(repository, "show", f"HEAD:{path}", text=False)
        ).hexdigest()
        for path in PROTECTED_FILES
    }
    return {
        "commit": commit,
        "protected_file_count": len(PROTECTED_FILES),
        "protected_files": hashes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository", type=Path, default=Path("third_party/SkillX")
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = inspect_skillx(options.repository)
    if options.output:
        write_json(options.output, report)
    print(yaml.safe_dump(report, sort_keys=False))
