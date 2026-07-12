"""Record immutable source identities without tracking vendored repositories."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SOURCE_REPOSITORIES = {
    "agentbench": Path("third_party/AgentBench"),
    "skillx": Path("third_party/SkillX"),
}


def _run_git(repository: Path, arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed in {repository}: {detail}")
    return result.stdout.strip()


def inspect_repository(repository: Path, display_path: str) -> dict[str, Any]:
    """Return the revision and dirty state required to reproduce a source checkout."""
    if not repository.is_dir():
        raise FileNotFoundError(f"source repository does not exist: {repository}")

    commit = _run_git(repository, ["rev-parse", "HEAD"])
    status = _run_git(repository, ["status", "--short", "--untracked-files=all"])
    changes = tuple(line for line in status.splitlines() if line)
    remote_result = subprocess.run(
        ["git", "-C", str(repository), "remote", "get-url", "origin"],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )

    return {
        "path": display_path,
        "commit": commit,
        "origin": remote_result.stdout.strip() if remote_result.returncode == 0 else None,
        "clean": not changes,
        "changes": list(changes),
    }


def build_source_lock(repository_root: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    source_lock = {
        name: inspect_repository(root / relative_path, relative_path.as_posix())
        for name, relative_path in SOURCE_REPOSITORIES.items()
    }
    source_lock["main_repository"] = inspect_repository(root, ".")
    source_lock["python"] = sys.version
    source_lock["platform"] = platform.platform()
    return source_lock


def write_source_lock(repository_root: Path, output: Path) -> Path:
    output_path = output if output.is_absolute() else repository_root / output
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_source_lock(repository_root)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=output_path.parent,
            encoding="utf-8",
            newline="\n",
        ) as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return output_path


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record source revisions for an experiment run.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reproducibility/source-lock.json"),
    )
    options = parser.parse_args(arguments)

    output_path = write_source_lock(options.repository_root, options.output)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
