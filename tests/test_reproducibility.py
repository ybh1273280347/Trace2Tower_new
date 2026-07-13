from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from trace2tower.reproducibility import (
    SOURCE_REPOSITORIES,
    build_source_lock,
    write_source_lock,
)


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def initialize_repository(repository: Path, remote: str) -> str:
    repository.mkdir(parents=True, exist_ok=True)
    run_git(repository, "init", "--initial-branch=main")
    run_git(repository, "config", "user.name", "Trace2Tower Test")
    run_git(repository, "config", "user.email", "test@trace2tower.local")
    run_git(repository, "remote", "add", "origin", remote)
    (repository / "source.txt").write_text("source\n", encoding="utf-8")
    run_git(repository, "add", "source.txt")
    run_git(repository, "commit", "-m", "Initialize source")
    return run_git(repository, "rev-parse", "HEAD")


@pytest.fixture
def repository_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    commits = {
        "main_repository": initialize_repository(
            tmp_path, "https://example.com/main.git"
        )
    }
    commits.update(
        {
            name: initialize_repository(
                tmp_path / relative_path,
                f"https://example.com/{name}.git",
            )
            for name, relative_path in SOURCE_REPOSITORIES.items()
        }
    )
    return tmp_path, commits


def test_build_source_lock_records_revisions_and_dirty_state(
    repository_tree: tuple[Path, dict[str, str]],
) -> None:
    repository_root, commits = repository_tree
    agentbench = repository_root / "third_party" / "AgentBench"
    (agentbench / "local-change.txt").write_text("dirty\n", encoding="utf-8")

    source_lock = build_source_lock(repository_root)

    assert source_lock["main_repository"]["commit"] == commits["main_repository"]
    assert source_lock["skillx"]["commit"] == commits["skillx"]
    assert source_lock["skillx"]["clean"] is True
    assert source_lock["agentbench"]["clean"] is False
    assert source_lock["agentbench"]["changes"] == ["?? local-change.txt"]
    assert source_lock["agentbench"]["path"] == "third_party/AgentBench"


def test_write_source_lock_uses_requested_output(
    repository_tree: tuple[Path, dict[str, str]],
) -> None:
    repository_root, _ = repository_tree
    output = Path("artifacts/reproducibility/source-lock.json")

    output_path = write_source_lock(repository_root, output)

    assert output_path == (repository_root / output).resolve()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["main_repository"]["origin"] == "https://example.com/main.git"


def test_build_source_lock_requires_both_upstream_repositories(tmp_path: Path) -> None:
    initialize_repository(tmp_path, "https://example.com/main.git")

    with pytest.raises(FileNotFoundError, match="AgentBench"):
        build_source_lock(tmp_path)
