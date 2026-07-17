from __future__ import annotations

from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "trace2tower"
TRACE2TOWER_ROOT = SOURCE_ROOT / "methods" / "trace2tower"


def test_package_roots_only_expose_package_entry_points() -> None:
    assert {path.name for path in SOURCE_ROOT.glob("*.py")} == {"__init__.py"}
    assert {path.name for path in TRACE2TOWER_ROOT.glob("*.py")} == {"__init__.py"}
    assert not (SOURCE_ROOT / "methods" / "global_e2e").exists()
    assert not (SOURCE_ROOT / "methods" / "manual_skill").exists()


def test_core_trace2tower_stages_do_not_depend_on_benchmark_adapters() -> None:
    core_directories = ("core", "preprocessing", "eigen_trace", "induction")
    for directory in core_directories:
        for path in (TRACE2TOWER_ROOT / directory).glob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "trace2tower.methods.trace2tower.adapters" not in source
