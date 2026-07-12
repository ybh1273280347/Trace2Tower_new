from __future__ import annotations

import json
from pathlib import Path

from trace2tower.checkpoint import EpisodeCheckpoint, EpisodeKey


def result_for(key: EpisodeKey, primary_score: float) -> dict[str, object]:
    return {
        "benchmark": key.benchmark,
        "split": key.split,
        "method": key.method,
        "sample_id": key.sample_id,
        "repeat_id": key.repeat_id,
        "primary_score": primary_score,
        "error": None,
    }


def test_checkpoint_recovers_after_interrupted_append(tmp_path: Path) -> None:
    results_path = tmp_path / "episodes.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    first = EpisodeKey("webshop", "train", "no_skill", "1000", 0)
    second = EpisodeKey("webshop", "train", "no_skill", "1001", 0)
    checkpoint = EpisodeCheckpoint(results_path, errors_path)
    checkpoint.write_result(result_for(first, 0.5))

    with results_path.open("ab") as result_file:
        result_file.write(b'{"benchmark":"webshop"')

    resumed = EpisodeCheckpoint(results_path, errors_path)
    assert resumed.pending([first, second]) == [second]

    resumed.write_error(second, "temporary provider timeout")
    assert resumed.pending([second]) == [second]
    assert resumed.write_result(result_for(second, 0.75)) is True
    assert resumed.write_result(result_for(second, 0.75)) is False

    final = EpisodeCheckpoint(results_path, errors_path)
    assert final.pending([first, second]) == []
    records = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert [record["sample_id"] for record in records] == ["1000", "1001"]
    assert "temporary provider timeout" in errors_path.read_text(encoding="utf-8")

