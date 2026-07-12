"""Crash-safe JSONL checkpoints for episode execution."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    benchmark: str
    split: str
    method: str
    sample_id: str
    repeat_id: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> EpisodeKey:
        try:
            return cls(
                benchmark=str(record["benchmark"]),
                split=str(record["split"]),
                method=str(record["method"]),
                sample_id=str(record["sample_id"]),
                repeat_id=int(record["repeat_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("episode record has an invalid checkpoint key") from exc


class EpisodeCheckpoint:
    """Persist official results while leaving failed attempts eligible for rerun."""

    def __init__(self, results_path: Path, errors_path: Path):
        self.results_path = results_path
        self.errors_path = errors_path
        self._repair_trailing_partial_line(results_path)
        self._repair_trailing_partial_line(errors_path)
        self._completed = self._load_completed_keys()

    def pending(self, episode_keys: Iterable[EpisodeKey]) -> list[EpisodeKey]:
        return [key for key in episode_keys if key not in self._completed]

    def write_result(self, record: Mapping[str, Any]) -> bool:
        key = EpisodeKey.from_record(record)
        if record.get("primary_score") is None or record.get("error") is not None:
            raise ValueError("only official episode results can complete a checkpoint")
        if key in self._completed:
            return False

        self._append_json(self.results_path, dict(record))
        self._completed.add(key)
        return True

    def write_error(self, key: EpisodeKey, error: str) -> None:
        self._append_json(self.errors_path, {**asdict(key), "error": error})

    def _load_completed_keys(self) -> set[EpisodeKey]:
        if not self.results_path.exists():
            return set()

        completed: set[EpisodeKey] = set()
        with self.results_path.open(encoding="utf-8") as result_file:
            for line_number, line in enumerate(result_file, start=1):
                try:
                    record = json.loads(line)
                    key = EpisodeKey.from_record(record)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid checkpoint record at {self.results_path}:{line_number}"
                    ) from exc
                if key in completed:
                    location = f"{self.results_path}:{line_number}"
                    raise ValueError(f"duplicate checkpoint key at {location}")
                completed.add(key)
        return completed

    @staticmethod
    def _append_json(path: Path, record: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as output_file:
            output_file.write(encoded)
            output_file.flush()
            os.fsync(output_file.fileno())

    @staticmethod
    def _repair_trailing_partial_line(path: Path) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return

        with path.open("r+b") as checkpoint_file:
            checkpoint_file.seek(-1, os.SEEK_END)
            if checkpoint_file.read(1) == b"\n":
                return

            checkpoint_file.seek(0)
            content = checkpoint_file.read()
            last_complete_line = content.rfind(b"\n")
            checkpoint_file.truncate(last_complete_line + 1)
