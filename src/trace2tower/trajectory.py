from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trace2tower.benchmarks.models import ClickableKind
from trace2tower.results import FinishReason, MethodName


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    observation: str
    action_name: str | None
    action_arguments: dict[str, Any] | None
    next_observation: str
    reward: float
    done: bool
    valid_action: bool
    admissible_actions: tuple[str, ...]
    clickable_types: dict[str, ClickableKind]


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    benchmark: str
    split: str
    method: MethodName
    sample_id: str
    repeat_id: int
    task_goal: str
    steps: tuple[StepRecord, ...]
    primary_score: float
    finish_reason: FinishReason


class TrajectoryWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def write(self, trajectory: EpisodeTrajectory) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        key = (
            f"{trajectory.benchmark}|{trajectory.split}|{trajectory.method}|"
            f"{trajectory.sample_id}|{trajectory.repeat_id}"
        )
        filename = hashlib.sha256(key.encode()).hexdigest() + ".json"
        output_path = self.output_dir / filename
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=self.output_dir,
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(asdict(trajectory), output_file, ensure_ascii=False, separators=(",", ":"))
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        return output_path
