from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trace2tower.benchmarks.models import ClickableKind
from trace2tower.core.manifests import Benchmark, ExperimentSplit
from trace2tower.core.results import FinishReason, MethodName


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
    retrieved_skill_ids: tuple[str, ...] = ()
    retrieved_context_skill_ids: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StepRecord:
        return cls(
            step_index=int(record["step_index"]),
            observation=str(record["observation"]),
            action_name=record["action_name"],
            action_arguments=record["action_arguments"],
            next_observation=str(record["next_observation"]),
            reward=float(record["reward"]),
            done=bool(record["done"]),
            valid_action=bool(record["valid_action"]),
            admissible_actions=tuple(record["admissible_actions"]),
            clickable_types={
                value: ClickableKind(kind) for value, kind in record["clickable_types"].items()
            },
            retrieved_skill_ids=tuple(record.get("retrieved_skill_ids", ())),
            retrieved_context_skill_ids=tuple(record.get("retrieved_context_skill_ids", ())),
        )


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    run_id: str
    benchmark: Benchmark
    split: ExperimentSplit
    method: MethodName
    sample_id: str
    repeat_id: int
    task_goal: str
    steps: tuple[StepRecord, ...]
    primary_score: float
    finish_reason: FinishReason

    @property
    def trajectory_id(self) -> str:
        return f"{self.benchmark}:{self.split}:{self.method}:{self.sample_id}:{self.repeat_id}"

    def to_record(self) -> dict[str, Any]:
        return {"trajectory_id": self.trajectory_id, **asdict(self)}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EpisodeTrajectory:
        trajectory = cls(
            run_id=str(record["run_id"]),
            benchmark=Benchmark(record["benchmark"]),
            split=ExperimentSplit(record["split"]),
            method=MethodName(record["method"]),
            sample_id=str(record["sample_id"]),
            repeat_id=int(record["repeat_id"]),
            task_goal=str(record["task_goal"]),
            steps=tuple(StepRecord.from_record(step) for step in record["steps"]),
            primary_score=float(record["primary_score"]),
            finish_reason=FinishReason(record["finish_reason"]),
        )
        if record.get("trajectory_id") != trajectory.trajectory_id:
            raise ValueError("trajectory_id does not match the episode key")
        if any(step.step_index != index for index, step in enumerate(trajectory.steps)):
            raise ValueError("trajectory steps are not contiguous")
        return trajectory


class TrajectoryWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def write(self, trajectory: EpisodeTrajectory) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(trajectory.trajectory_id.encode()).hexdigest() + ".json"
        output_path = self.output_dir / filename
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=self.output_dir,
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(
                trajectory.to_record(),
                output_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        return output_path


class TrajectoryReader:
    @staticmethod
    def read_episode_files(input_dir: Path) -> tuple[EpisodeTrajectory, ...]:
        trajectories = tuple(
            EpisodeTrajectory.from_record(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(input_dir.glob("*.json"))
        )
        TrajectoryReader._validate_unique(trajectories)
        return trajectories

    @staticmethod
    def read_jsonl(path: Path) -> tuple[EpisodeTrajectory, ...]:
        trajectories = tuple(
            EpisodeTrajectory.from_record(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
        TrajectoryReader._validate_unique(trajectories)
        return trajectories

    @staticmethod
    def _validate_unique(trajectories: tuple[EpisodeTrajectory, ...]) -> None:
        trajectory_ids = [trajectory.trajectory_id for trajectory in trajectories]
        if len(trajectory_ids) != len(set(trajectory_ids)):
            raise ValueError("duplicate trajectory_id")


def materialize_trajectory_shard(input_dir: Path, output_path: Path) -> int:
    return write_trajectory_jsonl(TrajectoryReader.read_episode_files(input_dir), output_path)


def write_trajectory_jsonl(trajectories: Iterable[EpisodeTrajectory], output_path: Path) -> int:
    ordered = sorted(trajectories, key=lambda trajectory: trajectory.trajectory_id)
    TrajectoryReader._validate_unique(tuple(ordered))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=output_path.parent,
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        temporary_path = Path(output_file.name)
        for trajectory in ordered:
            json.dump(
                trajectory.to_record(),
                output_file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, output_path)
    return len(ordered)
