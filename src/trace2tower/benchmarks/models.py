from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from trace2tower.manifests import Benchmark, ManifestEntry


class ClickableKind(StrEnum):
    BUTTON = "button"
    PRODUCT_LINK = "product_link"
    OPTION = "option"


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    observation: str
    admissible_actions: tuple[str, ...]
    clickable_types: dict[str, ClickableKind]
    search_available: bool
    reward: float
    done: bool
    valid_action: bool


@dataclass(frozen=True, slots=True)
class EpisodeStart:
    task_goal: str
    state: EnvironmentState


class BenchmarkEnvironment(Protocol):
    benchmark: Benchmark
    tool_schemas: tuple[dict[str, Any], ...]

    async def reset(self, entry: ManifestEntry) -> EpisodeStart: ...

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> EnvironmentState: ...

    async def close(self) -> None: ...
