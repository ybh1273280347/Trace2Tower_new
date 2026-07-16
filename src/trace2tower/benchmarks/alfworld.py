from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as parquet

from trace2tower.benchmarks.models import EnvironmentState, EpisodeStart
from trace2tower.manifests import Benchmark, ManifestEntry


_TASK_GOAL_RE = re.compile(r"(?im)^Your task is to:\s*(.+)$")


def parse_alfworld_observation_goal(observation: str) -> str:
    match = _TASK_GOAL_RE.search(observation)
    return " ".join(match.group(1).split()) if match else ""


class AlfworldEnvironment:
    benchmark = Benchmark.ALFWORLD
    tool_schemas = (
        {
            "type": "function",
            "function": {
                "name": "take_action",
                "description": "Execute one action from the available action list.",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        },
    )

    def __init__(self, dataset_root: Path, server_url: str):
        self.dataset_root = dataset_root
        self.server_url = server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60, trust_env=False)
        self.session_id: str | None = None
        self.current_state: EnvironmentState | None = None

    async def reset(self, entry: ManifestEntry) -> EpisodeStart:
        table = parquet.read_table(
            self.dataset_root / f"{entry.source_split}.parquet",
            columns=["extra_info"],
        )
        item = table.column("extra_info").combine_chunks()[entry.dataset_index].as_py()
        response = await self.client.post(
            f"{self.server_url}/reset",
            json={"game_relative_path": item["game_relative_path"]},
        )
        response.raise_for_status()
        payload = response.json()
        self.session_id = payload["session_id"]
        self.current_state = self._state(payload, True)
        task_goal = parse_alfworld_observation_goal(payload["observation"])
        return EpisodeStart(
            task_goal=task_goal or item["goal_text"],
            state=self.current_state,
        )

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> EnvironmentState:
        if tool_name != "take_action" or not isinstance(arguments.get("action"), str):
            return EnvironmentState("Invalid action.", (), {}, False, 0, False, False)
        if self.session_id is None:
            raise RuntimeError("ALFWorld environment has not been reset")
        valid_action = bool(
            self.current_state and arguments["action"] in self.current_state.admissible_actions
        )
        response = await self.client.post(
            f"{self.server_url}/step",
            json={"session_id": self.session_id, "action": arguments["action"]},
        )
        response.raise_for_status()
        self.current_state = self._state(response.json(), valid_action)
        return self.current_state

    async def close(self) -> None:
        if self.session_id is not None:
            await self.client.post(
                f"{self.server_url}/close", json={"session_id": self.session_id}
            )
            self.session_id = None
            self.current_state = None
        await self.client.aclose()

    @staticmethod
    def _state(payload: dict[str, Any], valid_action: bool) -> EnvironmentState:
        return EnvironmentState(
            observation=payload["observation"],
            admissible_actions=tuple(payload["admissible_actions"]),
            clickable_types={},
            search_available=False,
            reward=float(payload["won"]),
            done=bool(payload["done"]),
            valid_action=valid_action,
        )
