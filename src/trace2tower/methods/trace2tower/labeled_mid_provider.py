from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from trace2tower.agent import SkillSelection
from trace2tower.benchmarks.models import EnvironmentState
from trace2tower.llm_runtime import CommonLLMRuntime
from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.models import PrimitiveAction
from trace2tower.methods.trace2tower.provider import Trace2TowerSkillProvider
from trace2tower.methods.trace2tower.retrieval import format_tower_context


TRANSFORMATION_ACTION_PREFIXES = {
    PrimitiveAction.CLEAN: "clean",
    PrimitiveAction.COOL: "cool",
    PrimitiveAction.HEAT: "heat",
}


@dataclass(frozen=True, slots=True)
class LabeledMidTask:
    task_goal: str
    target_action_name: str
    destination_action_name: str
    transformation_mid_id: str

    @classmethod
    def from_record(cls, record: dict) -> LabeledMidTask:
        return cls(
            task_goal=str(record["task_goal"]),
            target_action_name=str(record["target_action_name"]).casefold(),
            destination_action_name=str(
                record["destination_action_name"]
            ).casefold(),
            transformation_mid_id=str(record["transformation_mid_id"]),
        )


class LabeledMidDiagnosticProvider:
    """用人工阶段标签诊断现有 Mid 卡本身是否具有执行价值。"""

    def __init__(
        self,
        base_provider: Trace2TowerSkillProvider,
        labels: tuple[LabeledMidTask, ...],
        *,
        manipulation_mid_id: str,
    ):
        if base_provider.snapshot.benchmark is not Benchmark.ALFWORLD:
            raise ValueError("labeled Mid diagnostic supports ALFWorld only")
        labels_by_goal = {label.task_goal: label for label in labels}
        if not labels_by_goal or len(labels_by_goal) != len(labels):
            raise ValueError("labeled Mid diagnostic requires unique task goals")
        referenced_mid_ids = {
            manipulation_mid_id,
            *(label.transformation_mid_id for label in labels),
        }
        if not referenced_mid_ids <= set(base_provider.mid_cards):
            raise ValueError("labeled Mid diagnostic references unknown Mid cards")
        for label in labels:
            card = base_provider.mid_cards[label.transformation_mid_id]
            actions = set(card.grounding_actions)
            if len(actions & set(TRANSFORMATION_ACTION_PREFIXES)) != 1:
                raise ValueError(
                    "each labeled transformation Mid must bind one transformation action"
                )

        self.base_provider = base_provider
        self.labels_by_goal = labels_by_goal
        self.manipulation_mid_id = manipulation_mid_id
        self.transformed_goals: set[str] = set()

    @classmethod
    def from_path(
        cls,
        runtime: CommonLLMRuntime,
        snapshot_path: Path,
        *,
        graph_profile_path: Path,
        labels_path: Path,
        manipulation_mid_id: str,
        high_similarity_threshold: float = -1.0,
        min_event_compatibility: float = 0.1,
    ) -> LabeledMidDiagnosticProvider:
        base_provider = Trace2TowerSkillProvider.from_path(
            runtime,
            snapshot_path,
            graph_profile_path=graph_profile_path,
            mid_context_budget=0,
            low_top_k=0,
            high_similarity_threshold=high_similarity_threshold,
            min_event_compatibility=min_event_compatibility,
        )
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = tuple(
            LabeledMidTask.from_record(record) for record in payload["tasks"]
        )
        return cls(
            base_provider,
            labels,
            manipulation_mid_id=manipulation_mid_id,
        )

    async def select_task(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        if task_goal not in self.labels_by_goal:
            raise ValueError("labeled Mid diagnostic received an unlabeled task")
        # 诊断集每个目标只运行一次；任务开始时清除上一轮同目标的阶段状态。
        self.transformed_goals.discard(task_goal)
        return await self.base_provider.select_task(task_goal, state)

    async def select_state(
        self,
        task_goal: str,
        state: EnvironmentState,
    ) -> SkillSelection:
        label = self.labels_by_goal[task_goal]
        actions = tuple(action.casefold() for action in state.admissible_actions)
        observation = state.observation.casefold()
        target = re.escape(label.target_action_name)
        transformation_card = self.base_provider.mid_cards[
            label.transformation_mid_id
        ]
        transformation_action = next(
            action
            for action in transformation_card.grounding_actions
            if action in TRANSFORMATION_ACTION_PREFIXES
        )
        transformation_prefix = TRANSFORMATION_ACTION_PREFIXES[
            transformation_action
        ]

        if re.search(
            rf"\byou {transformation_prefix}(?:ed)? (?:the )?{target}(?:\s+\d+)?\b",
            observation,
        ):
            self.transformed_goals.add(task_goal)

        target_pick_available = any(
            action.startswith(f"take {label.target_action_name} ")
            for action in actions
        )
        target_place_available = any(
            action.startswith(f"move {label.target_action_name} ")
            and f" to {label.destination_action_name}" in action
            for action in actions
        )
        holding_target = any(
            action.startswith(f"move {label.target_action_name} ")
            for action in actions
        )
        closed_receptacle = " is closed." in observation and any(
            action.startswith("open ") for action in actions
        )

        selected_mid_id = None
        if task_goal not in self.transformed_goals and holding_target:
            selected_mid_id = label.transformation_mid_id
        elif target_pick_available or target_place_available or closed_receptacle:
            selected_mid_id = self.manipulation_mid_id

        if selected_mid_id is None:
            return SkillSelection((), "")
        card = self.base_provider.mid_cards[selected_mid_id]
        return SkillSelection(
            (selected_mid_id,),
            format_tower_context(None, (card,)),
            0,
            0,
            (selected_mid_id,),
        )
