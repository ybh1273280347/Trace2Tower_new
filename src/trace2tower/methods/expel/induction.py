from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from trace2tower.core.manifests import Benchmark
from trace2tower.core.trajectory import EpisodeTrajectory


class RuleOperationName(StrEnum):
    REMOVE = "REMOVE"
    AGREE = "AGREE"
    EDIT = "EDIT"
    ADD = "ADD"


@dataclass(frozen=True, slots=True)
class RuleOperation:
    name: RuleOperationName
    rule_number: int | None
    text: str


def parse_rule_operations(output: str) -> tuple[RuleOperation, ...]:
    operations = []
    pattern = re.compile(r"^(AGREE|REMOVE|EDIT|ADD)(?:\s+(\d+))?:\s*(.+)$")
    for raw_line in output.splitlines():
        match = pattern.match(raw_line.strip())
        if not match:
            continue
        text = match.group(3).strip()
        if text and text.endswith("."):
            operations.append(
                RuleOperation(
                    RuleOperationName(match.group(1)),
                    int(match.group(2)) if match.group(2) else None,
                    text,
                )
            )
    return tuple(operations[:4])


def apply_rule_operations(
    weighted_rules: tuple[tuple[str, int], ...],
    operations: tuple[RuleOperation, ...],
) -> tuple[tuple[str, int], ...]:
    rules = list(weighted_rules)
    for operation_name in RuleOperationName:
        for operation in operations:
            if operation.name is not operation_name:
                continue
            if operation.name is RuleOperationName.ADD:
                if operation.text not in {rule for rule, _ in rules}:
                    rules.append((operation.text, 2))
                continue
            if operation.rule_number is None or not 1 <= operation.rule_number <= len(rules):
                continue
            index = operation.rule_number - 1
            text, weight = rules[index]
            if operation.name is RuleOperationName.REMOVE:
                rules[index] = (text, weight - 1)
            elif operation.name is RuleOperationName.AGREE:
                rules[index] = (text, weight + 1)
            elif operation.name is RuleOperationName.EDIT:
                rules[index] = (operation.text, weight + 1)
    return tuple(sorted((rule for rule in rules if rule[1] > 0), key=lambda item: -item[1]))


def comparison_messages(
    benchmark: Benchmark,
    task_goal: str,
    success_history: str,
    failure_history: str,
    rules: tuple[str, ...],
) -> list[dict[str, str]]:
    environment = "household environment" if benchmark is Benchmark.ALFWORLD else "online store"
    return [
        {
            "role": "system",
            "content": (
                "You are an advanced reasoning agent that updates a reusable rule set by "
                f"contrasting successful and unsuccessful trials in an {environment}."
            ),
        },
        {
            "role": "user",
            "content": _operation_prompt(
                "Here are two previous trials to compare and critique:\n"
                f"TRIAL TASK:\n{task_goal}\n\n"
                f"SUCCESSFUL TRIAL:\n{success_history}\n\n"
                f"FAILED TRIAL:\n{failure_history}",
                rules,
            ),
        },
    ]


def success_messages(
    benchmark: Benchmark,
    success_histories: tuple[str, ...],
    rules: tuple[str, ...],
) -> list[dict[str, str]]:
    environment = "household environment" if benchmark is Benchmark.ALFWORLD else "online store"
    return [
        {
            "role": "system",
            "content": (
                "You are an advanced reasoning agent that extracts reusable high-level rules "
                f"from successful trials in an {environment}."
            ),
        },
        {
            "role": "user",
            "content": _operation_prompt(
                "Here are the successful trials:\n\n" + "\n\n".join(success_histories),
                rules,
            ),
        },
    ]


def render_trajectory(trajectory: EpisodeTrajectory, *, max_chars: int) -> str:
    lines = [f"Task: {trajectory.task_goal}"]
    if trajectory.steps:
        lines.append(f"Initial observation: {_compact(trajectory.steps[0].observation, 1800)}")
    for step in trajectory.steps:
        arguments = json.dumps(step.action_arguments or {}, ensure_ascii=False, sort_keys=True)
        lines.append(f"Action: {step.action_name} {arguments}")
        lines.append(f"Observation: {_compact(step.next_observation, 1800)}")
        if sum(len(line) + 1 for line in lines) >= max_chars:
            lines.append("[Trajectory truncated to the frozen ExpeL context budget]")
            break
    return "\n".join(lines)[:max_chars]


def _operation_prompt(evidence: str, rules: tuple[str, ...]) -> str:
    existing = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, 1)) or "(none)"
    return f"""{evidence}

Here are the EXISTING RULES:
{existing}

Update the rules so they are general, concise, and useful on future tasks. Do not mention the
specific trials. Use at most four operations, one per line, in exactly one of these formats:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Every rule must end with a period. Prefer editing or removing redundant rules once the list is
substantial, and add only insights that differ materially from the existing rules."""


def _compact(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= max_chars else normalized[: max_chars - 3] + "..."
