from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from trace2tower.llm_runtime import CommonLLMRuntime, ModelRole


SKILLX_COMMIT = "36747f424a17ea041e476adf2ff976a206ec9c30"
_SKILLX_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "SkillX-native-36747f4"


@dataclass(frozen=True, slots=True)
class NativeSkillCandidate:
    skill_id: str
    name: str
    document: str
    content: str


@dataclass(frozen=True, slots=True)
class NativeRewriteResult:
    plan: str | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class NativeSelectionResult:
    skills: tuple[NativeSkillCandidate, ...]
    input_tokens: int | None
    output_tokens: int | None


class NativeSkillXInference:
    """Frozen SkillX rewrite and selector behavior over project-owned retrieval."""

    def __init__(
        self,
        runtime: CommonLLMRuntime,
        *,
        max_output_tokens: int,
        max_attempts: int = 3,
    ):
        if max_output_tokens <= 0 or max_attempts <= 0:
            raise ValueError("SkillX inference limits must be positive")
        self.runtime = runtime
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts

    async def rewrite_plan(
        self,
        *,
        task: str,
        reference_task: str,
        reference_plan: str,
    ) -> NativeRewriteResult:
        reference_tasks = (
            f"Task-1 {reference_task}:\n"
            f"Reference plan:\n{reference_plan}\n\n"
        )
        messages = [
            {"role": "system", "content": PLAN_REWRITE_PROMPT},
            {
                "role": "user",
                "content": f"#Reference Tasks:\n{reference_tasks}\n\n# New Task: {task}",
            },
        ]
        input_tokens: list[int | None] = []
        output_tokens: list[int | None] = []
        for _ in range(self.max_attempts):
            try:
                result = await self.runtime.chat(
                    ModelRole.RENDERER,
                    messages,
                    temperature=0.0,
                    max_output_tokens=self.max_output_tokens,
                    prompt_cache_key="skillx-native:plan-rewrite:36747f4",
                )
            except Exception:
                continue
            input_tokens.append(result.usage.input_tokens)
            output_tokens.append(result.usage.output_tokens)
            plan = _extract_plan(result.content or "")
            if plan:
                return NativeRewriteResult(
                    plan,
                    _sum_tokens(input_tokens),
                    _sum_tokens(output_tokens),
                )
        return NativeRewriteResult(
            None,
            _sum_tokens(input_tokens),
            _sum_tokens(output_tokens),
        )

    async def select_skills(
        self,
        *,
        task: str,
        plan: str,
        skills: Sequence[NativeSkillCandidate],
        max_skills: int,
    ) -> NativeSelectionResult:
        candidates = tuple(skills)
        if max_skills <= 0:
            raise ValueError("maximum SkillX skills must be positive")
        if len(candidates) <= max_skills:
            return NativeSelectionResult(candidates, 0, 0)

        descriptions = str(
            [
                {
                    "skill_name": skill.name,
                    "skill_description": skill.document[:200],
                }
                for skill in candidates
            ]
        )
        prompt = SELECT_SKILL_PROMPT.format(
            user_task=task,
            plan=plan,
            skill_library=descriptions,
        )
        input_tokens: list[int | None] = []
        output_tokens: list[int | None] = []
        for _ in range(self.max_attempts):
            try:
                result = await self.runtime.chat(
                    ModelRole.RENDERER,
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_output_tokens=self.max_output_tokens,
                    prompt_cache_key="skillx-native:skill-selector:36747f4",
                )
            except Exception:
                continue
            input_tokens.append(result.usage.input_tokens)
            output_tokens.append(result.usage.output_tokens)
            selected_names = _extract_skill_names(result.content or "")
            if selected_names:
                selected = tuple(
                    skill for skill in candidates if skill.name in selected_names
                )[:max_skills]
                return NativeSelectionResult(
                    selected,
                    _sum_tokens(input_tokens),
                    _sum_tokens(output_tokens),
                )
        return NativeSelectionResult(
            candidates[:max_skills],
            _sum_tokens(input_tokens),
            _sum_tokens(output_tokens),
        )


def format_native_context(
    plan: str | None,
    skills: Sequence[NativeSkillCandidate],
) -> str:
    sections = []
    if skills:
        lines = [
            "# Skill Library",
            "The following skills provide guidance on how to accomplish common tasks:",
            "",
        ]
        for index, skill in enumerate(skills, 1):
            lines.extend(
                (
                    f"# Skill {index}: {skill.name}",
                    f"\nDescription:\n{skill.document}",
                    f"\nContent:\n{skill.content}",
                    "",
                )
            )
        lines.append("Note: Skills are for reference only. Use the actual tools for execution.")
        sections.append("\n".join(lines))
    if plan:
        sections.append(
            f"# Reference Plan\n{plan}\n\nNote: Adapt the plan to the specific task."
        )
    return "\n\n".join(sections)


def _extract_plan(text: str) -> str | None:
    match = re.search(r"<plan>(.*?)</plan>", text, flags=re.S)
    return match.group(1).strip() if match else None


def _extract_skill_names(text: str) -> tuple[str, ...]:
    match = re.search(r"```python\s*(.*?)\s*```", text, flags=re.S)
    candidates = (match.group(1).strip(),) if match else ()
    bracket_match = re.search(r"\[(.*?)\]", text, flags=re.S)
    if bracket_match:
        candidates += (f"[{bracket_match.group(1)}]",)
    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    return ()


def _load_assignment(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(f"frozen SkillX source is missing: {path}")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise RuntimeError(f"frozen SkillX assignment is missing: {name}")


PLAN_REWRITE_PROMPT = _load_assignment(
    _SKILLX_ROOT / "prompts" / "plan_prompts.py",
    "PLAN_REWRITE_PROMPTS",
)["default"]
SELECT_SKILL_PROMPT = _load_assignment(
    _SKILLX_ROOT / "inference" / "skill_selector.py",
    "SELECT_SKILL_PROMPT",
)


def _sum_tokens(counts: Sequence[int | None]) -> int | None:
    return None if any(count is None for count in counts) else sum(counts)
