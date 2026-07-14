from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from trace2tower.methods.global_e2e.models import GlobalE2ESkillLibrary
from trace2tower.methods.global_e2e.retrieval import format_global_e2e_card
from trace2tower.methods.skillx.models import SkillXExecutionLibrary
from trace2tower.methods.skillx.retrieval import format_retrieval
from trace2tower.methods.trace2tower.retrieval import format_tower_context
from trace2tower.methods.trace2tower.tower import TowerSnapshot


def expand_tower(
    payload: Mapping, skill_ids: Sequence[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    snapshot = TowerSnapshot.from_record(payload)
    high_cards = {card.skill_id: card for card in snapshot.high_cards}
    mid_cards = {card.skill_id: card for card in snapshot.mid_cards}
    unknown_ids = sorted(set(skill_ids) - set(high_cards) - set(mid_cards))
    if unknown_ids:
        raise ValueError(f"unknown Tower skill IDs: {unknown_ids}")

    selected_high_ids = [skill_id for skill_id in skill_ids if skill_id in high_cards]
    if len(selected_high_ids) > 1:
        raise ValueError("a Tower injection can contain at most one High skill")

    high_card = high_cards[selected_high_ids[0]] if selected_high_ids else None
    selected_mid_ids = [skill_id for skill_id in skill_ids if skill_id in mid_cards]

    selected_mid_cards = tuple(mid_cards[skill_id] for skill_id in selected_mid_ids)
    expanded_ids = ((high_card.skill_id,) if high_card else ()) + tuple(selected_mid_ids)
    referenced_child_ids = high_card.ordered_mid_ids if high_card else ()
    return (
        format_tower_context(high_card, selected_mid_cards),
        expanded_ids,
        referenced_child_ids,
    )


def expand_global_e2e(
    payload: Mapping, skill_ids: Sequence[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    library = GlobalE2ESkillLibrary.from_record(payload)
    cards = {card.skill_id: card for card in library.cards}
    unknown_ids = sorted(set(skill_ids) - set(cards))
    if unknown_ids:
        raise ValueError(f"unknown Global E2E skill IDs: {unknown_ids}")

    return (
        "\n\n".join(
            format_global_e2e_card(cards[skill_id]) for skill_id in skill_ids
        ),
        tuple(skill_ids),
        (),
    )


def expand_skillx(
    payload: Mapping, skill_ids: Sequence[str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    library = SkillXExecutionLibrary.from_record(payload)
    plans = {plan.plan_id: plan for plan in library.plans}
    skills = {skill.skill_id: skill for skill in library.skills}
    unknown_ids = sorted(set(skill_ids) - set(plans) - set(skills))
    if unknown_ids:
        raise ValueError(f"unknown SkillX IDs: {unknown_ids}")

    selected_plan_ids = [skill_id for skill_id in skill_ids if skill_id in plans]
    if len(selected_plan_ids) > 1:
        raise ValueError("a SkillX injection can contain at most one plan")

    plan = plans[selected_plan_ids[0]] if selected_plan_ids else None
    selected_skills = tuple(skills[skill_id] for skill_id in skill_ids if skill_id in skills)
    return format_retrieval(plan, selected_skills), tuple(skill_ids), ()


def artifact_kind(payload: Mapping) -> str:
    if {"snapshot_id", "mid_cards", "high_cards"} <= payload.keys():
        return "tower"
    if {"library_id", "plans", "skills", "plan_index"} <= payload.keys():
        return "skillx"
    if {"library_id", "cards", "prompt_sha256"} <= payload.keys():
        return "global_e2e"
    raise ValueError("artifact is not a recognized Tower, Global E2E, or SkillX library")


def render_markdown(
    artifact: Path,
    kind: str,
    requested_ids: Sequence[str],
    expanded_ids: Sequence[str],
    referenced_child_ids: Sequence[str],
    context: str,
) -> str:
    requested = ", ".join(f"`{skill_id}`" for skill_id in requested_ids)
    expanded = ", ".join(f"`{skill_id}`" for skill_id in expanded_ids)
    child_reference_line = ""
    if referenced_child_ids:
        children = ", ".join(f"`{skill_id}`" for skill_id in referenced_child_ids)
        child_reference_line = f"- High references these child Mid IDs: {children}\n"
    return (
        "# Skill ID Expansion\n\n"
        f"- Artifact: `{artifact.as_posix()}`\n"
        f"- Artifact type: `{kind}`\n"
        f"- Requested IDs: {requested}\n"
        f"- Expanded IDs: {expanded}\n"
        f"{child_reference_line}"
        f"- Injected context characters: `{len(context)}`\n\n"
        f"- Injected context SHA-256: `{hashlib.sha256(context.encode('utf-8')).hexdigest()}`\n\n"
        "## Injected Context\n\n"
        "The content below is produced by formatting exactly the requested IDs.\n\n"
        f"{context}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand selected skill IDs without embedding, retrieval, or agent rollout."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--skill-id", dest="skill_ids", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    payload = json.loads(options.artifact.read_text(encoding="utf-8"))
    kind = artifact_kind(payload)
    expand = {
        "tower": expand_tower,
        "global_e2e": expand_global_e2e,
        "skillx": expand_skillx,
    }[kind]
    context, expanded_ids, referenced_child_ids = expand(payload, options.skill_ids)

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        render_markdown(
            options.artifact,
            kind,
            options.skill_ids,
            expanded_ids,
            referenced_child_ids,
            context,
        ),
        encoding="utf-8",
    )
    print(f"wrote {options.output} ({len(context)} injected context characters)")


if __name__ == "__main__":
    main()
