from __future__ import annotations

from trace2tower.skills.induction import HighSkillCard, MidSkillCard


def format_tower_context(
    high_card: HighSkillCard | None,
    mid_cards: tuple[MidSkillCard, ...],
) -> str:
    sections = []
    if high_card is not None:
        sections.append(
            _format_card(
                "Strategy",
                high_card.name,
                high_card.description,
                high_card.procedure,
                high_card.constraints,
            )
        )
    sections.extend(
        _format_card("Skill", card.name, card.description, card.procedure, card.constraints)
        for card in mid_cards
    )
    return "\n\n".join(sections)


def mid_card_text(card: MidSkillCard) -> str:
    return "\n".join((card.name, card.description, *card.procedure, *card.constraints))


def high_card_text(card: HighSkillCard) -> str:
    return card.retrieval_condition or "\n".join(
        (card.name, card.description, *card.procedure, *card.constraints)
    )


def _format_card(
    kind: str,
    name: str,
    description: str,
    procedure: tuple[str, ...],
    constraints: tuple[str, ...],
) -> str:
    lines = [f"## {kind}: {name}", f"Use when: {description}", "Procedure:"]
    lines.extend(f"{index}. {step}" for index, step in enumerate(procedure, 1))
    if constraints:
        lines.append("Constraints:")
        lines.extend(f"- {constraint}" for constraint in constraints)
    return "\n".join(lines)
