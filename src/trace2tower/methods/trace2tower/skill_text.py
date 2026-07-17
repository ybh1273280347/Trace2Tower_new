from __future__ import annotations

from trace2tower.methods.trace2tower.skills import HighSkillCard, MidSkillCard


def mid_card_text(card: MidSkillCard) -> str:
    return "\n".join((card.name, card.description, *card.procedure, *card.constraints))


def high_card_text(card: HighSkillCard) -> str:
    return card.retrieval_condition or "\n".join(
        (card.name, card.description, *card.procedure, *card.constraints)
    )
