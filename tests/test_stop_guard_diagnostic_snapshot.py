from scripts.experiments.build.build_stop_guard_diagnostic_snapshot import (
    add_completion_guards,
)
from trace2tower.methods.trace2tower.induction.skills import HighSkillCard


def high_card(*, constraints: tuple[str, ...], final_step: str) -> HighSkillCard:
    return HighSkillCard(
        skill_id="high_fixed",
        ordered_mid_ids=("mid_a", "mid_b"),
        name="Original name",
        description="Original description",
        procedure=("Search for a product.", final_step),
        constraints=constraints,
        retrieval_condition="Original retrieval condition",
    )


def test_completion_guard_is_the_only_base_card_change() -> None:
    base = high_card(
        constraints=("Keep the original constraint.",),
        final_step="Buy after verification.",
    )
    stateful = high_card(
        constraints=("Unused stateful constraint.",),
        final_step="Once visibly matched, click Buy Now immediately.",
    )

    patched = add_completion_guards((base,), (stateful,))[0]

    assert patched == HighSkillCard(
        skill_id=base.skill_id,
        ordered_mid_ids=base.ordered_mid_ids,
        name=base.name,
        description=base.description,
        procedure=base.procedure,
        constraints=(
            "Keep the original constraint.",
            "Once visibly matched, click Buy Now immediately.",
        ),
        retrieval_condition=base.retrieval_condition,
    )
