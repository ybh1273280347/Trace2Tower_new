from trace2tower.methods.trace2tower.object_conditioned_retrieval import (
    ObjectConditionedHighProfile,
    retrieve_object_conditioned_high,
)
from trace2tower.semantic_index import SkillEmbeddingIndex


def test_object_conditioned_retrieval_rejects_semantically_closer_wrong_event() -> None:
    index = SkillEmbeddingIndex(
        ("cool_mug", "clean_mug"),
        ((1.0, 0.0), (0.8, 0.6)),
    )
    profiles = {
        "cool_mug": ObjectConditionedHighProfile("mug", "COOL", ("coffeemachine",)),
        "clean_mug": ObjectConditionedHighProfile("mug", "CLEAN", ("coffeemachine",)),
    }
    match = retrieve_object_conditioned_high(
        (1.0, 0.0),
        index,
        profiles,
        target_object="mug",
        transformation="CLEAN",
        destination="coffeemachine",
    )
    assert match is not None
    assert match.skill_id == "clean_mug"


def test_object_conditioned_retrieval_returns_none_without_exact_destination() -> None:
    index = SkillEmbeddingIndex(("clean_fork",), ((1.0, 0.0),))
    profiles = {
        "clean_fork": ObjectConditionedHighProfile("fork", "CLEAN", ("diningtable",)),
    }
    assert retrieve_object_conditioned_high(
        (1.0, 0.0),
        index,
        profiles,
        target_object="fork",
        transformation="CLEAN",
        destination="countertop",
    ) is None

