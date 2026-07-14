from trace2tower.methods.trace2tower.lineage import (
    build_mid_lineage,
    legal_merge_proposals,
    legal_split_proposals,
)
from trace2tower.methods.trace2tower.models import MidCluster


def cluster(cluster_id: str, members: tuple[str, ...], x: float) -> MidCluster:
    return MidCluster(cluster_id, members, (x, 1.0 - x))


def test_lineage_identifies_only_pure_structural_relations() -> None:
    old = (
        cluster("old-a", ("a1", "a2"), 1.0),
        cluster("old-b", ("b1",), 0.5),
        cluster("old-c", ("c1",), 0.0),
        cluster("old-d", ("d1",), 0.2),
    )
    candidate = (
        cluster("new-a1", ("a1", "n1"), 1.0),
        cluster("new-a2", ("a2",), 0.9),
        cluster("new-bc", ("b1", "c1", "n2"), 0.25),
        cluster("new-d", ("d1",), 0.2),
        cluster("new-only", ("n3",), 0.7),
    )
    lineage = build_mid_lineage(old, candidate)

    assert lineage.continuations == (("old-d", "new-d"),)
    assert lineage.splits == (("old-a", ("new-a1", "new-a2")),)
    assert lineage.merges == ((("old-b", "old-c"), "new-bc"),)
    assert lineage.new_candidate_cluster_ids == ("new-only",)
    assert legal_split_proposals(lineage)[0].source_skill_id == "old-a"
    merge = legal_merge_proposals(lineage, old, candidate)[0]
    assert (merge.left_skill_id, merge.right_skill_id) == ("old-b", "old-c")


def test_lineage_keeps_cross_split_merge_out_of_legal_proposals() -> None:
    old = (
        cluster("old-a", ("a1", "a2"), 1.0),
        cluster("old-b", ("b1", "b2"), 0.0),
    )
    candidate = (
        cluster("new-left", ("a1", "b1"), 0.5),
        cluster("new-right", ("a2", "b2"), 0.5),
    )
    lineage = build_mid_lineage(old, candidate)

    assert lineage.complex_old_skill_ids == ("old-a", "old-b")
    assert lineage.complex_candidate_cluster_ids == ("new-left", "new-right")
    assert len(lineage.complex_repartitions) == 1
    assert lineage.complex_repartitions[0].source_old_skill_ids == (
        "old-a",
        "old-b",
    )
    assert not legal_split_proposals(lineage)
    assert not legal_merge_proposals(lineage, old, candidate)
