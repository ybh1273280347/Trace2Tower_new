from trace2tower.manifests import Benchmark
from trace2tower.methods.trace2tower.full_refinement import (
    apply_mid_updates,
    select_structural_updates,
)
from trace2tower.methods.trace2tower.lineage import build_mid_lineage
from trace2tower.methods.trace2tower.models import HighPath, MidCluster, SegmentInstance
from trace2tower.methods.trace2tower.refinement import (
    EpisodePairKey,
    ObjectiveVector,
    RankedSkillObjective,
    SkillLevel,
)


def cluster(cluster_id: str, members: tuple[str, ...], x: float) -> MidCluster:
    return MidCluster(cluster_id, members, (x, 1.0 - x))


def ranked(skill_id: str, rank: int) -> RankedSkillObjective:
    return RankedSkillObjective(
        skill_id,
        Benchmark.WEBSHOP,
        SkillLevel.MID,
        1,
        ObjectiveVector(1.0 / rank, 0.0, 0.0, 0.0),
        10,
        (EpisodePairKey(Benchmark.WEBSHOP, skill_id, 0),) * 10,
        rank,
        (),
        (),
    )


def segment(segment_id: str, value: float) -> SegmentInstance:
    return SegmentInstance(
        segment_id,
        segment_id.split(":segment:")[0],
        0,
        0,
        (),
        (value, 1.0 - value),
        1.0,
        None,
        (),
        "",
        "",
    )


def test_full_refinement_applies_one_split_and_partitions_new_segments() -> None:
    old = (
        cluster("mid-a", ("t1:segment:0", "t2:segment:0"), 1.0),
        cluster("mid-b", ("t3:segment:0",), 0.0),
    )
    candidate = (
        cluster("candidate-a1", ("t1:segment:0", "t4:segment:0"), 1.0),
        cluster("candidate-a2", ("t2:segment:0",), 0.8),
        cluster("candidate-b", ("t3:segment:0", "t5:segment:0"), 0.0),
    )
    lineage = build_mid_lineage(old, candidate)
    selection = select_structural_updates(
        lineage,
        old,
        candidate,
        (HighPath("high", ("mid-a", "mid-b"), 1, 0, 1, ("t1",)),),
        {"mid-a": ranked("mid-a", 2), "mid-b": ranked("mid-b", 1)},
        max_high_path_length=4,
        minimum_exposure_count=10,
    )
    assert selection.split is not None

    segments = {
        segment_id: segment(segment_id, index / 5)
        for index, segment_id in enumerate(
            (
                "t1:segment:0",
                "t2:segment:0",
                "t3:segment:0",
                "t4:segment:0",
                "t5:segment:0",
            )
        )
    }
    refined = apply_mid_updates(lineage, selection, old, candidate, segments)
    assigned = [
        member for cluster_item in refined.clusters for member in cluster_item.member_segment_ids
    ]
    assert len(assigned) == len(set(assigned)) == 5
    assert len(refined.replacement_by_old_id["mid-a"]) == 2
    assert len(refined.primary_replacement_by_old_id["mid-a"]) == 1
    assert refined.replacement_by_old_id["mid-b"] == ("mid-b",)


def test_full_refinement_applies_complex_repartition_atomically() -> None:
    old = (
        cluster("mid-a", ("t1:segment:0", "t2:segment:0"), 1.0),
        cluster("mid-b", ("t3:segment:0", "t4:segment:0"), 0.0),
        cluster("mid-c", ("t5:segment:0",), 0.5),
    )
    candidate = (
        cluster("candidate-left", ("t1:segment:0", "t3:segment:0"), 0.6),
        cluster(
            "candidate-right",
            ("t2:segment:0", "t4:segment:0", "t6:segment:0"),
            0.4,
        ),
        cluster("candidate-c", ("t5:segment:0",), 0.5),
    )
    lineage = build_mid_lineage(old, candidate)
    selection = select_structural_updates(
        lineage,
        old,
        candidate,
        (HighPath("high", ("mid-a", "mid-c"), 1, 0, 1, ("t1",)),),
        {
            "mid-a": ranked("mid-a", 3),
            "mid-b": ranked("mid-b", 2),
            "mid-c": ranked("mid-c", 1),
        },
        max_high_path_length=4,
        minimum_exposure_count=10,
    )
    assert selection.repartition is not None

    segments = {
        segment_id: segment(segment_id, index / 6)
        for index, segment_id in enumerate(
            (
                "t1:segment:0",
                "t2:segment:0",
                "t3:segment:0",
                "t4:segment:0",
                "t5:segment:0",
                "t6:segment:0",
            )
        )
    }
    refined = apply_mid_updates(lineage, selection, old, candidate, segments)
    assert "mid-a" not in {item.cluster_id for item in refined.clusters}
    assert "mid-b" not in {item.cluster_id for item in refined.clusters}
    assert len(refined.replacement_by_old_id["mid-a"]) == 2
    assert len(refined.replacement_by_old_id["mid-b"]) == 2
    assert len(refined.primary_replacement_by_old_id["mid-a"]) == 1
