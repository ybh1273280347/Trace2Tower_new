from trace2tower.methods.trace2tower.core.models import MidCluster
from trace2tower.methods.trace2tower.induction.high_communities import discover_high_communities


def test_high_community_discovery_does_not_require_a_cluster_count() -> None:
    clusters = (
        MidCluster("mid_a", tuple(f"t{i}:segment:0-0" for i in (0, 1)), (0.0,)),
        MidCluster("mid_b", tuple(f"t{i}:segment:1-1" for i in (0, 1)), (1.0,)),
        MidCluster("mid_c", tuple(f"t{i}:segment:0-0" for i in (2, 3)), (2.0,)),
        MidCluster("mid_d", tuple(f"t{i}:segment:1-1" for i in (2, 3)), (3.0,)),
    )
    records = tuple(
        {
            "trajectory_id": f"t{i}",
            "primary_score": 1.0,
            "segments": [
                {"segment_id": f"t{i}:segment:{order}-{order}", "start_step": order}
                for order in (0, 1)
            ],
        }
        for i in range(4)
    )
    discovery = discover_high_communities(records, clusters, (), success_threshold=0.999)

    assert len(discovery.communities) == 2
    assert discovery.modularity > 0
