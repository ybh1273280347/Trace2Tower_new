from __future__ import annotations

from trace2tower.methods.trace2tower.models import MidCluster
from trace2tower.methods.trace2tower.three_signal_retrieval import (
    build_mid_transition_signal_profile,
    retrieve_mid_three_signal,
)
from trace2tower.semantic_index import SkillEmbeddingIndex


def test_three_signal_retrieval_prefers_successful_observed_transition() -> None:
    clusters = (
        MidCluster("mid_search", ("success:segment:0-0", "failure:segment:0-0"), ()),
        MidCluster("mid_pick", ("success:segment:1-1",), ()),
        MidCluster("mid_wrong", ("failure:segment:1-1",), ()),
    )
    records = (
        {
            "primary_score": 1.0,
            "segments": (
                {"segment_id": "success:segment:0-0", "start_step": 0},
                {"segment_id": "success:segment:1-1", "start_step": 1},
            ),
        },
        {
            "primary_score": 0.0,
            "segments": (
                {"segment_id": "failure:segment:0-0", "start_step": 0},
                {"segment_id": "failure:segment:1-1", "start_step": 1},
            ),
        },
    )
    profile = build_mid_transition_signal_profile(records, clusters)
    index = SkillEmbeddingIndex(
        ("mid_search", "mid_pick", "mid_wrong"),
        ((1.0, 0.0), (0.8, 0.2), (0.8, 0.2)),
    )

    matches = retrieve_mid_three_signal(
        (1.0, 0.0),
        index,
        frozenset(("mid_pick", "mid_wrong")),
        profile,
        top_k=2,
        score_threshold=0.0,
        anchor_top_k=1,
    )

    assert tuple(match.skill_id for match in matches) == ("mid_pick", "mid_wrong")
    assert matches[0].outcome_consistency > matches[1].outcome_consistency
