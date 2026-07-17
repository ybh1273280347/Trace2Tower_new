from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import yaml
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from trace2tower.methods.trace2tower.config import Trace2TowerConfig
from trace2tower.methods.trace2tower.graph import (
    GraphComponents,
    _nearest_neighbors,
    build_graph,
    embedding_node_groups,
)
from trace2tower.methods.trace2tower.models import (
    AlfworldEventType,
    SegmentInstance,
    WebShopEventType,
)
from trace2tower.methods.trace2tower.spectral import (
    cluster_representation,
    semantic_only_clustering,
    separate_exclusive_event_clusters,
    spectral_clustering,
)
from trace2tower.results import MethodName


def config(**overrides) -> Trace2TowerConfig:
    values = {
        "method": MethodName.TRACE2TOWER,
        "semantic_only": False,
        "use_transition_edge": True,
        "use_outcome_edge": True,
        "use_contrastive_decomposition": True,
        "failure_penalty": 1.0,
        "min_mid_clusters": 2,
        "max_mid_clusters": 4,
        "random_state": 42,
    }
    values.update(overrides)
    return Trace2TowerConfig(**values)


def segment(
    segment_id: str,
    trajectory_id: str,
    step: int,
    embedding: tuple[float, ...],
    score: float,
    event_type: WebShopEventType | None = WebShopEventType.QUERY_FORMULATION,
) -> SegmentInstance:
    return SegmentInstance(
        segment_id=segment_id,
        trajectory_id=trajectory_id,
        start_step=step,
        end_step=step,
        transition_ids=(f"{segment_id}:transition",),
        embedding=embedding,
        trajectory_score=score,
        event_type=event_type,
        raw_actions=("look",),
        observation_before="before",
        observation_after="after",
    )


def test_rho_uses_own_score_with_unit_smoothing_prior() -> None:
    segments = (
        (segment("s0", "t0", 0, (1.0, 0.0), 1.0),),
        (segment("s1", "t1", 0, (1.0, 0.0), 0.0),),
    )
    graph = build_graph(segments, config())
    assert graph.rho == pytest.approx((0.5, 0.5))


def test_collapsed_duplicate_embeddings_use_full_group_outcome() -> None:
    groups = (
        (segment("s0", "t0", 0, (1.0, 0.0), 1.0),),
        (segment("s1", "t1", 0, (1.0, 0.0), 1.0),),
        (segment("s2", "t2", 0, (1.0, 0.0), 0.0),),
        (segment("s3", "t3", 0, (0.0, 1.0), 0.0),),
    )
    graph = build_graph(groups, config(collapse_duplicate_embeddings=True))
    assert len(graph.segment_ids) == 2
    assert sorted(len(group) for group in graph.node_member_segment_ids) == [1, 3]
    assert sorted(graph.rho) == pytest.approx([0.0, 2 / 3])

    clustering = cluster_representation(
        graph.segment_ids,
        graph.segment_embeddings,
        np.eye(2),
        2,
        42,
        (),
        node_member_segment_ids=graph.node_member_segment_ids,
    )
    assert sorted(len(cluster.member_segment_ids) for cluster in clustering.clusters) == [1, 3]


def test_semantic_only_clustering_expands_collapsed_embedding_nodes() -> None:
    segments = (
        segment("s0", "t0", 0, (1.0, 0.0), 1.0),
        segment("s1", "t1", 0, (1.0, 0.0), 1.0),
        segment("s2", "t2", 0, (1.0, 0.0), 0.0),
        segment("s3", "t3", 0, (0.0, 1.0), 0.0),
    )
    node_groups = embedding_node_groups(
        segments,
        collapse_duplicate_embeddings=True,
    )
    node_segments = tuple(group[0] for group in node_groups)

    clustering = semantic_only_clustering(
        tuple(segment.segment_id for segment in node_segments),
        np.asarray([segment.embedding for segment in node_segments]),
        cluster_count=2,
        random_state=42,
        node_member_segment_ids=tuple(
            tuple(segment.segment_id for segment in group) for group in node_groups
        ),
    )

    assert len(clustering.labels) == 2
    assert sorted(len(cluster.member_segment_ids) for cluster in clustering.clusters) == [1, 3]
    assert {
        segment_id
        for cluster in clustering.clusters
        for segment_id in cluster.member_segment_ids
    } == {"s0", "s1", "s2", "s3"}


def test_config_rejects_string_boolean() -> None:
    record = yaml.safe_load(
        """
method: trace2tower
semantic_only: "false"
use_transition_edge: true
use_outcome_edge: true
use_contrastive_decomposition: true
failure_penalty: 1
min_mid_clusters: 2
max_mid_clusters: 20
random_state: 42
"""
    )
    with pytest.raises(ValueError, match="switches must be booleans"):
        Trace2TowerConfig.from_record(record)


def test_graph_components_and_signed_formula() -> None:
    groups = (
        (segment("s0", "a0", 0, (1.0, 0.0), 1.0),),
        (segment("s1", "a1", 0, (1.0, 0.0), 1.0),),
        (segment("s2", "b0", 0, (0.0, 1.0), 0.0),),
        (segment("s3", "b1", 0, (0.0, 1.0), 0.0),),
    )
    full = build_graph(groups, config())
    assert np.all((0 <= full.rho) & (full.rho <= 1))
    for matrix in (full.semantic, full.transition, full.outcome, full.base):
        assert matrix.data.size == 0 or np.all((0 <= matrix.data) & (matrix.data <= 1))
        assert (matrix - matrix.T).nnz == 0
    assert (full.adjacency - full.adjacency.T).nnz == 0
    assert np.isfinite(full.laplacian.data).all()
    assert full.adjacency[0, 1] > 0
    assert full.adjacency[2, 3] < 0


def test_observed_transitions_connect_different_event_types() -> None:
    query = WebShopEventType.QUERY_FORMULATION
    candidate = WebShopEventType.CANDIDATE_SELECTION
    purchase = WebShopEventType.PURCHASE_DECISION
    groups = (
        (
            segment("q0", "t0", 0, (1.0, 0.0), 1.0, query),
            segment("c0", "t0", 1, (0.0, 1.0), 1.0, candidate),
        ),
        (
            segment("q1", "t1", 0, (1.0, 0.0), 0.0, query),
            segment("p1", "t1", 1, (-1.0, 0.0), 0.0, purchase),
        ),
    )
    graph = build_graph(groups, config())

    assert graph.transition[0, 1] == pytest.approx(0.5)
    assert graph.transition[2, 3] == pytest.approx(0.5)
    assert graph.cross_event_edge_count > 0
    assert graph.transition_edge_count == 2
    assert graph.positive[0, 1] > graph.negative[0, 1]
    assert graph.negative[2, 3] > graph.positive[2, 3]
    assert graph.adjacency[0, 1] > 0
    assert graph.adjacency[2, 3] < 0


def test_transition_graph_rejects_unlabeled_segments() -> None:
    groups = (
        (
            segment("s0", "t0", 0, (1.0, 0.0), 1.0, None),
            segment("s1", "t0", 1, (0.0, 1.0), 1.0, None),
        ),
    )

    with pytest.raises(ValueError, match="requires domain event labels"):
        build_graph(groups, config())


def test_config_rejects_event_stratification_as_non_algorithmic() -> None:
    record = config().to_record()
    record["event_type_stratification"] = True
    with pytest.raises(ValueError, match="not part of the Trace2Tower algorithm"):
        Trace2TowerConfig.from_record(record)


def test_zero_degree_laplacian_is_finite() -> None:
    graph = build_graph(
        ((segment("s0", "t0", 0, (0.0, 0.0), 0.0),),),
        config(),
    )
    assert graph.edge_count == 0
    assert np.allclose(graph.laplacian.toarray(), ((1.0,),))
    assert np.isfinite(graph.laplacian.data).all()


def test_nearest_neighbor_batches_match_one_shot_query() -> None:
    generator = np.random.default_rng(42)
    values = generator.normal(size=(257, 8))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    expected = (
        NearestNeighbors(
            n_neighbors=6,
            metric="cosine",
            algorithm="brute",
        )
        .fit(values)
        .kneighbors(values, return_distance=False)
    )

    actual = _nearest_neighbors(values, neighbor_count=5)

    assert len(actual) == len(values)
    for source, targets in enumerate(expected):
        without_self = [target for target in targets if target != source][:5]
        assert actual[source].tolist() == without_self


def two_block_graph() -> GraphComponents:
    block = np.ones((4, 4)) - np.eye(4)
    adjacency = sparse.block_diag((block, block), format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scaling = sparse.diags(1 / np.sqrt(degree))
    laplacian = sparse.eye(8, format="csr") - scaling @ adjacency @ scaling
    zeros = sparse.csr_matrix((8, 8))
    embeddings = np.asarray(
        [(1.0, 0.0)] * 4 + [(0.0, 1.0)] * 4,
        dtype=np.float64,
    )
    return GraphComponents(
        segment_ids=tuple(f"s{index}" for index in range(8)),
        event_types=(None,) * 8,
        segment_embeddings=embeddings,
        rho=np.ones(8),
        semantic=adjacency,
        transition=zeros,
        outcome=adjacency,
        base=adjacency,
        positive=adjacency,
        negative=zeros,
        adjacency=adjacency,
        laplacian=laplacian.tocsr(),
        neighbor_count=3,
        edge_count=12,
        transition_edge_count=0,
        cross_event_edge_count=0,
    )


def test_eigengap_selects_two_blocks_and_semantic_only_reuses_k() -> None:
    graph = two_block_graph()
    result = spectral_clustering(graph, config())
    assert result.cluster_count == 2
    assert len(set(result.labels[:4])) == 1
    assert len(set(result.labels[4:])) == 1
    assert result.labels[0] != result.labels[4]

    semantic = semantic_only_clustering(
        graph.segment_ids,
        graph.segment_embeddings,
        cluster_count=result.cluster_count,
        random_state=42,
    )
    assert semantic.cluster_count == result.cluster_count
    assert semantic.labels == result.labels


def test_kmeans_partition_is_invariant_to_eigenvector_sign_flip() -> None:
    graph = two_block_graph()
    representation = np.asarray([(1.0, 0.2)] * 4 + [(0.1, 1.0)] * 4)
    original = cluster_representation(
        graph.segment_ids,
        graph.segment_embeddings,
        representation,
        2,
        42,
        (),
    )
    flipped = cluster_representation(
        graph.segment_ids,
        graph.segment_embeddings,
        representation * np.asarray((-1.0, 1.0)),
        2,
        42,
        (),
    )
    assert original.labels == flipped.labels


def test_exclusive_events_get_operator_clusters_without_merging_other_roles() -> None:
    graph = two_block_graph()
    graph = replace(
        graph,
        event_types=(
            AlfworldEventType.GOTO_LOCATION,
            AlfworldEventType.GOTO_LOCATION,
            AlfworldEventType.CLEAN_OBJECT,
            AlfworldEventType.CLEAN_OBJECT,
            AlfworldEventType.GOTO_LOCATION,
            AlfworldEventType.GOTO_LOCATION,
            AlfworldEventType.CLEAN_OBJECT,
            AlfworldEventType.CLEAN_OBJECT,
        ),
    )
    spectral = spectral_clustering(graph, config())

    refined = separate_exclusive_event_clusters(
        spectral,
        graph,
        frozenset((AlfworldEventType.CLEAN_OBJECT,)),
    )

    memberships = {frozenset(cluster.member_segment_ids) for cluster in refined.clusters}
    assert refined.cluster_count == 3
    assert frozenset(("s2", "s3", "s6", "s7")) in memberships
    assert frozenset(("s0", "s1")) in memberships
    assert frozenset(("s4", "s5")) in memberships
