"""The stage engine and the executors it drives.

Two kinds of test here. The structural ones check that a malformed candidate is
rejected before any compute is spent on it, and that the rejection says what to do
instead — the agent repairs its plan from these messages. The behavioural ones check
that the methods do what the registry claims: a manifold method has to actually
recover the Swiss roll's parameter, or every claim the report makes about preserved
structure is unfounded.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll

from drtools.executors import ExecutionError
from drtools.loaders import load
from drtools.pipeline import (
    PipelineError,
    normalise_stages,
    run_pipeline,
    validate_stages,
)


@pytest.fixture(scope="module")
def swiss_roll():
    X, t = make_swiss_roll(n_samples=600, noise=0.05, random_state=0)
    return X, t


def unrolling_correlation(embedding: np.ndarray, t: np.ndarray) -> float:
    """How well the best single coordinate recovers position along the roll."""
    return max(
        abs(spearmanr(embedding[:, i], t).statistic) for i in range(embedding.shape[1])
    )


# ------------------------------------------------------------------- stage syntax


def test_bare_names_and_full_mappings_mean_the_same_thing() -> None:
    assert normalise_stages(["pca"]) == [{"op": "pca", "params": {}}]


def test_a_stage_without_an_op_is_rejected() -> None:
    with pytest.raises(PipelineError, match="missing required key 'op'"):
        normalise_stages([{"params": {"n_components": 2}}])


def test_an_empty_candidate_is_rejected() -> None:
    with pytest.raises(PipelineError, match="at least one stage"):
        validate_stages([])


def test_a_terminal_method_cannot_feed_another_stage() -> None:
    """t-SNE-like coordinates have no metric for a downstream method to consume."""
    with pytest.raises(PipelineError, match="terminal method"):
        validate_stages([{"op": "diffusion_maps"}, {"op": "pca"}])


def test_a_candidate_must_end_in_a_reduction() -> None:
    with pytest.raises(PipelineError, match="must end in a reduction"):
        validate_stages([{"op": "log1p"}])


def test_pca_may_feed_a_terminal_method() -> None:
    stages = validate_stages([{"op": "pca", "params": {"n_components": 10}},
                              {"op": "diffusion_maps"}])
    assert [s["op"] for s in stages] == ["pca", "diffusion_maps"]


# ---------------------------------------------------------------- stage recording


def test_each_stage_records_what_it_ran_with_and_where_values_came_from() -> None:
    X, labels, _ = load("blobs", n_samples=200, n_features=10)

    result = run_pipeline(
        X, labels, [{"op": "standardise"}, {"op": "pca", "params": {"n_components": 3}}]
    )

    assert [s.op for s in result.stages] == ["standardise", "pca"]
    assert result.embedding.shape == (200, 3)
    assert result.stages[1].param_provenance["n_components"] == "specified"
    assert result.stages[1].param_provenance["whiten"] == "registry_default"
    assert result.total_duration_s > 0


def test_subsampling_keeps_labels_aligned_and_declares_the_caveat() -> None:
    """A metric computed after this stage describes the subset, and must say so."""
    X, labels, _ = load("blobs", n_samples=1000, n_clusters=5)

    result = run_pipeline(
        X, labels, [{"op": "subsample", "params": {"n_samples": 250}}, {"op": "pca"}]
    )

    assert result.embedding.shape[0] == result.labels.shape[0]
    assert result.embedding.shape[0] <= 255
    assert set(np.unique(result.labels)) == set(range(5)), "a class was lost"
    assert "subsample" in result.stages[0].notes["strategy"] or result.stages[0].notes[
        "strategy"
    ] == "stratified by label"
    assert "describe the subsample" in result.stages[0].notes["caveat"]


def test_constant_features_are_dropped_and_counted() -> None:
    X, labels, meta = load("blobs", n_samples=100, n_features=6)
    X = np.hstack([X, np.zeros((100, 4))])

    result = run_pipeline(X, labels, [{"op": "drop_constant"}, {"op": "pca"}])

    assert result.stages[0].notes["n_dropped"] == 4
    assert result.stages[0].output_shape == (100, 6)


def test_feature_selection_reports_what_it_kept() -> None:
    X, labels, _ = load("sparse_counts", n_cells=150, n_genes=400)

    result = run_pipeline(
        X,
        labels,
        [
            {"op": "select_variable_features", "params": {"n_features": 100}},
            {"op": "pca", "params": {"n_components": 5}},
        ],
    )

    assert result.stages[0].notes["n_features_out"] == 100
    assert result.stages[0].notes["criterion"] == "variance"


# ------------------------------------------------------------------- sparse paths


def test_sparse_pca_uses_truncated_svd_and_says_it_did_not_centre() -> None:
    """The distinction matters: uncentred, the first component tracks magnitude."""
    X = sp.csr_array(np.abs(np.random.default_rng(0).standard_normal((80, 30))))

    result = run_pipeline(X, None, [{"op": "pca", "params": {"n_components": 4}}])

    notes = result.stages[0].notes
    assert notes["solver"] == "TruncatedSVD"
    assert notes["centred"] is False
    assert "uncentred" in notes["caveat"]


def test_standardising_a_sparse_matrix_is_refused_with_the_memory_cost() -> None:
    X = sp.csr_array(np.eye(500, 4000, dtype=np.float64))

    with pytest.raises(ExecutionError, match="GB dense"):
        run_pipeline(X, None, [{"op": "standardise"}, {"op": "pca"}])


def test_a_dense_only_method_names_the_stage_that_would_fix_it() -> None:
    X = sp.csr_array(np.eye(100, 20, dtype=np.float64))

    with pytest.raises(ExecutionError, match="densify"):
        run_pipeline(X, None, [{"op": "diffusion_maps"}])


# --------------------------------------------------------------- method behaviour


def test_diffusion_maps_unrolls_the_swiss_roll(swiss_roll) -> None:
    """The claim in the registry is that it preserves manifold structure. Check it."""
    X, t = swiss_roll

    result = run_pipeline(X, None, [{"op": "diffusion_maps", "params": {"alpha": 1.0}}])

    assert unrolling_correlation(result.embedding, t) > 0.9


def test_laplacian_eigenmaps_unrolls_the_swiss_roll(swiss_roll) -> None:
    X, t = swiss_roll

    result = run_pipeline(
        X, None, [{"op": "laplacian_eigenmaps", "params": {"n_neighbors": 12}}]
    )

    assert unrolling_correlation(result.embedding, t) > 0.9


def test_pca_cannot_unroll_the_swiss_roll(swiss_roll) -> None:
    """The negative control. Without it the positive results prove nothing."""
    X, t = swiss_roll

    result = run_pipeline(X, None, [{"op": "pca", "params": {"n_components": 2}}])

    assert unrolling_correlation(result.embedding, t) < 0.5


def test_laplacian_eigenmaps_refuses_a_disconnected_graph_and_says_how_to_fix_it() -> None:
    """It would otherwise return component indicators that plot as a clean clustering."""
    X, labels, _ = load("blobs", n_samples=400, n_clusters=5)

    with pytest.raises(ExecutionError, match="disconnected components"):
        run_pipeline(X, labels, [{"op": "laplacian_eigenmaps", "params": {"n_neighbors": 8}}])


def test_pca_recovers_the_rank_of_a_low_rank_dataset() -> None:
    X, labels, _ = load("linear_subspace", n_samples=400, rank=5)

    result = run_pipeline(X, labels, [{"op": "pca", "params": {"n_components": 5}}])

    assert result.stages[0].notes["cumulative_explained_variance"] > 0.95


def test_diffusion_maps_reports_the_bandwidth_and_time_it_used(swiss_roll) -> None:
    """Different t give different, equally valid geometries, so both must be recorded."""
    X, _ = swiss_roll

    result = run_pipeline(X, None, [{"op": "diffusion_maps", "params": {"t": 2}}])

    notes = result.stages[0].notes
    assert notes["t"] == 2
    assert notes["epsilon"] > 0
    assert "scaling criterion" in notes["epsilon_source"]
    # The bandwidth search also estimates intrinsic dimension, which is a free check
    # on the two-NN estimate from reconnaissance. The roll is a 2-D manifold.
    assert 1.5 < notes["dimension_implied_by_bandwidth"] < 3.0


@pytest.mark.parametrize("n_samples", [300, 700, 1500])
def test_the_bandwidth_rule_holds_across_sampling_densities(n_samples: int) -> None:
    """The regression this rule exists for.

    A bandwidth taken from the k-th neighbour distance scales with sampling density,
    while the gaps a manifold method must not bridge do not, so the roll unrolled at
    n = 1500 and collapsed at n = 400. Anything that reintroduces a density-scaled
    default should fail here rather than in a report.
    """
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.05, random_state=0)

    result = run_pipeline(X, None, [{"op": "diffusion_maps", "params": {"alpha": 1.0}}])

    assert unrolling_correlation(result.embedding, t) > 0.9


def test_requesting_more_components_than_exist_is_refused_clearly() -> None:
    X, labels, _ = load("swiss_roll", n_samples=100)

    with pytest.raises(ExecutionError, match="at most 3"):
        run_pipeline(X, labels, [{"op": "pca", "params": {"n_components": 10}}])


def test_a_pipeline_is_deterministic_under_a_fixed_seed() -> None:
    X, labels, _ = load("blobs", n_samples=300, n_features=8)
    stages = [{"op": "subsample", "params": {"n_samples": 150}}, {"op": "pca"}]

    first = run_pipeline(X, labels, stages, seed=11)
    second = run_pipeline(X, labels, stages, seed=11)

    np.testing.assert_allclose(first.embedding, second.embedding)
