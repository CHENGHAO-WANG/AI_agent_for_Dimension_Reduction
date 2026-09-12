"""Manifold and neighbour-embedding executors.

The behavioural tests check that each method does what its registry entry claims, and
the negative controls matter as much as the positive ones: if MDS *also* unrolled the
Swiss roll, then unrolling would not be evidence of anything. The refusal tests check
that a precondition failure produces a message naming the parameter to change, since
that message is what the agent revises its plan from.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll

from drtools.executors import ExecutionError
from drtools.loaders import load
from drtools.pipeline import run_pipeline


@pytest.fixture(scope="module")
def roll():
    """Noise-free, so that a method's failure is the method's and not the data's."""
    return make_swiss_roll(n_samples=700, noise=0.0, random_state=0)


def unrolling(embedding: np.ndarray, t: np.ndarray) -> float:
    return max(
        abs(spearmanr(embedding[:, i], t).statistic) for i in range(embedding.shape[1])
    )


def neighbourhood_overlap(X: np.ndarray, embedding: np.ndarray, k: int = 10) -> float:
    """Fraction of each point's k nearest neighbours that survive the embedding."""
    from sklearn.neighbors import NearestNeighbors

    def neighbours(data):
        _, index = NearestNeighbors(n_neighbors=k + 1).fit(data).kneighbors(data)
        return [set(row[1:]) for row in index]

    before, after = neighbours(X), neighbours(embedding)
    return float(np.mean([len(a & b) / k for a, b in zip(before, after)]))


# -------------------------------------------------------------------------- mds


def test_mds_reports_stress_and_which_variant_ran(roll) -> None:
    X, _ = roll

    result = run_pipeline(
        X[:300], None, [{"op": "mds", "params": {"metric": False, "n_init": 1}}]
    )

    notes = result.stages[0].notes
    assert notes["variant"] == "non-metric (ordinal)"
    assert notes["stress"] >= 0
    assert "not a goodness-of-fit" in notes["caveat"]


def test_mds_cannot_unroll_the_swiss_roll(roll) -> None:
    """The negative control for the manifold methods: MDS uses ambient distances."""
    X, t = roll

    result = run_pipeline(X[:400], None, [{"op": "mds", "params": {"n_init": 1}}])

    assert unrolling(result.embedding, t[:400]) < 0.6


# ----------------------------------------------------------------------- isomap


def test_isomap_unrolls_the_swiss_roll(roll) -> None:
    """The canonical case: geodesic distance along the graph rather than through space."""
    X, t = roll

    result = run_pipeline(X, None, [{"op": "isomap", "params": {"n_neighbors": 10}}])

    assert unrolling(result.embedding, t) > 0.95
    assert result.stages[0].notes["graph_connected"] is True


def test_isomap_refuses_a_disconnected_graph_and_explains_the_geodesic(roll) -> None:
    X, labels, _ = load("blobs", n_samples=400, n_clusters=5)

    with pytest.raises(ExecutionError, match="no path and the geodesic is infinite"):
        run_pipeline(X, labels, [{"op": "isomap", "params": {"n_neighbors": 8}}])


# -------------------------------------------------------------------------- lle


@pytest.mark.parametrize("method", ["standard", "modified", "ltsa"])
def test_lle_variants_unroll_the_roll_at_a_small_neighbourhood(roll, method) -> None:
    X, t = roll

    result = run_pipeline(
        X, None, [{"op": "lle", "params": {"n_neighbors": 8, "method": method}}]
    )

    assert unrolling(result.embedding, t) > 0.9
    assert result.stages[0].notes["variant"] == method


def test_lle_degrades_when_neighbourhoods_span_folds_of_the_manifold(roll) -> None:
    """Documents the sensitivity the registry warns about, so it cannot regress silently.

    Once a neighbourhood reaches across two sheets of the roll, "locally linear" is
    false and the reconstruction weights stop meaning anything. This is a property of
    the method, not a bug, and the planner is told about it in the registry.
    """
    X, t = roll

    tight = run_pipeline(X, None, [{"op": "lle", "params": {"n_neighbors": 8}}])
    loose = run_pipeline(X, None, [{"op": "lle", "params": {"n_neighbors": 32}}])

    assert unrolling(tight.embedding, t) > 0.9
    assert unrolling(loose.embedding, t) < 0.6


def test_hessian_lle_names_the_neighbour_minimum_it_needs() -> None:
    """scikit-learn enforces this deep in a solve, in terms that name no plan parameter."""
    X, labels, _ = load("blobs", n_samples=200, n_features=5)

    with pytest.raises(ExecutionError, match="needs at least 6 neighbours"):
        run_pipeline(
            X, labels, [{"op": "lle", "params": {"n_neighbors": 4, "method": "hessian"}}]
        )


def test_lle_refuses_a_disconnected_graph() -> None:
    X, labels, _ = load("blobs", n_samples=400, n_clusters=5)

    with pytest.raises(ExecutionError, match="disconnected"):
        run_pipeline(X, labels, [{"op": "lle", "params": {"n_neighbors": 6}}])


# ------------------------------------------------------------------------- tsne


def test_tsne_preserves_local_neighbourhoods(roll) -> None:
    """What t-SNE claims to be good at, measured rather than assumed."""
    X, _ = roll

    result = run_pipeline(
        X[:400], None, [{"op": "tsne", "params": {"perplexity": 30, "n_iter": 250}}]
    )

    assert neighbourhood_overlap(X[:400], result.embedding) > 0.5


def test_tsne_refuses_a_perplexity_too_large_for_the_sample_count() -> None:
    """The default of 30 is meaningless at n = 60, and would still plot happily."""
    X, labels, _ = load("blobs", n_samples=60, n_features=5)

    with pytest.raises(ExecutionError, match="degenerates into a featureless blob"):
        run_pipeline(X, labels, [{"op": "tsne", "params": {"perplexity": 30}}])


def test_tsne_carries_its_interpretation_caveat_in_the_result(roll) -> None:
    """The caveat travels with the result rather than relying on anyone attaching it."""
    X, _ = roll

    result = run_pipeline(
        X[:300], None, [{"op": "tsne", "params": {"perplexity": 15, "n_iter": 250}}]
    )

    caveat = result.stages[0].notes["caveat"]
    assert "cluster sizes" in caveat and "not interpretable" in caveat


# ------------------------------------------------------------------------- umap


def test_umap_preserves_local_neighbourhoods(roll) -> None:
    X, _ = roll

    result = run_pipeline(
        X[:400], None, [{"op": "umap", "params": {"n_neighbors": 15}}]
    )

    assert neighbourhood_overlap(X[:400], result.embedding) > 0.5
    assert result.stages[0].notes["min_dist"] == 0.1


def test_umap_accepts_sparse_input_as_the_registry_claims() -> None:
    """Most methods here need dense input; the registry says UMAP does not."""
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    dense = (rng.random((200, 40)) < 0.2) * rng.random((200, 40))
    X = sp.csr_array(dense.astype(np.float64))

    result = run_pipeline(X, None, [{"op": "umap", "params": {"n_neighbors": 10}}])

    assert result.embedding.shape == (200, 2)


def test_umap_refuses_more_neighbours_than_samples() -> None:
    X, labels, _ = load("blobs", n_samples=20, n_features=4)

    with pytest.raises(ExecutionError, match="not smaller than the 20 samples"):
        run_pipeline(X, labels, [{"op": "umap", "params": {"n_neighbors": 30}}])
