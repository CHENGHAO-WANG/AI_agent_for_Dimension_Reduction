"""Every row covered: fit on the subsample, project the rest (section 3.12).

Three kinds of test. The exact ones check that a projection is the fitted stage applied
to new rows and nothing else: PCA's projected rows are the fitted model's `transform`,
a z-score reuses the fitted means, and each Nyström extension, applied to the rows it
was fitted on, returns their fitted coordinates. The measured ones check that projected
rows are placed as well as the fit placed its own, on a Swiss roll where position along
the roll is known. The structural ones check that the registry's `new_rows` claims hold
for every executor, and that what cannot project is refused.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances

import drtools.pipeline as pipeline
from drtools.executors import Context, ExecutionError, get_executor
from drtools.executors.spectral import DiffusionNystrom, LaplacianNystrom
from drtools.loaders import load
from drtools.pipeline import PipelineError, run_pipeline
from drtools.plan import _stage_peaks
from drtools.registry import load_registry

SUBSAMPLE = {"op": "subsample", "params": {"n_samples": 500}}


@pytest.fixture(scope="module")
def roll():
    X, t = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
    return X, t


def unrolling(embedding: np.ndarray, t: np.ndarray) -> float:
    return max(
        abs(spearmanr(embedding[:, i], t).statistic) for i in range(embedding.shape[1])
    )


def split(result, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    fitted = result.coverage.fitted_index
    return fitted, np.setdiff1d(np.arange(n_rows), fitted)


# ------------------------------------------------------------------------ exact


def test_projected_rows_are_the_fitted_pca_applied_to_them(roll) -> None:
    X, _ = roll
    result = run_pipeline(X, None, [SUBSAMPLE, {"op": "pca", "params": {}}])
    fitted, rest = split(result, len(X))

    model = PCA(n_components=2, random_state=0).fit(X[fitted])
    np.testing.assert_allclose(result.embedding[fitted], model.transform(X[fitted]))
    np.testing.assert_allclose(result.embedding[rest], model.transform(X[rest]))


def test_a_z_score_after_the_subsample_reuses_the_fitted_means() -> None:
    """Section 3.12's list: the z-score means and standard deviations the fit produced."""
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(900, 4))
    context = Context(seed=0)
    kept = X[:300]
    _, _ = get_executor("standardise")(kept, context)

    projected = context.projection.function(X[300:])
    expected = (X[300:] - kept.mean(axis=0)) / kept.std(axis=0)
    np.testing.assert_allclose(projected, expected, rtol=1e-12)
    assert context.projection.kind == "fitted parameters"


def test_a_projected_row_is_scaled_to_the_fitted_median_total() -> None:
    X = np.random.default_rng(0).poisson(3.0, size=(200, 30)).astype(float) + 1.0
    context = Context(seed=0)
    _, notes = get_executor("normalise_total")(X[:100], context)

    projected = context.projection.function(X[100:])
    np.testing.assert_allclose(projected.sum(axis=1), notes["target_total"])


def test_the_diffusion_extension_returns_fitted_rows_their_own_coordinates(roll) -> None:
    X, _ = roll
    context = Context(seed=0)
    embedding, _ = get_executor("diffusion_maps")(X[:400], context, n_components=3)

    assert isinstance(context.projection.function, DiffusionNystrom)
    np.testing.assert_allclose(
        context.projection.function(X[:400]), embedding, rtol=1e-6, atol=1e-10
    )


def test_the_laplacian_kernel_is_the_fitted_affinity_evaluated_at_a_row(roll) -> None:
    """Queried at a fitted row with itself left out, the kernel is its affinity row."""
    X, _ = roll
    fit = X[:400]
    model = SpectralEmbedding(
        n_components=2, affinity="nearest_neighbors", n_neighbors=15, random_state=0
    )
    embedding = model.fit_transform(fit)
    extension = LaplacianNystrom(fit, embedding, model.affinity_matrix_, 15)

    distances = pairwise_distances(fit)
    np.fill_diagonal(distances, np.inf)
    affinity = sp.csr_matrix(model.affinity_matrix_).toarray()
    np.fill_diagonal(affinity, 0.0)
    np.testing.assert_allclose(extension.kernel(distances), affinity)

    # And through the kernel, the eigen-equation returns each fitted row's coordinate.
    weights = extension.kernel(distances)
    rebuilt = weights @ embedding / weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        rebuilt / extension.eigenvalues, embedding, rtol=1e-6, atol=1e-10
    )


# --------------------------------------------------------------------- measured


@pytest.mark.parametrize(
    "op, floor",
    [("diffusion_maps", 0.95), ("laplacian_eigenmaps", 0.9)],
)
def test_the_nystrom_extensions_unroll_the_rows_they_project(roll, op, floor) -> None:
    """Measured on day 14: 0.989 and 0.931 on the projected rows, 500 of 1,500 fitted."""
    X, t = roll
    result = run_pipeline(X, None, [SUBSAMPLE, {"op": op, "params": {"n_components": 2}}])
    fitted, rest = split(result, len(X))

    assert unrolling(result.embedding[rest], t[rest]) > floor
    assert result.stages[1].projection["kind"] == "nystrom"


@pytest.mark.parametrize(
    "op", ["isomap", "lle", "kernel_pca", "diffusion_maps", "laplacian_eigenmaps"]
)
def test_projected_rows_are_placed_as_well_as_fitted_ones(roll, op) -> None:
    """The projection is judged against the fit, not against the method.

    Isomap and LLE fitted on 500 noisy rows do not unroll the roll -- that is the fit,
    which the subsample is part of. What the projection owes is to place the other rows
    as well as the fit placed its own; measured on day 14 the two agree to within 0.08
    for every method here.
    """
    X, t = roll
    result = run_pipeline(X, None, [SUBSAMPLE, {"op": op, "params": {"n_components": 2}}])
    fitted, rest = split(result, len(X))

    on_fitted = unrolling(result.embedding[fitted], t[fitted])
    on_projected = unrolling(result.embedding[rest], t[rest])
    assert on_projected > on_fitted - 0.1


# ------------------------------------------------------------------- structural


def _methods_by_new_rows(kind: str) -> list[str]:
    registry = load_registry()
    return sorted(
        name
        for name, spec in registry.ops.items()
        if (spec.is_reduction or spec.is_visualization) and spec.raw.get("new_rows") == kind
    )


@pytest.mark.parametrize(
    "op", _methods_by_new_rows("transform") + _methods_by_new_rows("nystrom")
)
def test_every_method_that_claims_to_place_new_rows_does(roll, op) -> None:
    """The registry's `new_rows` is what registration reads; the executor must agree."""
    X, _ = roll
    X = X[:400]
    stages = [{"op": "subsample", "params": {"n_samples": 200}},
              {"op": op, "params": {"n_components": 2}}]
    result = run_pipeline(X, None, stages)

    assert result.embedding.shape == (400, 2)
    assert np.isfinite(result.embedding).all()
    expected = load_registry()[op].raw["new_rows"]
    assert result.stages[1].projection["kind"] == (
        "nystrom" if expected == "nystrom" else "transform"
    )


def test_every_preprocessing_stage_can_follow_a_subsample() -> None:
    registry = load_registry()
    X = np.random.default_rng(0).poisson(4.0, size=(300, 12)).astype(float) + 1.0
    for name, spec in registry.ops.items():
        if spec.is_reduction or spec.is_visualization or name == "subsample":
            continue
        stages = [{"op": "subsample", "params": {"n_samples": 100}}, {"op": name},
                  {"op": "pca", "params": {}}]
        result = run_pipeline(X, None, stages)
        assert result.embedding.shape == (300, 2), name
        assert result.stages[1].projection is not None, name


def test_mds_cannot_follow_a_subsample(roll) -> None:
    """Registration refuses this first; the engine refuses it too, and says what to do."""
    X, _ = roll
    stages = [{"op": "subsample", "params": {"n_samples": 100}},
              {"op": "mds", "params": {}}]
    with pytest.raises(ExecutionError) as error:
        run_pipeline(X[:300], None, stages)
    assert "no way to place rows" in str(error.value)
    assert "reject mds" in str(error.value)
    assert error.value.op == "mds"


def test_a_candidate_subsamples_once(roll) -> None:
    X, _ = roll
    stages = [{"op": "subsample", "params": {"n_samples": 400}},
              {"op": "subsample", "params": {"n_samples": 200}},
              {"op": "pca", "params": {}}]
    with pytest.raises(PipelineError, match="subsamples twice"):
        run_pipeline(X[:600], None, stages)


def test_a_subsample_that_keeps_every_row_fits_every_row(roll) -> None:
    X, _ = roll
    stages = [{"op": "subsample", "params": {"n_samples": 5000}},
              {"op": "pca", "params": {}}]
    result = run_pipeline(X, None, stages)
    assert result.as_dict()["rows"]["fitted_on"] == "every row"
    assert result.coverage.fitted_index is None


# ---------------------------------------------------------------------- chunking


@pytest.mark.parametrize("op", ["pca", "isomap", "diffusion_maps"])
def test_chunked_projection_matches_projecting_at_once(roll, op, monkeypatch) -> None:
    X, _ = roll
    stages = [SUBSAMPLE, {"op": op, "params": {"n_components": 2}}]
    whole = run_pipeline(X, None, stages)

    # Rows per chunk are the budget over eight bytes times the fitted rows, here 500.
    monkeypatch.setattr(pipeline, "PROJECTION_CHUNK_BYTES", 8 * 500 * 64)
    chunked = run_pipeline(X, None, stages)

    assert whole.coverage.n_chunks == 1
    assert chunked.coverage.chunk_rows == 64
    assert chunked.coverage.n_chunks == -(-1000 // 64)
    np.testing.assert_allclose(chunked.embedding, whole.embedding, rtol=1e-8, atol=1e-10)


def test_the_record_says_which_rows_were_fitted_and_which_projected(roll) -> None:
    X, _ = roll
    stages = [{"op": "standardise"}, SUBSAMPLE,
              {"op": "pca", "params": {}}]
    record = run_pipeline(X, None, stages).as_dict()

    assert record["rows"] == {
        "n_rows": 1500,
        "n_fitted": 500,
        "n_projected": 1000,
        "fitted_on": "subsample",
        "chunk_rows": 1000,
        "n_chunks": 1,
    }
    # A stage before the subsample ran on every row, so it projected nothing.
    assert "projection" not in record["stages"][0]
    assert record["stages"][2]["projection"]["n_rows"] == 1000
    assert record["total_duration_s"] >= sum(s["duration_s"] for s in record["stages"])


# ------------------------------------------------------------------------ memory


def test_the_memory_estimate_holds_the_matrix_the_subsample_was_taken_from() -> None:
    """The rows not kept are projected from it, so it lives until the fit is done."""
    profile = {"shape": {"n_samples": 100_000, "n_features": 2_000, "dtype": "float64",
                         "storage": "dense"}, "values": {}}
    stages = [{"op": "drop_constant"}, {"op": "standardise"},
              {"op": "subsample", "params": {"n_samples": 5_000}},
              {"op": "isomap", "params": {}}]
    _, peaks = _stage_peaks(stages, profile, load_registry())

    standardised = 100_000 * 2_000 * 8
    isomap_input = 5_000 * 2_000 * 8
    isomap_output = 5_000 * 100 * 8  # the largest d tuning could choose (day 15)
    assert dict(peaks)["isomap"] == standardised + isomap_input + isomap_output


def test_labels_come_back_for_every_row() -> None:
    X, labels, _ = load("blobs", n_samples=600, n_clusters=3)
    stages = [{"op": "subsample", "params": {"n_samples": 150}},
              {"op": "pca", "params": {}}]
    result = run_pipeline(X, labels, stages)
    np.testing.assert_array_equal(result.labels, labels)
