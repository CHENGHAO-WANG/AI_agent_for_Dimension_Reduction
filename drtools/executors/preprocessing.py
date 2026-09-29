"""Preprocessing stages.

Each one is a transformation the planner has to choose deliberately and can be held to
afterwards. None of them run implicitly: if a dataset needs normalising, that appears
in the plan as a stage and in the report as a decision, rather than happening inside
some other method's constructor where nobody can see it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix
from drtools.executors import Context, ExecutionError, executor


def _feature_variance(X: Matrix) -> np.ndarray:
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_of_squares = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        return np.maximum(mean_of_squares - mean**2, 0.0)
    return np.asarray(X).var(axis=0)


#: Section 3.10's tolerance for a constant feature.
CONSTANT_EPS = 1e-12


def constant_features(X: Matrix) -> tuple[np.ndarray, np.ndarray]:
    """Which features are constant, and which of those only up to the tolerance.

    A feature is constant when `max - min <= eps * max(1, max |x|)`. The range, not the
    variance: a variance computed in floating point is rarely exactly zero for a
    constant non-zero column (defect 19), while the range of stored identical values is.
    The floor of 1 makes the test absolute below magnitude 1, which catches a column
    zero only up to rounding, at the cost of dropping a feature whose genuine range is
    below 1e-12. That is why the second mask exists: every drop the tolerance made,
    whose range was not exactly zero, is listed so a wrong one is visible.

    On sparse data a column's implicit zeros count toward its minimum and maximum.
    """
    if sp.issparse(X):
        X = sp.csc_matrix(X)
        high = np.asarray(X.max(axis=0).todense()).ravel().astype(np.float64)
        low = np.asarray(X.min(axis=0).todense()).ravel().astype(np.float64)
    else:
        dense = np.asarray(X)
        high = dense.max(axis=0).astype(np.float64)
        low = dense.min(axis=0).astype(np.float64)
    spread = high - low
    magnitude = np.maximum(np.abs(high), np.abs(low))
    constant = spread <= CONSTANT_EPS * np.maximum(1.0, magnitude)
    return constant, constant & (spread > 0)


@executor("drop_constant")
def drop_constant(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    constant, by_tolerance = constant_features(X)
    keep = np.flatnonzero(~constant)
    if keep.size == 0:
        raise ExecutionError(
            "every feature is constant; there is no variation to embed"
        )
    # Indices into the loaded matrix, so the report can name them whatever ran before.
    original = (
        ctx.feature_index
        if ctx.feature_index is not None
        else np.arange(X.shape[1])
    )
    tolerated = [int(original[i]) for i in np.flatnonzero(by_tolerance)]
    ctx.select_features(keep)
    ctx.project_with(lambda Z: Z[:, keep], "fitted parameters")
    return X[:, keep], {
        "n_features_in": int(X.shape[1]),
        "n_features_out": int(keep.size),
        "n_dropped": int(X.shape[1] - keep.size),
        "dropped_by_tolerance": tolerated,
    }


@executor("normalise_total")
def normalise_total(
    X: Matrix, ctx: Context, *, target: float | None = None, **_: Any
) -> tuple[Matrix, dict[str, Any]]:
    totals = np.asarray(X.sum(axis=1)).ravel()
    positive = totals[totals > 0]
    if positive.size == 0:
        raise ExecutionError("every sample totals zero; nothing to normalise")

    chosen = float(target) if target is not None else float(np.median(positive))
    scaled = _scale_rows_to(X, chosen)
    # The target is what the fit measured, so a projected row is scaled to the median
    # total of the fitted rows rather than to one of its own chunk's.
    ctx.project_with(lambda Z: _scale_rows_to(Z, chosen), "fitted parameters")

    return scaled, {
        "target_total": chosen,
        "target_source": "specified" if target is not None else "median sample total",
        "n_zero_total_samples": int((totals == 0).sum()),
    }


def _scale_rows_to(X: Matrix, target: float) -> Matrix:
    totals = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(
        target, totals, out=np.ones_like(totals, dtype=np.float64), where=totals > 0
    )
    return sp.diags(scale) @ X if sp.issparse(X) else np.asarray(X) * scale[:, None]


@executor("log1p")
def log1p(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    values = X.data if sp.issparse(X) else X
    if np.any(values < 0):
        raise ExecutionError(
            "log1p requires non-negative input; this matrix contains negative values, "
            "so it has most likely already been centred or scaled"
        )
    ctx.project_with(lambda Z: log1p(Z, Context())[0], "row-wise")
    return (X.log1p() if sp.issparse(X) else np.log1p(X)), {"base": "natural"}


@executor("standardise")
def standardise(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    """Centre and scale every feature, returning a dense matrix in the input's dtype.

    Sparse input is densified rather than refused: centring makes every zero non-zero,
    so the output is dense whatever the input, and the plan validator has already
    counted that dense matrix against the memory limit before this runs. Float32 stays
    float32, which is ample precision for a z-score and halves the matrix.
    """
    dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float64
    dense = np.asarray(X.todense() if sp.issparse(X) else X, dtype=dtype)
    mean = dense.mean(axis=0, dtype=np.float64)
    std = dense.std(axis=0, dtype=np.float64)
    n_constant = int((std == 0).sum())
    std[std == 0] = 1.0
    out = dense - mean.astype(dtype)
    out /= std.astype(dtype)

    def project(Z: Matrix) -> np.ndarray:
        z = np.asarray(Z.todense() if sp.issparse(Z) else Z, dtype=dtype)
        return (z - mean.astype(dtype)) / std.astype(dtype)

    ctx.project_with(project, "fitted parameters")
    ctx.keep("mean", mean)
    ctx.keep("std", std)
    return out, {
        "n_constant_features_left_unscaled": n_constant,
        "densified": bool(sp.issparse(X)),
        "dtype": str(np.dtype(dtype)),
    }


@executor("l2_normalise")
def l2_normalise(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    if sp.issparse(X):
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    else:
        norms = np.linalg.norm(np.asarray(X), axis=1)
    n_zero = int((norms == 0).sum())
    if n_zero:
        raise ExecutionError(
            f"{n_zero} samples have zero norm, so their direction is undefined. Drop "
            "them before normalising."
        )
    scale = 1.0 / norms
    scaled = sp.diags(scale) @ X if sp.issparse(X) else np.asarray(X) * scale[:, None]
    ctx.project_with(lambda Z: l2_normalise(Z, Context())[0], "row-wise")
    return scaled, {"norm": "l2"}


@executor("select_variable_features")
def select_variable_features(
    X: Matrix,
    ctx: Context,
    *,
    n_features: int = 2000,
    **_: Any,
) -> tuple[Matrix, dict[str, Any]]:
    available = X.shape[1]
    if n_features >= available:
        ctx.project_with(lambda Z: Z, "row-wise")
        return X, {
            "n_features_in": int(available),
            "n_features_out": int(available),
            "note": f"requested {n_features} of {available} features; kept all",
        }

    # Variance is the only criterion (section 3.10): a feature's share of the expected
    # squared Euclidean distance is its share of the variance.
    variance = _feature_variance(X)
    keep = np.sort(np.argsort(variance)[::-1][:n_features])
    ctx.select_features(keep)
    ctx.project_with(lambda Z: Z[:, keep], "fitted parameters")
    return X[:, keep], {
        "criterion": "variance",
        "n_features_in": int(available),
        "n_features_out": int(keep.size),
        "score_threshold": float(variance[keep].min()),
    }


@executor("densify")
def densify(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    ctx.project_with(lambda Z: densify(Z, Context())[0], "row-wise")
    if not sp.issparse(X):
        return X, {"note": "input was already dense"}
    n, d = X.shape
    return np.asarray(X.todense(), dtype=np.float64), {
        "memory_gb": round(n * d * 8 / 1e9, 3)
    }


@executor("subsample")
def subsample(
    X: Matrix, ctx: Context, *, n_samples: int = 5000, **_: Any
) -> tuple[Matrix, dict[str, Any]]:
    """Stratified when labels are present, so small classes are not lost.

    The rows kept are the ones every later stage is fitted on. The pipeline engine
    projects the rest through the fitted stages, so the candidate's embedding still
    covers every row (section 3.12).
    """
    available = X.shape[0]
    if n_samples >= available:
        return X, {
            "n_samples_in": int(available),
            "n_samples_out": int(available),
            "note": f"requested {n_samples} of {available} samples; kept all",
        }

    rng = np.random.default_rng(ctx.seed)
    labels = ctx.labels
    if labels is None:
        index = np.sort(rng.choice(available, size=n_samples, replace=False))
        strategy = "uniform"
    else:
        classes, counts = np.unique(labels, return_counts=True)
        quota = np.maximum(1, np.floor(n_samples * counts / counts.sum()).astype(int))
        chosen = [
            rng.choice(
                np.flatnonzero(labels == cls),
                size=min(int(take), int(count)),
                replace=False,
            )
            for cls, take, count in zip(classes, quota, counts)
        ]
        index = np.sort(np.concatenate(chosen))
        strategy = "stratified by label"

    ctx.select_samples(index)
    return X[index], {
        "n_samples_in": int(available),
        "n_samples_out": int(index.size),
        "strategy": strategy,
        "seed": int(ctx.seed),
        "caveat": "later stages are fitted on these rows only; every other row is "
        "projected through the fitted stages, so the embedding covers the full dataset "
        "but was learned from fewer rows",
    }
