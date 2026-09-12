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


@executor("drop_constant")
def drop_constant(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    variance = _feature_variance(X)
    keep = np.flatnonzero(variance > 0)
    if keep.size == 0:
        raise ExecutionError(
            "every feature is constant; there is no variation to embed"
        )
    ctx.select_features(keep)
    return X[:, keep], {
        "n_features_in": int(X.shape[1]),
        "n_features_out": int(keep.size),
        "n_dropped": int(X.shape[1] - keep.size),
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
    scale = np.divide(
        chosen, totals, out=np.ones_like(totals, dtype=np.float64), where=totals > 0
    )
    scaled = sp.diags(scale) @ X if sp.issparse(X) else np.asarray(X) * scale[:, None]

    return scaled, {
        "target_total": chosen,
        "target_source": "specified" if target is not None else "median sample total",
        "n_zero_total_samples": int((totals == 0).sum()),
    }


@executor("log1p")
def log1p(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    values = X.data if sp.issparse(X) else X
    if np.any(values < 0):
        raise ExecutionError(
            "log1p requires non-negative input; this matrix contains negative values, "
            "so it has most likely already been centred or scaled"
        )
    return (X.log1p() if sp.issparse(X) else np.log1p(X)), {"base": "natural"}


@executor("standardise")
def standardise(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
    if sp.issparse(X):
        n, d = X.shape
        raise ExecutionError(
            "standardise centres the data, which makes every zero non-zero and would "
            f"turn this sparse matrix into {n * d * 8 / 1e9:.2f} GB dense. Select "
            "features first, or use a sparse-capable reduction instead."
        )
    dense = np.asarray(X, dtype=np.float64)
    std = dense.std(axis=0)
    n_constant = int((std == 0).sum())
    std[std == 0] = 1.0
    return (dense - dense.mean(axis=0)) / std, {
        "n_constant_features_left_unscaled": n_constant
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
    return scaled, {"norm": "l2"}


@executor("select_variable_features")
def select_variable_features(
    X: Matrix,
    ctx: Context,
    *,
    n_features: int = 2000,
    criterion: str = "variance",
    **_: Any,
) -> tuple[Matrix, dict[str, Any]]:
    available = X.shape[1]
    if n_features >= available:
        return X, {
            "n_features_in": int(available),
            "n_features_out": int(available),
            "note": f"requested {n_features} of {available} features; kept all",
        }

    variance = _feature_variance(X)
    if criterion == "dispersion":
        mean = (
            np.asarray(X.mean(axis=0)).ravel()
            if sp.issparse(X)
            else np.asarray(X).mean(axis=0)
        )
        score = np.divide(
            variance, mean, out=np.zeros_like(variance), where=mean > 0
        )
    elif criterion == "variance":
        score = variance
    else:  # pragma: no cover - the registry constrains this
        raise ExecutionError(f"unknown selection criterion {criterion!r}")

    keep = np.sort(np.argsort(score)[::-1][:n_features])
    ctx.select_features(keep)
    return X[:, keep], {
        "criterion": criterion,
        "n_features_in": int(available),
        "n_features_out": int(keep.size),
        "score_threshold": float(score[keep].min()),
    }


@executor("densify")
def densify(X: Matrix, ctx: Context, **_: Any) -> tuple[Matrix, dict[str, Any]]:
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
    """Stratified when labels are present, so small classes are not lost."""
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
        "caveat": "metrics computed after this stage describe the subsample, not the "
        "full dataset",
    }
