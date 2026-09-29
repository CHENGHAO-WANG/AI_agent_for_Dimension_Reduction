"""Linear reductions: PCA and its variants.

PCA carries the baseline's weight in every run. When a neighbour embedding beats it,
that difference is the evidence that the data is genuinely nonlinear; when it does not,
the honest conclusion is that the extra machinery bought nothing. Either way the
comparison only means something if the baseline is computed properly, which on sparse
input means being explicit that TruncatedSVD does not centre.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, KernelPCA, MiniBatchSparsePCA, TruncatedSVD

from drtools.contract import Matrix
from drtools.executors import (
    Context,
    ExecutionError,
    executor,
    require_dense,
    require_pairwise_affordable,
)


@executor("pca")
def pca(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    whiten: bool = False,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_samples, n_features = X.shape
    limit = min(n_samples, n_features)
    if n_components > limit:
        raise ExecutionError(
            f"pca cannot produce {n_components} components from a "
            f"{n_samples}x{n_features} matrix; at most {limit} exist"
        )

    if sp.issparse(X):
        # TruncatedSVD is the sparse-safe route, but it is an SVD of the uncentred
        # matrix rather than a PCA, and the difference is worth stating: the leading
        # component of uncentred data often tracks sample magnitude.
        if n_components >= limit:
            raise ExecutionError(
                f"sparse input uses TruncatedSVD, which needs strictly fewer than "
                f"{limit} components; requested {n_components}"
            )
        model = TruncatedSVD(
            n_components=n_components, random_state=ctx.seed, algorithm="randomized"
        )
        embedding = model.fit_transform(X)
        ctx.project_with(model.transform, "transform")
        ctx.keep("components", model.components_)
        notes = {
            "solver": "TruncatedSVD",
            "centred": False,
            "caveat": "sparse input was decomposed without centring, so this is an SVD "
            "of the uncentred matrix; the first component may reflect sample magnitude "
            "rather than contrast between samples",
        }
    else:
        model = PCA(n_components=n_components, whiten=whiten, random_state=ctx.seed)
        embedding = model.fit_transform(np.asarray(X, dtype=np.float64))
        ctx.project_with(lambda Z: model.transform(_dense_float(Z)), "transform")
        ctx.keep("components", model.components_)
        notes = {"solver": "PCA", "centred": True, "whitened": bool(whiten)}

    ratios = np.asarray(model.explained_variance_ratio_)
    if ctx.measure_criterion:
        # Q(d) at every d up to the fit's: the share of the variance the first d keep.
        ctx.criterion = {
            "kind": "explained_variance",
            "curve": [float(v) for v in np.cumsum(ratios)],
        }
    notes.update(
        {
            "explained_variance_ratio": [float(r) for r in ratios],
            "cumulative_explained_variance": float(ratios.sum()),
        }
    )
    return embedding, notes


@executor("kernel_pca")
def kernel_pca(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    kernel: str = "rbf",
    gamma: float | None = None,
    width_multiplier: float = 1.0,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    dense = require_dense(X, "kernel_pca")
    require_pairwise_affordable(dense.shape[0], "kernel_pca")

    # scikit-learn's default gamma of 1/n_features ignores the scale of the data
    # entirely. Tying it to the median pairwise distance at least puts the kernel
    # width in the range where the kernel actually varies. The polynomial kernel keeps
    # scikit-learn's default, and the cosine kernel has no width at all. Each label is
    # the registry's default_rule for that kernel, and a test holds the two together.
    recorded_gamma = gamma
    if kernel == "cosine":
        gamma_source = "not applicable"
    elif gamma is not None:
        gamma_source = "specified"
    elif kernel in {"rbf", "sigmoid"}:
        gamma = recorded_gamma = _median_heuristic_gamma(dense, ctx.seed)
        gamma_source = "median pairwise distance heuristic"
    else:
        gamma = recorded_gamma = 1.0 / dense.shape[1]
        gamma_source = "1/n_features (scikit-learn default)"
    # Tuning scales the width, from the rule or an explicit value alike, on the rows
    # this fit sees (section 3.5).
    if gamma is not None and width_multiplier != 1.0:
        gamma = recorded_gamma = float(gamma) * float(width_multiplier)

    model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        random_state=ctx.seed,
        eigen_solver="auto",
    )
    embedding = model.fit_transform(dense)
    # The kernel between a chunk of new rows and every fitted row is built whole, which
    # is why the engine sizes its chunks by the number of fitted rows.
    ctx.project_with(lambda Z: model.transform(_dense_float(Z)), "transform")
    if embedding.shape[1] < n_components:
        raise ExecutionError(
            f"kernel_pca returned {embedding.shape[1]} of {n_components} requested "
            "components; the kernel matrix is rank-deficient, which usually means the "
            "bandwidth is far too small and every point looks equally dissimilar"
        )
    if ctx.measure_criterion:
        ctx.criterion = {
            "kind": "kernel_variance",
            "curve": _kernel_variance_curve(model, dense, kernel, gamma),
        }
    return embedding, {
        "kernel": kernel,
        "gamma": float(recorded_gamma) if recorded_gamma is not None else None,
        "gamma_source": gamma_source,
        "width_multiplier": float(width_multiplier),
    }


def _kernel_variance_curve(
    model: Any, dense: np.ndarray, kernel: str, gamma: float | None
) -> list[float]:
    """Q(d): the share of the centred kernel's trace the first d eigenvalues carry.

    The trace of the centred kernel is the total variance in feature space, so this is
    the exact kernel counterpart of PCA's explained variance.
    """
    from sklearn.metrics.pairwise import pairwise_kernels

    options = {} if kernel == "cosine" else {"gamma": gamma}
    matrix = pairwise_kernels(dense, metric=kernel, **options)
    total = float(np.trace(matrix) - matrix.sum() / matrix.shape[0])
    eigenvalues = np.clip(np.asarray(model.eigenvalues_, dtype=np.float64), 0.0, None)
    return [float(v) for v in np.cumsum(eigenvalues) / total] if total > 0 else []


@executor("sparse_pca")
def sparse_pca(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    alpha: float = 1.0,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    dense = require_dense(X, "sparse_pca")

    model = MiniBatchSparsePCA(
        n_components=n_components, alpha=alpha, random_state=ctx.seed
    )
    embedding = model.fit_transform(dense)
    ctx.project_with(lambda Z: model.transform(_dense_float(Z)), "transform")

    loadings = np.asarray(model.components_)
    ctx.keep("components", loadings)
    nonzero = int(np.count_nonzero(loadings))
    return embedding, {
        "alpha": float(alpha),
        "loading_sparsity": float(1.0 - nonzero / loadings.size),
        "n_nonzero_loadings": nonzero,
        "caveat": "components are neither orthogonal nor ordered by explained variance, "
        "so they cannot be interpreted the way PCA's components are",
    }


def _dense_float(Z: Matrix) -> np.ndarray:
    """A chunk of new rows in the dense float64 form these methods were fitted on."""
    return np.asarray(Z.todense() if sp.issparse(Z) else Z, dtype=np.float64)


def _median_heuristic_gamma(X: np.ndarray, seed: int, cap: int = 1000) -> float:
    """gamma = 1 / median squared pairwise distance, on a capped random subsample."""
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(seed)
    sample = (
        X
        if X.shape[0] <= cap
        else X[rng.choice(X.shape[0], size=cap, replace=False)]
    )
    distances = pairwise_distances(sample, squared=True)
    median = float(np.median(distances[distances > 0])) if distances.any() else 1.0
    return 1.0 / median if median > 0 else 1.0
