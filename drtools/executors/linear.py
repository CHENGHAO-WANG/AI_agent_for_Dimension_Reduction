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
        notes = {"solver": "PCA", "centred": True, "whitened": bool(whiten)}

    ratios = np.asarray(model.explained_variance_ratio_)
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
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    dense = require_dense(X, "kernel_pca")
    require_pairwise_affordable(dense.shape[0], "kernel_pca")

    # scikit-learn's default gamma of 1/n_features ignores the scale of the data
    # entirely. Tying it to the median pairwise distance at least puts the kernel
    # width in the range where the kernel actually varies.
    gamma_source = "specified"
    if gamma is None and kernel in {"rbf", "sigmoid"}:
        gamma = _median_heuristic_gamma(dense, ctx.seed)
        gamma_source = "median pairwise distance heuristic"

    model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        random_state=ctx.seed,
        eigen_solver="auto",
    )
    embedding = model.fit_transform(dense)
    if embedding.shape[1] < n_components:
        raise ExecutionError(
            f"kernel_pca returned {embedding.shape[1]} of {n_components} requested "
            "components; the kernel matrix is rank-deficient, which usually means the "
            "bandwidth is far too small and every point looks equally dissimilar"
        )
    return embedding, {
        "kernel": kernel,
        "gamma": float(gamma) if gamma is not None else None,
        "gamma_source": gamma_source if kernel in {"rbf", "sigmoid"} else "not applicable",
    }


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

    loadings = np.asarray(model.components_)
    nonzero = int(np.count_nonzero(loadings))
    return embedding, {
        "alpha": float(alpha),
        "loading_sparsity": float(1.0 - nonzero / loadings.size),
        "n_nonzero_loadings": nonzero,
        "caveat": "components are neither orthogonal nor ordered by explained variance, "
        "so they cannot be interpreted the way PCA's components are",
    }


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
