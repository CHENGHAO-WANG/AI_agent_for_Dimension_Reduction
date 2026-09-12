"""Neighbour embeddings: t-SNE and UMAP.

These are the methods most likely to be over-read. Both produce pictures that look like
maps and are not: cluster sizes and inter-cluster distances are artefacts of the
optimisation rather than properties of the data. The registry says so and the report
repeats it, but the notes returned here also carry the caveat, so it travels with the
result rather than depending on someone remembering to attach it.

Both are also the methods whose defaults are most often wrong. scikit-learn's t-SNE
defaults to perplexity 30 regardless of sample count, which is meaningless at n = 100,
so the executor checks the ratio and refuses rather than producing a degenerate
embedding that still plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix
from drtools.executors import Context, ExecutionError, executor, require_dense


@executor("tsne")
def tsne(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    initialization: str = "pca",
    n_iter: int = 500,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    from openTSNE import TSNE

    dense = require_dense(X, "tsne")
    n_samples = dense.shape[0]

    # Perplexity is an effective neighbourhood size. Once it approaches n/3 the
    # "neighbourhood" is most of the dataset, the conditional distributions flatten,
    # and the result is a disc of points that still plots perfectly happily.
    if perplexity >= n_samples / 3:
        suggested = max(5.0, round(n_samples / 100) * 5.0)
        raise ExecutionError(
            f"perplexity={perplexity:g} is not usable with {n_samples} samples: it is "
            f"at or above n/3 = {n_samples / 3:.0f}, so each point's neighbourhood "
            "spans most of the data and the embedding degenerates into a featureless "
            f"blob. Try perplexity around {suggested:g} for this sample size."
        )

    model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        initialization=initialization,
        n_iter=n_iter,
        random_state=ctx.seed,
        verbose=False,
    )
    embedding = np.asarray(model.fit(dense))

    return embedding, {
        "perplexity": float(perplexity),
        "perplexity_as_fraction_of_n": float(perplexity / n_samples),
        "initialization": initialization,
        "n_iter": int(n_iter),
        "implementation": "openTSNE (FFT-accelerated)",
        "caveat": "cluster sizes and the distances between clusters are not "
        "interpretable; only which points are near which other points is. Apparent "
        "gaps may be optimisation artefacts rather than structure.",
    }


@executor("umap")
def umap(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    import umap as umap_module

    n_samples = X.shape[0]
    if n_neighbors >= n_samples:
        raise ExecutionError(
            f"n_neighbors={n_neighbors} is not smaller than the {n_samples} samples "
            "available"
        )
    if metric == "correlation" and sp.issparse(X):
        raise ExecutionError(
            "umap's correlation metric is not implemented for sparse input; add a "
            "densify stage or use the cosine metric, which behaves similarly on "
            "non-negative data"
        )

    if sp.issparse(X) and not isinstance(X, sp.spmatrix):
        # UMAP's inner loops are numba-compiled, and numba understands only scipy's
        # legacy sparse *matrix* types. Handed the newer sparse *array* — which is what
        # scipy's own operations increasingly return — type inference fails with a
        # message about "non-precise type pyobject" that says nothing about sparsity.
        # The two are the same data, so converting is free and keeps the registry's
        # handles_sparse claim honest.
        X = sp.csr_matrix(X)

    model = umap_module.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=ctx.seed,
        verbose=False,
    )
    embedding = np.asarray(model.fit_transform(X))

    return embedding, {
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "determinism_note": "a fixed random_state makes UMAP reproducible at the cost "
        "of single-threaded optimisation; it will be slower than its parallel default",
        "caveat": "inter-cluster distances carry more information than t-SNE's but are "
        "still not metric, and min_dist is a display choice that changes how tightly "
        "points pack without changing what the data is like",
    }
