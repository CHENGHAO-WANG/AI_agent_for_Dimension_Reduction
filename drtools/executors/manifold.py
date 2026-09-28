"""Manifold reductions: MDS, Isomap, and locally linear embedding.

MDS sets no projection. It has no `transform` and no standard extension, so a
candidate cannot fit it on a subsample, and above its limit it is refused (section
3.12).

These are the methods with real preconditions, and the preconditions fail quietly.
Isomap on a disconnected graph produces infinite geodesics that scikit-learn patches
over, leaving an embedding that looks fine and means nothing. Modified LLE with too few
neighbours raises an error from deep inside a least-squares solve that names no
parameter the caller set. Each executor therefore checks what it needs up front and
says, in terms of the plan, what would fix it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.neighbors import kneighbors_graph

from drtools.constraints import lle_neighbour_minimum
from drtools.contract import Matrix
from drtools.executors import (
    Context,
    ExecutionError,
    executor,
    require_dense,
    require_pairwise_affordable,
)


@executor("mds")
def mds(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    metric: bool = True,
    n_init: int = 4,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    dense = require_dense(X, "mds")
    require_pairwise_affordable(dense.shape[0], "mds")

    # `metric_mds` and an explicit `init` are the current spellings; the older `metric`
    # and the implicit init both warn that they change meaning in scikit-learn 1.10.
    # Naming the initialisation also removes a source of run-to-run variation.
    model = MDS(
        n_components=n_components,
        metric_mds=metric,
        n_init=n_init,
        init="classical_mds",
        random_state=ctx.seed,
        normalized_stress="auto",
    )
    embedding = model.fit_transform(dense)
    if ctx.measure_criterion:
        # One value per fit: MDS is not nested in d, so tuning fits it at each d.
        ctx.criterion = {
            "kind": "stress",
            "value": 1.0 - kruskal_stress_1(dense, embedding),
        }

    return embedding, {
        "variant": "metric" if metric else "non-metric (ordinal)",
        "stress": float(model.stress_),
        "n_init": int(n_init),
        "n_iter": int(model.n_iter_),
        "caveat": "stress is the objective being minimised, not a goodness-of-fit that "
        "can be compared across datasets or across dimensionalities",
    }


@executor("isomap")
def isomap(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    n_neighbors: int = 10,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    dense = require_dense(X, "isomap")
    n_samples = dense.shape[0]
    require_pairwise_affordable(n_samples, "isomap")

    if n_neighbors >= n_samples:
        raise ExecutionError(
            f"n_neighbors={n_neighbors} is not smaller than the {n_samples} samples "
            "available"
        )

    n_parts, sizes = _graph_components(dense, n_neighbors)
    if n_parts > 1:
        raise ExecutionError(
            f"the k={n_neighbors} neighbourhood graph has {n_parts} disconnected "
            f"components (largest holds {int(sizes.max())} of {n_samples} points). "
            "Isomap measures distance as a path through this graph, so between "
            "components there is no path and the geodesic is infinite. Raise "
            f"n_neighbors above {n_neighbors} until the graph connects, or choose a "
            "method that does not need a connected graph."
        )

    model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    embedding = model.fit_transform(dense)
    # A new row's geodesic distance to every fitted row runs through its nearest fitted
    # neighbours, so each chunk costs a chunk-by-fitted-rows matrix.
    ctx.project_with(lambda Z: model.transform(require_dense(Z, "isomap")), "transform")
    if ctx.measure_criterion:
        ctx.criterion = {
            "kind": "residual_variance",
            "curve": _isomap_fit_curve(model.dist_matrix_, embedding),
        }

    return embedding, {
        "n_neighbors": int(n_neighbors),
        "graph_connected": True,
        "reconstruction_error": float(model.reconstruction_error()),
        "caveat": "a single spurious edge bridging two folds of the manifold "
        "short-circuits the geodesic and corrupts the whole embedding, not just its "
        "neighbourhood; a large n_neighbors makes that more likely",
    }


@executor("lle")
def lle(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    n_neighbors: int = 12,
    method: str = "standard",
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    dense = require_dense(X, "lle")
    n_samples = dense.shape[0]

    if n_neighbors >= n_samples:
        raise ExecutionError(
            f"n_neighbors={n_neighbors} is not smaller than the {n_samples} samples "
            "available"
        )

    # The same function day 12's validator refuses with, so a plan refused at
    # registration and a stage refused here are refused by one rule.
    minimum = lle_neighbour_minimum(method, n_components)
    if n_neighbors < minimum:
        raise ExecutionError(
            f"lle(method={method!r}) needs at least {minimum} neighbours to produce "
            f"{n_components} components, but the plan asks for {n_neighbors}. The "
            f"{method} variant solves a local problem whose size grows with the output "
            "dimension; raise n_neighbors or lower n_components."
        )

    n_parts, sizes = _graph_components(dense, n_neighbors)
    if n_parts > 1:
        raise ExecutionError(
            f"the k={n_neighbors} neighbourhood graph has {n_parts} disconnected "
            f"components (largest holds {int(sizes.max())} of {n_samples} points). "
            "LLE solves a single eigenproblem over the whole graph, so a disconnected "
            "one yields component indicators rather than coordinates. Raise "
            f"n_neighbors above {n_neighbors} until it connects."
        )

    model = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        method=method,
        random_state=ctx.seed,
        eigen_solver="dense" if n_samples <= 2000 else "auto",
    )
    embedding = model.fit_transform(dense)
    # scikit-learn places a new row by its barycentric weights on its nearest fitted
    # rows, whichever variant was fitted.
    ctx.project_with(lambda Z: model.transform(require_dense(Z, "lle")), "transform")

    # LLE's characteristic failure is collapse: whole regions squashed toward a point,
    # which looks like tight clustering. Comparing the spread of the coordinates catches
    # it, since a healthy embedding uses its dimensions comparably.
    spread = embedding.std(axis=0)
    collapse_ratio = float(spread.max() / spread.min()) if spread.min() > 0 else None

    notes = {
        "variant": method,
        "n_neighbors": int(n_neighbors),
        "reconstruction_error": float(model.reconstruction_error_),
        "coordinate_spread_ratio": collapse_ratio,
    }
    if collapse_ratio is not None and collapse_ratio > 50:
        notes["warning"] = (
            f"coordinate spreads differ by a factor of {collapse_ratio:,.0f}, which "
            "usually means the embedding has partially collapsed — a known failure of "
            "standard LLE when the local weight problem is ill-conditioned. The "
            "modified variant is the usual remedy."
        )
    return embedding, notes


def kruskal_stress_1(dense: np.ndarray, embedding: np.ndarray) -> float:
    """Kruskal's stress-1 against the input distances, which metric MDS fits directly.

    sqrt(sum (delta - d)^2 / sum d^2), over pairs. Normalised, unlike the SMACOF
    objective the executor reports, so it can be compared across d (section 3.7).
    """
    from scipy.spatial.distance import pdist

    delta, fitted = pdist(dense), pdist(embedding)
    denominator = float((fitted**2).sum())
    return float(np.sqrt(((delta - fitted) ** 2).sum() / denominator)) if denominator else 1.0


def _isomap_fit_curve(geodesic: np.ndarray, embedding: np.ndarray) -> list[float]:
    """Q(d) = 1 - residual variance at every d up to the fit's (Tenenbaum et al.).

    The residual variance is 1 - R^2 between the graph's geodesic distances and the
    embedding's Euclidean distances. The squared distances grow one coordinate at a
    time, so the whole curve costs one pass over the pairs per coordinate.
    """
    upper = np.triu_indices(geodesic.shape[0], k=1)
    target = geodesic[upper]
    squared = np.zeros_like(target)
    curve: list[float] = []
    for column in embedding.T:
        squared += (column[upper[0]] - column[upper[1]]) ** 2
        r = np.corrcoef(target, np.sqrt(squared))[0, 1]
        curve.append(float(r * r) if np.isfinite(r) else 0.0)
    return curve


def _graph_components(X: np.ndarray, n_neighbors: int) -> tuple[int, np.ndarray]:
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity")
    n_parts, membership = connected_components(graph.maximum(graph.T), directed=False)
    _, sizes = np.unique(membership, return_counts=True)
    return n_parts, sizes
