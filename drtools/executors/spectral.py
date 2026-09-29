"""Spectral reductions: Laplacian Eigenmaps and Diffusion Maps.

Both read structure off the eigenvectors of an operator built from local similarities,
and both are defined on a connected graph. On a fragmented graph they do not error —
they return indicator functions of the components, which plot as a handful of tight
dots and look like a wonderful clustering. That failure mode is the reason the recon
pass measures connectivity before the planner ever selects them, and the reason these
executors check it again and say so rather than returning something plausible.

Neither method ships a way to place rows it was not fitted on, so both get the Nyström
extension here (section 3.12): an eigenvector satisfies mu * psi = P psi for the
random-walk operator P, and the same equation, with P's row for a new point computed
from its similarities to the fitted rows, defines psi at that point.

Diffusion Maps is implemented here rather than taken from a library. `datafold`, the
obvious dependency, imports a scikit-learn private symbol that no longer exists, and
pinning scikit-learn backwards to suit it would cascade through umap-learn and scanpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from drtools.contract import Matrix
from drtools.executors import (
    Context,
    ExecutionError,
    executor,
    require_dense,
    require_pairwise_affordable,
)


@executor("laplacian_eigenmaps")
def laplacian_eigenmaps(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_samples = X.shape[0]
    if n_neighbors >= n_samples:
        raise ExecutionError(
            f"n_neighbors={n_neighbors} is not smaller than the {n_samples} samples "
            "available"
        )

    n_parts, sizes = _graph_components(X, n_neighbors)
    if n_parts > 1:
        raise ExecutionError(
            f"the k={n_neighbors} neighbourhood graph has {n_parts} disconnected "
            f"components (largest holds {sizes.max()} of {n_samples} points). "
            "Laplacian Eigenmaps on a disconnected graph returns indicator functions "
            "of the components, which look like a clean clustering but encode nothing "
            f"about within-component structure. Raise n_neighbors above {n_neighbors} "
            "until the graph connects, or choose a method that does not need a "
            "connected graph."
        )

    model = SpectralEmbedding(
        n_components=n_components,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=ctx.seed,
    )
    embedding = model.fit_transform(X)
    extension = LaplacianNystrom(X, embedding, model.affinity_matrix_, n_neighbors)
    ctx.project_with(extension, "nystrom")
    if ctx.measure_criterion:
        # The random-walk eigenvalues of the fitted coordinates, largest first; the
        # eigengap rule reads them, so tuning fits one coordinate more than d_max.
        ctx.criterion = {
            "kind": "eigengap",
            "eigenvalues": [float(v) for v in extension.eigenvalues],
        }
    return embedding, {
        "n_neighbors": int(n_neighbors),
        "graph_connected": True,
        "eigenvalues": [float(v) for v in extension.eigenvalues],
        "caveat": "the embedding's scale is arbitrary and inter-cluster distances are "
        "not meaningful; only local neighbourhood structure is represented",
    }


@executor("diffusion_maps")
def diffusion_maps(
    X: Matrix,
    ctx: Context,
    *,
    n_components: int = 2,
    epsilon: float | None = None,
    alpha: float = 1.0,
    t: int = 1,
    width_multiplier: float = 1.0,
    **_: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coifman and Lafon's diffusion maps.

    Gaussian kernel, density normalisation by alpha, row-normalisation to a Markov
    operator, then the leading non-trivial eigenvectors scaled by their eigenvalues
    raised to the diffusion time.

    The eigendecomposition is done on the symmetric conjugate of the Markov matrix
    rather than on the Markov matrix itself. They share eigenvalues, but the conjugate
    is symmetric, so `eigh` gives real, ordered, numerically stable results where a
    general solver on the non-symmetric operator would return complex values with
    tiny imaginary parts that then have to be discarded by hand.
    """
    dense = require_dense(X, "diffusion_maps")
    n_samples = dense.shape[0]
    require_pairwise_affordable(n_samples, "diffusion_maps")

    if n_components >= n_samples:
        raise ExecutionError(
            f"diffusion_maps cannot produce {n_components} non-trivial components from "
            f"{n_samples} samples"
        )

    squared_distances = pairwise_distances(dense, squared=True)

    epsilon_source = "specified"
    estimated_dimension = None
    rule_epsilon, floor_doublings = None, 0
    if epsilon is None:
        rule_epsilon, estimated_dimension = _bandwidth_by_kernel_scaling(
            squared_distances, ctx.seed
        )
        epsilon, floor_doublings = _connectivity_floor(squared_distances, rule_epsilon, alpha)
        epsilon_source = "kernel-sum scaling criterion (Coifman and Singer)"
    # Tuning scales the bandwidth, from the criterion or an explicit value alike, on the
    # rows this fit sees (section 3.5).
    epsilon = float(epsilon) * float(width_multiplier)
    if epsilon <= 0:
        raise ExecutionError(
            "the kernel bandwidth resolved to zero, which happens when most points are "
            "exact duplicates; deduplicate before embedding"
        )

    eigenvalues, eigenvectors, inverse_sqrt_degree, density = _diffusion_spectrum(
        squared_distances, epsilon, alpha
    )
    support = _support(eigenvectors)
    if _isolates(eigenvalues, support):
        found = (
            f"its leading eigenvalue repeats to within {SPLIT_GAP:g}, so the kernel "
            "falls apart into separate pieces"
            if _splits(eigenvalues)
            else f"{int((support < LOCALISED_SUPPORT).sum())} of its {support.size} "
            f"leading coordinates live on fewer than {LOCALISED_SUPPORT} samples"
        )
        raise ExecutionError(
            f"diffusion_maps at epsilon = {epsilon:.4g}: {found}, so the kernel "
            "isolates individual samples and each takes a coordinate of its own. The bandwidth is too narrow "
            "for this data: widen it (a larger width_multiplier or epsilon), or leave "
            "epsilon unset so the connectivity floor sets it."
        )

    # Back to the right eigenvectors of the Markov operator. The first is the constant
    # stationary vector and carries no information, so the coordinates start at one.
    psi = eigenvectors * inverse_sqrt_degree[:, None]
    kept = slice(1, n_components + 1)
    embedding = psi[:, kept] * np.power(eigenvalues[kept], t)
    ctx.project_with(
        DiffusionNystrom(
            dense, psi[:, kept], eigenvalues[kept], density, epsilon, alpha, t
        ),
        "nystrom",
    )
    if ctx.measure_criterion:
        # Every non-trivial eigenvalue, not only the kept ones: the share of diffusion
        # distance d coordinates retain is a share of all of them (section 3.7).
        ctx.criterion = {
            "kind": "diffusion_distance",
            "eigenvalues": [float(v) for v in eigenvalues[1:]],
            "t": int(t),
        }

    spectral_gap = (
        float(eigenvalues[1] - eigenvalues[2]) if eigenvalues.size > 2 else None
    )
    return np.ascontiguousarray(embedding), {
        "epsilon": float(epsilon),
        "epsilon_source": epsilon_source,
        "epsilon_from_rule": None if rule_epsilon is None else float(rule_epsilon),
        "connectivity_floor_doublings": int(floor_doublings),
        "localised_leading_coordinates": int((support < LOCALISED_SUPPORT).sum()),
        "dimension_implied_by_bandwidth": estimated_dimension,
        "alpha": float(alpha),
        "t": int(t),
        "width_multiplier": float(width_multiplier),
        "eigenvalues": [float(v) for v in eigenvalues[: n_components + 1]],
        "spectral_gap_after_first_coordinate": spectral_gap,
        "caveat": "the geometry depends on the diffusion time t; a different t gives a "
        "different and equally valid embedding, so t belongs in any description of "
        "this result",
    }


class LaplacianNystrom:
    """Place new rows in a fitted Laplacian Eigenmaps embedding.

    scikit-learn's coordinates y satisfy mu * y_i = sum_j W_ij y_j / d_i, where W is its
    symmetrised k-nearest-neighbour affinity with the self-loops removed -- the graph
    Laplacian ignores the diagonal -- and d_i is W's row sum. mu is recovered from each
    coordinate as a Rayleigh quotient. A new row x gets the same equation's value,
    mu * y(x) = sum_j w(x, j) y_j / sum_j w(x, j), with w the fitted affinity evaluated
    at x: 1/2 for each fitted row among x's k - 1 nearest (k counts the point itself, as
    in the fit), and 1/2 more where x falls inside that fitted row's own neighbourhood,
    the distance to its k - 1-th nearest other row.

    Because the affinity is a 0/1 graph and mu is close to 1 for the leading
    coordinates, this lands close to the mean of the new row's neighbours. That is what
    the standard extension is for this kernel, rather than a stand-in for one.
    """

    def __init__(
        self,
        X_fit: Matrix,
        embedding: np.ndarray,
        affinity: Any,
        n_neighbors: int,
    ) -> None:
        weights = sp.csr_matrix(affinity, dtype=np.float64).tolil()
        weights.setdiag(0.0)
        weights = weights.tocsr()
        degree = np.asarray(weights.sum(axis=1)).ravel()
        self.embedding = np.asarray(embedding, dtype=np.float64)
        self.eigenvalues = np.array(
            [
                float(y @ (weights @ y)) / float(y @ (degree * y))
                for y in self.embedding.T
            ]
        )
        self.X_fit = X_fit
        self.n_others = n_neighbors - 1
        self.index = NearestNeighbors(n_neighbors=n_neighbors).fit(X_fit)
        # Each fitted row's own neighbourhood radius: the distance to its k - 1-th
        # nearest other row. The tree search and `pairwise_distances` round differently,
        # and a fitted row that is exactly another's last neighbour sits on the radius,
        # so the radius is widened by a relative 1e-9 to count it on both sides.
        distances, _ = self.index.kneighbors(X_fit)
        self.radius = distances[:, -1] * (1.0 + 1e-9)

    def kernel(self, distances: np.ndarray) -> np.ndarray:
        """w(x, j) for each row of `distances`, a chunk against every fitted row."""
        nearest = np.argpartition(distances, self.n_others - 1, axis=1)[
            :, : self.n_others
        ]
        forward = np.zeros_like(distances)
        np.put_along_axis(forward, nearest, 1.0, axis=1)
        backward = (distances <= self.radius[None, :]).astype(np.float64)
        return 0.5 * (forward + backward)

    def __call__(self, Z: Matrix) -> np.ndarray:
        distances = pairwise_distances(Z, self.X_fit)
        weights = self.kernel(distances)
        mean = weights @ self.embedding / weights.sum(axis=1, keepdims=True)
        return mean / self.eigenvalues[None, :]


class DiffusionNystrom:
    """Place new rows in a fitted diffusion map.

    The fitted coordinates are psi * lambda**t, where psi are right eigenvectors of the
    Markov operator P built from the density-normalised kernel. For a new row x, P's
    row is the fitted kernel against every fitted row, normalised as in the fit with
    the fitted rows' densities held fixed. Then psi(x) = sum_j P(x, j) psi_j / lambda,
    and the coordinate is psi(x) * lambda**t. Applied to a fitted row, this returns
    that row's coordinate, which is the test.

    The density at x divides every entry of its row alike and cancels when the row is
    normalised, so P(x, .) is a softmax of -d^2 / epsilon - alpha * log q_j. Computed
    that way, a row far from every fitted row still gets a row that sums to one rather
    than a kernel that underflowed to zero.
    """

    def __init__(
        self,
        X_fit: np.ndarray,
        psi: np.ndarray,
        eigenvalues: np.ndarray,
        density: np.ndarray,
        epsilon: float,
        alpha: float,
        t: int,
    ) -> None:
        self.X_fit = X_fit
        self.psi = np.asarray(psi, dtype=np.float64)
        self.eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self.density = density
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.t = int(t)

    def __call__(self, Z: Matrix) -> np.ndarray:
        dense = require_dense(Z, "diffusion_maps")
        log_kernel = -pairwise_distances(dense, self.X_fit, squared=True) / self.epsilon
        log_kernel -= self.alpha * np.log(self.density)[None, :]
        log_kernel -= log_kernel.max(axis=1, keepdims=True)
        markov = np.exp(log_kernel)
        markov /= markov.sum(axis=1, keepdims=True)
        psi = markov @ self.psi / self.eigenvalues[None, :]
        return psi * np.power(self.eigenvalues, self.t)[None, :]


def _bandwidth_by_kernel_scaling(
    squared_distances: np.ndarray, seed: int, n_grid: int = 60, cap: int = 800
) -> tuple[float, float | None]:
    """Choose the bandwidth by the kernel-sum scaling criterion.

    The obvious heuristic — the median distance to the k-th neighbour — fails in a way
    that is easy to miss. It scales with how densely the data happens to be sampled,
    while the gaps a manifold method must *not* bridge do not. On a sparsely sampled
    Swiss roll the k-th neighbour is far enough away that the kernel reaches across
    adjacent sheets, diffusion short-circuits between them, and the leading coordinates
    stop tracking position along the roll. Measured here, that heuristic recovered the
    roll parameter at rho = 1.00 for n = 1000 but only 0.22 for n = 400.

    Coifman and Singer's criterion has no such dependence. Sum the kernel over all pairs
    for a range of bandwidths and look at log S(eps) against log eps: the curve is flat
    at both ends — every point isolated, or every point mutually adjacent — and linear
    in between, where the kernel resolves the manifold. The slope there is half the
    intrinsic dimension, which comes free and is worth reporting as a check on the
    two-NN estimate from reconnaissance.

    Which point of that regime to take is the part worth stating, because the steepest
    point is the wrong one. It maximises the dimension estimate, but on this data it
    sits at eps ~ 33 regardless of n, well above the range that works: measured across
    a bandwidth sweep, the Swiss roll unrolls at rho >= 0.99 for eps in 0.5 to 4 and
    collapses to rho < 0.4 by eps = 8, because a wide kernel reaches across adjacent
    sheets and diffusion short-circuits between them. So the bandwidth taken here is
    the *lower edge* of the linear regime — the smallest eps whose slope has reached
    half the maximum — which is where the kernel has just begun to resolve the manifold
    rather than isolating every point. That lands at 3.1 for n = 300 and 0.5 for
    n = 1500, inside the working range at both ends.

    The search runs on a capped random submatrix, since it is O(n_grid * n^2) and only
    needs to locate a scale.
    """
    n_samples = squared_distances.shape[0]
    if n_samples > cap:
        rng = np.random.default_rng(seed)
        index = rng.choice(n_samples, size=cap, replace=False)
        squared_distances = squared_distances[np.ix_(index, index)]

    positive = squared_distances[squared_distances > 0]
    if positive.size == 0:
        return 0.0, None

    low, high = np.percentile(positive, [1, 99])
    grid = np.geomspace(max(low, np.finfo(float).tiny) / 100, high * 10, n_grid)
    kernel_sums = np.array(
        [float(np.exp(-squared_distances / bandwidth).sum()) for bandwidth in grid]
    )

    log_grid, log_sums = np.log(grid), np.log(kernel_sums)
    slopes = np.gradient(log_sums, log_grid)
    steepest = float(slopes.max())
    if steepest <= 0:
        return float(np.median(positive)), None

    in_regime = np.flatnonzero(slopes >= 0.5 * steepest)
    return float(grid[in_regime[0]]), float(2.0 * steepest)


#: A coordinate living on fewer samples than this describes individual samples, not
#: structure. The kernel isolates samples when more than a quarter of its leading
#: coordinates are like that. Measured on day 20 among the leading 20: PBMC3k at the
#: rule's bandwidth had 19, at eight times it 11, at 32 times 2; a Swiss roll of 400
#: rows had 2 -- an isolated sample at a thin end -- and still unrolled.
LOCALISED_SUPPORT = 5
LOCALISED_SHARE = 0.25
SUPPORT_CHECKED = 20
#: A leading eigenvalue repeated to within this splits the kernel into numerically
#: separate pieces, each owning an eigenvalue of 1, and leaves the eigenvectors of that
#: eigenspace an arbitrary rotation, so support cannot be read there at all. Found on day
#: 21: on PathMNIST 23 samples had every kernel entry below 1e-16, the gap was 2e-15,
#: and the fit counted 4 localised coordinates where the same kernel reads 2 or 20. A
#: resolved Swiss roll's slow diffusion gaps near 1e-4.
SPLIT_GAP = 1e-10
MAX_FLOOR_DOUBLINGS = 12
#: Up to this many rows the floor searches with the full solver the fit uses. Near-equal
#: eigenvalues leave their eigenvectors defined only up to a rotation, and support is
#: not rotation-invariant, so two solvers can disagree about the same kernel.
FULL_SOLVE_LIMIT = 4000


def _diffusion_spectrum(
    squared_distances: np.ndarray, epsilon: float, alpha: float, leading: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of the symmetric conjugate of the diffusion operator.

    Gaussian kernel; density normalisation by alpha, where alpha = 1 divides out the
    sampling density and recovers the Laplace-Beltrami operator of the manifold; then
    the symmetric conjugate of the row-normalised Markov matrix, in decreasing order.
    `leading` computes only that many, for the floor's search.
    """
    kernel = np.exp(-squared_distances / epsilon)
    density = kernel.sum(axis=1)
    if alpha:
        scaling = np.power(density, -alpha)
        kernel = kernel * np.outer(scaling, scaling)
    degree = kernel.sum(axis=1)
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    symmetric = kernel * np.outer(inverse_sqrt_degree, inverse_sqrt_degree)
    symmetric = (symmetric + symmetric.T) / 2.0
    n = symmetric.shape[0]
    if leading is not None and leading < n:
        # A direct solver for the top few: an iterative one stalls on the cluster of
        # eigenvalues near 1 that a narrow kernel produces, which is when this runs.
        from scipy.linalg import eigh

        eigenvalues, eigenvectors = eigh(symmetric, subset_by_index=[n - leading, n - 1])
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order], inverse_sqrt_degree, density


def _support(eigenvectors: np.ndarray) -> np.ndarray:
    """How many samples each leading non-trivial coordinate effectively lives on.

    The inverse participation ratio of a unit eigenvector: 1 for a coordinate on one
    sample, n for one spread evenly over all of them. Read on the leading coordinates
    whatever d the fit delivers, since it describes the kernel, not the embedding.
    """
    leading = eigenvectors[:, 1 : 1 + SUPPORT_CHECKED]
    return 1.0 / np.sum(leading**4, axis=0)


def _splits(eigenvalues: np.ndarray) -> bool:
    return bool(eigenvalues[0] - eigenvalues[1] < SPLIT_GAP)


def _isolates(eigenvalues: np.ndarray, support: np.ndarray) -> bool:
    return _splits(eigenvalues) or bool(
        (support < LOCALISED_SUPPORT).sum() > LOCALISED_SHARE * support.size
    )


def _connectivity_floor(
    squared_distances: np.ndarray, epsilon: float, alpha: float
) -> tuple[float, int]:
    """The rule's bandwidth, doubled until the kernel stops isolating samples.

    The rule reads the kernel sum, which in many noisy dimensions can settle below the
    distance from a typical sample to its nearest neighbour: on PBMC3k's 83 principal
    components it chose 58 against a median nearest squared distance of 232, and every
    leading eigenvalue was 1. Searched on the fit's own rows: a subsample holds fewer of
    the isolated samples, so it passes a bandwidth the whole fails -- on PBMC3k 16 times
    the rule against 32.
    """
    # ponytail: above FULL_SOLVE_LIMIT the search uses a partial solver, which can pass a
    # width the fit's full solver then refuses; the refusal still holds, at the cost of a
    # failed Attempt. Search on the full solver there too if that is seen.
    leading = None if squared_distances.shape[0] <= FULL_SOLVE_LIMIT else SUPPORT_CHECKED + 1
    for doublings in range(MAX_FLOOR_DOUBLINGS + 1):
        width = epsilon * 2.0**doublings
        values, vectors, _, _ = _diffusion_spectrum(
            squared_distances, width, alpha, leading=leading
        )
        if not _isolates(values, _support(vectors)):
            return width, doublings
    raise ExecutionError(
        f"diffusion_maps: no bandwidth up to {2**MAX_FLOOR_DOUBLINGS} times the rule's "
        f"{epsilon:.4g} stops the kernel isolating individual samples. The data may be "
        "dominated by noise dimensions; reduce with PCA to fewer components first."
    )


def _graph_components(X: Matrix, n_neighbors: int) -> tuple[int, np.ndarray]:
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity")
    n_parts, membership = connected_components(
        graph.maximum(graph.T), directed=False
    )
    _, sizes = np.unique(membership, return_counts=True)
    return n_parts, sizes
