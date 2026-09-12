"""Synthetic datasets with known structure.

Two jobs. They are the daily smoke test — the whole pipeline runs on them in
seconds, so breakage is caught the day it is introduced. And because their
structure is known by construction, they are the substrate for the agent-behaviour
tests: a Swiss roll *should* lead the agent to manifold methods and *should* show
Isomap beating PCA, while isotropic blobs should show no manifold advantage at all.

Every generator returns the loader contract: ``(X, labels, meta)``.

``meta["expected"]`` records what a correct agent ought to conclude. It is never
read by the pipeline itself — only by the tests that grade the agent's decisions.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.datasets import make_blobs, make_s_curve, make_swiss_roll

Dataset = tuple[np.ndarray, np.ndarray | None, dict[str, Any]]


def swiss_roll(n_samples: int = 2000, noise: float = 0.05, seed: int = 0) -> Dataset:
    """A 2-D manifold embedded in 3-D. The textbook case for geodesic methods.

    Labels bin the roll's unrolled position, so a method that recovers the manifold
    produces a smooth colour gradient while PCA smears the bins together.
    """
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    labels = np.digitize(t, np.quantile(t, np.linspace(0, 1, 7)[1:-1]))
    meta = {
        "name": "swiss_roll",
        "source": "synthetic",
        "label_kind": "derived",
        "label_names": [f"band_{i}" for i in range(labels.max() + 1)],
        "expected": {
            "intrinsic_dim": 2,
            "structure": "single connected manifold",
            "knn_graph_connected": True,
            "manifold_methods_beat_linear": True,
        },
    }
    return X.astype(np.float64), labels, meta


def s_curve(n_samples: int = 2000, noise: float = 0.05, seed: int = 0) -> Dataset:
    """A second 2-D manifold in 3-D, less pathological than the roll.

    Useful as a contrast case: methods that mangle the Swiss roll often handle this.
    """
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=seed)
    labels = np.digitize(t, np.quantile(t, np.linspace(0, 1, 7)[1:-1]))
    meta = {
        "name": "s_curve",
        "source": "synthetic",
        "label_kind": "derived",
        "label_names": [f"band_{i}" for i in range(labels.max() + 1)],
        "expected": {
            "intrinsic_dim": 2,
            "structure": "single connected manifold",
            "knn_graph_connected": True,
            "manifold_methods_beat_linear": True,
        },
    }
    return X.astype(np.float64), labels, meta


def blobs(
    n_samples: int = 2000,
    n_features: int = 20,
    n_clusters: int = 5,
    cluster_std: float = 1.0,
    seed: int = 0,
) -> Dataset:
    """Isotropic Gaussian clusters in a linear subspace.

    The negative control. There is no manifold here, so an agent that reaches for
    Isomap or LLE is over-reading the data; PCA should be competitive with anything.
    The k-NN graph is deliberately liable to fragment across well-separated clusters,
    which is exactly the condition that breaks spectral methods.
    """
    X, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    meta = {
        "name": "blobs",
        "source": "synthetic",
        "label_kind": "ground_truth",
        "label_names": [f"cluster_{i}" for i in range(n_clusters)],
        "expected": {
            "intrinsic_dim": None,
            "structure": "separated clusters, no manifold",
            "knn_graph_connected": False,
            "manifold_methods_beat_linear": False,
        },
    }
    return X.astype(np.float64), labels, meta


def linear_subspace(
    n_samples: int = 2000,
    n_features: int = 50,
    rank: int = 5,
    noise: float = 0.1,
    seed: int = 0,
) -> Dataset:
    """Data lying near a low-rank linear subspace, plus isotropic noise.

    The case where the correct answer is "PCA, and stop there". A fast-decaying
    spectrum with a clean elbow at `rank` is the evidence the recon pass should find,
    and an agent that then spends its budget on neighbour embeddings has misread it.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.standard_normal((n_features, rank)))[0]
    scales = np.geomspace(10.0, 1.0, rank)
    coords = rng.standard_normal((n_samples, rank)) * scales
    X = coords @ basis.T + noise * rng.standard_normal((n_samples, n_features))
    labels = np.digitize(coords[:, 0], np.quantile(coords[:, 0], [0.25, 0.5, 0.75]))
    meta = {
        "name": "linear_subspace",
        "source": "synthetic",
        "label_kind": "derived",
        "label_names": [f"quartile_{i}" for i in range(4)],
        "expected": {
            "intrinsic_dim": rank,
            "structure": "low-rank linear",
            "knn_graph_connected": True,
            "manifold_methods_beat_linear": False,
        },
    }
    return X, labels, meta


def sparse_counts(
    n_cells: int = 800,
    n_genes: int = 1500,
    n_types: int = 4,
    n_markers: int = 60,
    dropout: float = 0.6,
    seed: int = 0,
) -> Dataset:
    """A crude scRNA-seq stand-in: non-negative integer counts, sparse, overdispersed.

    Not a serious simulation — it exists so the preprocessing path can be exercised
    without downloading anything. What it does reproduce is the property that matters
    for planning: Euclidean distances on the raw counts are dominated by library-size
    variation, so an agent that skips normalisation and log1p gets a visibly worse
    embedding. Cell types differ only in a small marker block, as real ones do.
    """
    rng = np.random.default_rng(seed)

    base = rng.gamma(shape=0.4, scale=2.0, size=n_genes)
    types = rng.integers(0, n_types, size=n_cells)
    markers = rng.choice(n_genes, size=(n_types, n_markers), replace=True)

    rate = np.tile(base, (n_cells, 1))
    for t in range(n_types):
        idx = np.flatnonzero(types == t)
        rate[np.ix_(idx, markers[t])] *= rng.uniform(6.0, 14.0)

    # Library size varies by an order of magnitude, as in real droplet data.
    lib = rng.lognormal(mean=0.0, sigma=0.5, size=(n_cells, 1))
    rate = rate * lib

    # Negative binomial via a gamma-Poisson mixture gives realistic overdispersion.
    counts = rng.poisson(rng.gamma(shape=2.0, scale=rate / 2.0))
    counts = counts * (rng.random(counts.shape) > dropout)

    meta = {
        "name": "sparse_counts",
        "source": "synthetic",
        "label_kind": "ground_truth",
        "label_names": [f"type_{i}" for i in range(n_types)],
        "expected": {
            "intrinsic_dim": None,
            "structure": "separated cell types in a sparse count matrix",
            "knn_graph_connected": False,
            "requires_normalisation": True,
            "manifold_methods_beat_linear": False,
        },
    }
    return counts.astype(np.float64), types, meta


GENERATORS: dict[str, Callable[..., Dataset]] = {
    "swiss_roll": swiss_roll,
    "s_curve": s_curve,
    "blobs": blobs,
    "linear_subspace": linear_subspace,
    "sparse_counts": sparse_counts,
}


def generate(name: str, seed: int = 0, **kwargs: Any) -> Dataset:
    """Build a named synthetic dataset. Deterministic given `seed`."""
    try:
        generator = GENERATORS[name]
    except KeyError:
        known = ", ".join(sorted(GENERATORS))
        raise KeyError(f"unknown synthetic dataset {name!r}; known: {known}") from None
    return generator(seed=seed, **kwargs)
