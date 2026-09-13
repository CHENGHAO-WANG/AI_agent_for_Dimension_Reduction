"""Reconnaissance: cheap probes that turn a profile into evidence about structure.

The point of this stage is to stop the planner being a lookup table. "n=2700, sparse,
therefore t-SNE" is a decision made from metadata; "the spectrum has a clean elbow at 9
components, the two-NN estimator puts intrinsic dimension near 2.1, and the k-NN graph
is a single connected component" is a decision made from the data. Everything here runs
in seconds on a subsample, which is what makes it affordable to do before committing.

Two honesty constraints shape the implementation.

First, the probes are computed on a *probe representation*, not on the raw matrix —
a k-NN graph over raw UMI counts measures sequencing depth, not biology. The
representation is chosen by a fixed documented rule and reported alongside the results,
so the agent knows what its evidence is conditional on.

Second, when that representation differs from the raw data, the spectrum is computed on
both. How much the structure changes under normalisation is itself evidence, and it is
the cheapest possible check on whether preprocessing matters for this dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from drtools.contract import Matrix

RECON_SAMPLE = 5_000
SPECTRUM_SAMPLE = 20_000
PROBE_COMPONENTS = 50
DEFAULT_K = 15


def reconnaissance(
    X: Matrix,
    labels: np.ndarray | None,
    profile: dict[str, Any],
    *,
    seed: int = 0,
    max_samples: int = RECON_SAMPLE,
    k: int | None = None,
    thumbnail_path: Any = None,
) -> dict[str, Any]:
    """Probe `X` for structural evidence. Never mutates its inputs."""
    n_samples = X.shape[0]
    index = _subsample_index(n_samples, max_samples, labels, seed)

    transform, reason = _choose_probe_representation(profile)
    probe_full = _apply_transform(X, transform)

    spectrum = {"probe": _spectrum(probe_full, seed=seed)}
    if transform:
        spectrum["raw"] = _spectrum(X, seed=seed)

    probe = probe_full[index]
    n_components = min(
        PROBE_COMPONENTS,
        min(probe.shape) - 1 if sp.issparse(probe) else min(probe.shape),
    )
    coords = _reduce(probe, n_components, seed=seed)

    k = k or min(DEFAULT_K, max(2, coords.shape[0] - 1))
    result = {
        "probe_representation": {
            "transform": transform,
            "reason": reason,
            "reduced_to_components": int(n_components),
            "note": "intrinsic dimension and the neighbourhood graph are measured on "
            "this representation, not on the raw matrix",
        },
        "subsample": {
            "n_used": int(len(index)),
            "n_total": int(n_samples),
            "stratified": labels is not None and len(index) < n_samples,
            "seed": seed,
        },
        "spectrum": spectrum,
        "intrinsic_dimension": _intrinsic_dimension(coords),
        "neighbourhood": _neighbourhood(coords, k=k),
        "thumbnail": _thumbnail(coords, labels, index, seed, thumbnail_path),
    }
    result["observations"] = _observations(result, profile)
    return result


# ------------------------------------------------------------------ probe choice


def _choose_probe_representation(profile: dict[str, Any]) -> tuple[list[str], str]:
    """Pick the representation the probes run on, by a fixed, stated rule."""
    values, features = profile["values"], profile["features"]

    if values["suspected_kind"] == "counts":
        return (
            ["normalise_total", "log1p"],
            "values are non-negative integers and sample totals vary, so raw distances "
            "would be dominated by total magnitude rather than profile shape",
        )
    scale_ratio = features["std_ratio_p95_p05"]
    if scale_ratio is not None and scale_ratio > 100:
        return (
            ["standardise"],
            f"feature scales span a factor of {scale_ratio:,.0f}, so a few features "
            "would otherwise dominate every distance",
        )
    return [], "values are already on a comparable scale; probed as given"


def _apply_transform(X: Matrix, transform: list[str]) -> Matrix:
    """Apply the probe transform. Kept separate from the real preprocessing stages."""
    out = X
    for step in transform:
        if step == "normalise_total":
            totals = np.asarray(out.sum(axis=1)).ravel()
            target = float(np.median(totals[totals > 0])) if (totals > 0).any() else 1.0
            scale = np.divide(
                target, totals, out=np.ones_like(totals, dtype=np.float64),
                where=totals > 0,
            )
            out = (
                sp.diags(scale) @ out
                if sp.issparse(out)
                else np.asarray(out) * scale[:, None]
            )
        elif step == "log1p":
            out = out.log1p() if sp.issparse(out) else np.log1p(out)
        elif step == "standardise":
            dense = np.asarray(out.todense() if sp.issparse(out) else out, dtype=float)
            std = dense.std(axis=0)
            std[std == 0] = 1.0
            out = (dense - dense.mean(axis=0)) / std
        else:  # pragma: no cover - guarded by _choose_probe_representation
            raise ValueError(f"unknown probe transform {step!r}")
    return out


def _subsample_index(
    n_samples: int, cap: int, labels: np.ndarray | None, seed: int
) -> np.ndarray:
    """Stratify when labels exist, so small classes survive the subsample."""
    if n_samples <= cap:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    if labels is None:
        return np.sort(rng.choice(n_samples, size=cap, replace=False))

    chosen: list[np.ndarray] = []
    classes, counts = np.unique(labels, return_counts=True)
    quota = np.maximum(1, np.floor(cap * counts / counts.sum()).astype(int))
    for cls, take in zip(classes, quota):
        members = np.flatnonzero(labels == cls)
        take = min(take, members.size)
        chosen.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def _reduce(X: Matrix, n_components: int, *, seed: int) -> np.ndarray:
    """Project onto leading components so neighbour searches are affordable."""
    if sp.issparse(X):
        return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(X)
    dense = np.asarray(X, dtype=np.float64)
    if n_components >= min(dense.shape):
        return dense
    return PCA(n_components=n_components, random_state=seed).fit_transform(dense)


# ---------------------------------------------------------------------- spectrum


def _spectrum(X: Matrix, *, seed: int, cap: int = SPECTRUM_SAMPLE) -> dict[str, Any]:
    """Explained variance, where it saturates, and how sharply it decays."""
    n_samples = X.shape[0]
    if n_samples > cap:
        rng = np.random.default_rng(seed)
        X = X[np.sort(rng.choice(n_samples, size=cap, replace=False))]

    # TruncatedSVD needs strictly fewer components than the smaller dimension;
    # dense PCA can take all of them, and on low-dimensional data it must, or the
    # cumulative curve never reaches its thresholds and the elbow is undefined.
    n_components = min(
        PROBE_COMPONENTS, min(X.shape) - 1 if sp.issparse(X) else min(X.shape)
    )
    if sp.issparse(X):
        model = TruncatedSVD(n_components=n_components, random_state=seed).fit(X)
        ratios = model.explained_variance_ratio_
        note = "TruncatedSVD without centering; centering a sparse matrix would make it dense"
    else:
        model = PCA(n_components=n_components, random_state=seed).fit(
            np.asarray(X, dtype=np.float64)
        )
        ratios = model.explained_variance_ratio_
        note = "PCA on centered data"

    cumulative = np.cumsum(ratios)
    eigenvalues = np.asarray(model.explained_variance_)
    participation = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())

    return {
        "n_components_computed": int(n_components),
        "explained_variance_ratio": [float(r) for r in ratios],
        "cumulative": [float(c) for c in cumulative],
        "n_components_for_80pct": _components_for(cumulative, 0.80),
        "n_components_for_90pct": _components_for(cumulative, 0.90),
        "n_components_for_95pct": _components_for(cumulative, 0.95),
        "elbow": _elbow(cumulative),
        "participation_ratio": participation,
        "decay_pc1_pc2": float(ratios[0] / ratios[1]) if ratios.size > 1 else None,
        "note": note,
    }


def _components_for(cumulative: np.ndarray, threshold: float) -> int | None:
    reached = np.flatnonzero(cumulative >= threshold)
    return int(reached[0] + 1) if reached.size else None


def _elbow(cumulative: np.ndarray) -> int | None:
    """Knee of the cumulative curve: the point furthest from the chord across it."""
    if cumulative.size < 3:
        return None
    x = np.linspace(0.0, 1.0, cumulative.size)
    y = (cumulative - cumulative[0]) / max(cumulative[-1] - cumulative[0], 1e-12)
    return int(np.argmax(y - x) + 1)


# --------------------------------------------------------- dimension and geometry


def _intrinsic_dimension(coords: np.ndarray) -> dict[str, Any]:
    """Two-NN estimator (Facco et al., 2017).

    Uses only the ratio of the two nearest-neighbour distances, so it is insensitive to
    density variation in a way that correlation-dimension estimators are not. The top
    decile of ratios is discarded because the tail is dominated by boundary points.
    """
    n_samples = coords.shape[0]
    if n_samples < 10:
        return {"twonn": None, "note": "too few samples to estimate"}

    distances, _ = NearestNeighbors(n_neighbors=3).fit(coords).kneighbors(coords)
    r1, r2 = distances[:, 1], distances[:, 2]
    usable = r1 > 0
    n_degenerate = int((~usable).sum())
    if usable.sum() < 10:
        return {
            "twonn": None,
            "note": f"{n_degenerate} points have a zero nearest-neighbour distance "
            "(duplicates); the estimator is undefined",
        }

    mu = np.sort(r2[usable] / r1[usable])
    n_usable = mu.size
    empirical_cdf = np.arange(1, n_usable + 1) / n_usable
    keep = int(0.9 * n_usable)
    x = np.log(mu[:keep])
    y = -np.log(np.maximum(1.0 - empirical_cdf[:keep], 1e-12))
    estimate = float(x @ y / (x @ x)) if (x @ x) > 0 else None

    return {
        "twonn": estimate,
        "n_used": int(n_usable),
        "n_zero_distance_points": n_degenerate,
        "ambient_probe_dim": int(coords.shape[1]),
        "note": "Facco et al. two-NN estimator on the probe representation",
    }


def _neighbourhood(coords: np.ndarray, *, k: int) -> dict[str, Any]:
    """Connectivity and density of the k-NN graph.

    Connectivity is the decisive fact for spectral methods: Laplacian Eigenmaps,
    Diffusion Maps and Isomap are all defined on a connected graph and either fail or
    silently return nonsense on a fragmented one.
    """
    graph = kneighbors_graph(coords, n_neighbors=k, mode="distance")
    symmetric = graph.maximum(graph.T)
    n_parts, membership = connected_components(symmetric, directed=False)
    _, part_sizes = np.unique(membership, return_counts=True)

    kth_distance = np.asarray(graph.max(axis=1).todense()).ravel()
    positive = kth_distance[kth_distance > 0]
    low, high = np.percentile(positive, [5, 95]) if positive.size else (0.0, 0.0)

    return {
        "k": int(k),
        "n_connected_components": int(n_parts),
        "largest_component_fraction": float(part_sizes.max() / part_sizes.sum()),
        "component_sizes": sorted(part_sizes.tolist(), reverse=True)[:10],
        "knn_distance_median": float(np.median(kth_distance)),
        "density_ratio_p95_p05": float(high / low) if low > 0 else None,
    }


# ------------------------------------------------------------------ observations


def _observations(recon: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """What the probes imply for method selection, each citing its evidence."""
    notes: list[dict[str, Any]] = []

    def note(text: str, *evidence: str) -> None:
        notes.append(
            {"observation": text, "evidence": [f"recon.{key}" for key in evidence]}
        )

    spectrum = recon["spectrum"]["probe"]
    intrinsic = recon["intrinsic_dimension"]["twonn"]
    graph = recon["neighbourhood"]

    elbow, n90 = spectrum["elbow"], spectrum["n_components_for_90pct"]
    if n90 is not None and n90 <= 10:
        note(
            f"{n90} components already explain 90% of the variance. The dominant "
            "structure is close to linear, so PCA is a strong baseline and nonlinear "
            "methods must earn their place against it rather than being assumed better.",
            "spectrum.probe.n_components_for_90pct",
        )
    elif n90 is None:
        note(
            "Variance is spread across more than the components computed here: no small "
            "linear subspace captures the data, which is the regime where nonlinear "
            "methods tend to pay off.",
            "spectrum.probe.cumulative",
        )
    if elbow is not None:
        note(
            f"The cumulative variance curve bends at roughly {elbow} components, which "
            "is the natural default for an intermediate PCA stage feeding a neighbour "
            "embedding.",
            "spectrum.probe.elbow",
            "spectrum.probe.participation_ratio",
        )

    if "raw" in recon["spectrum"]:
        raw_n90 = recon["spectrum"]["raw"]["n_components_for_90pct"]
        if raw_n90 is not None and n90 is not None and abs(raw_n90 - n90) >= 3:
            note(
                f"The spectrum shifts substantially under the probe transform "
                f"({raw_n90} components for 90% of variance raw, {n90} after). "
                "Preprocessing is not cosmetic for this dataset, and candidates with "
                "and without it are worth comparing directly.",
                "spectrum.raw.n_components_for_90pct",
                "spectrum.probe.n_components_for_90pct",
            )

    if intrinsic is not None:
        ambient = recon["intrinsic_dimension"]["ambient_probe_dim"]
        if intrinsic < 5:
            note(
                f"Two-NN puts the intrinsic dimension near {intrinsic:.1f} within a "
                f"{ambient}-dimensional probe space. A low-dimensional manifold is "
                "plausible, which is the assumption Isomap, LLE and Diffusion Maps "
                "depend on.",
                "intrinsic_dimension.twonn",
            )
        elif intrinsic > 15:
            note(
                f"Two-NN puts the intrinsic dimension near {intrinsic:.1f}. A 2-D "
                "embedding must discard a great deal, so expect all methods to score "
                "modestly and interpret apparent structure cautiously.",
                "intrinsic_dimension.twonn",
            )

    if graph["n_connected_components"] > 1:
        note(
            f"The k={graph['k']} neighbourhood graph splits into "
            f"{graph['n_connected_components']} components, the largest holding "
            f"{graph['largest_component_fraction']:.0%} of points. Isomap, Laplacian "
            "Eigenmaps and Diffusion Maps are defined on a connected graph: either "
            "raise k until it connects, or do not select them.",
            "neighbourhood.n_connected_components",
            "neighbourhood.largest_component_fraction",
        )
    else:
        note(
            f"The k={graph['k']} neighbourhood graph is connected, so graph-based "
            "spectral methods are applicable without raising k.",
            "neighbourhood.n_connected_components",
        )

    density = graph["density_ratio_p95_p05"]
    if density is not None and density > 10:
        note(
            f"Neighbourhood radii vary {density:,.0f}-fold across the data, so density "
            "is strongly non-uniform. Fixed-k methods will over-smooth dense regions "
            "and under-connect sparse ones; a larger k stabilises the graph at the cost "
            "of local detail.",
            "neighbourhood.density_ratio_p95_p05",
        )

    if recon["subsample"]["n_used"] < recon["subsample"]["n_total"]:
        note(
            f"These probes used {recon['subsample']['n_used']:,} of "
            f"{recon['subsample']['n_total']:,} samples"
            + (", stratified by label" if recon["subsample"]["stratified"] else "")
            + ". Conclusions about global structure carry over; conclusions about fine "
            "local density do not, since subsampling thins neighbourhoods.",
            "subsample.n_used",
            "subsample.n_total",
        )

    return notes


def _thumbnail(
    coords: np.ndarray,
    labels: np.ndarray | None,
    index: np.ndarray,
    seed: int,
    path: Any,
) -> dict[str, Any]:
    """A small picture of the data, for the agent to look at before it plans.

    Every other probe here returns a number. This one returns something to see, which
    matters because the agent can read an image: a spectrum and an intrinsic dimension
    describe structure, while a glance says whether there are two clumps or twenty,
    whether they are strung out or blobby, whether one of them is a thin filament that
    no summary statistic mentions.

    Deliberately cheap and deliberately plain. It is a 2-D projection of the probe
    coordinates already computed, not a fresh embedding, so it costs nothing beyond the
    drawing; and it carries no title or legend, so it tells the agent nothing except
    what the data looks like.
    """
    if path is None:
        return {"drawn": False, "reason": "no output path was given"}

    from drtools.viz import figure_thumbnail

    thumbnail_labels = None
    if labels is not None:
        thumbnail_labels = np.asarray(labels)[index]

    info = figure_thumbnail(coords[:, :2], thumbnail_labels, path)
    return {
        "drawn": True,
        "source": "first two components of the probe representation",
        "caveat": "a projection of the probe coordinates, not a fitted embedding; read "
        "it for the shape of the data, not as a result",
        **info,
    }
