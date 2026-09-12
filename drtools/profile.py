"""Describing a dataset before anything is done to it.

The profile is the agent's first contact with the data, and it is deliberately more
than a table of numbers. Alongside the measurements it carries `observations`: short
statements, each tied to the measurement that produced it, saying what a given
property *implies for the analysis*. "sparsity: 0.87" is a number the planner has to
interpret; "87% of entries are zero and library sizes vary 12-fold, so Euclidean
distances on the raw matrix mostly measure sequencing depth" is evidence it can act on.

Nothing here modifies the data or decides anything. Deciding is the planner's job; this
module's job is to make the decision well-posed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix

# Sampling caps. Profiling must stay cheap enough to run on anything.
DUPLICATE_SAMPLE = 5_000
QUANTILE_SAMPLE = 20_000


def profile_dataset(
    X: Matrix, labels: np.ndarray | None, meta: dict[str, Any]
) -> dict[str, Any]:
    """Measure `X` and state what the measurements imply. Never mutates its inputs."""
    n_samples, n_features = X.shape
    is_sparse = sp.issparse(X)
    values = X.data if is_sparse else np.asarray(X).ravel()

    shape_facts = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "aspect_ratio": float(n_features / n_samples),
        "dtype": str(X.dtype),
        "storage": "sparse_csr" if is_sparse else "dense",
        "memory_mb": round(_memory_bytes(X) / 1e6, 2),
    }

    n_zero = int(X.size - X.nnz) if is_sparse else int((values == 0).sum())
    value_facts = {
        "sparsity": float(n_zero / (n_samples * n_features)),
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "is_nonnegative": bool(values.min() >= 0) if values.size else True,
        "is_integer_valued": bool(np.all(values == np.round(values))),
        "n_distinct_sampled": int(
            np.unique(_subsample_values(values, QUANTILE_SAMPLE)).size
        ),
    }
    value_facts["suspected_kind"] = _suspect_kind(value_facts)

    feature_facts = _feature_statistics(X)
    sample_facts = _sample_statistics(X)
    label_facts = _label_statistics(labels, meta)

    profile = {
        "name": meta.get("name", "unnamed"),
        "spec": meta.get("spec"),
        "source": meta.get("source"),
        "modality": meta.get("modality"),
        "value_kind_declared": meta.get("value_kind"),
        "shape": shape_facts,
        "values": value_facts,
        "features": feature_facts,
        "samples": sample_facts,
        "labels": label_facts,
    }
    profile["observations"] = _observations(profile)
    return profile


# ------------------------------------------------------------------- measurements


def _memory_bytes(X: Matrix) -> int:
    if sp.issparse(X):
        return int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    return int(X.nbytes)


def _subsample_values(values: np.ndarray, cap: int) -> np.ndarray:
    if values.size <= cap:
        return values
    rng = np.random.default_rng(0)
    return values[rng.choice(values.size, size=cap, replace=False)]


def _suspect_kind(value_facts: dict[str, Any]) -> str:
    """What sort of measurement this looks like, on the evidence of the values alone."""
    if value_facts["n_distinct_sampled"] <= 2 and value_facts["max"] <= 1:
        return "binary"
    if value_facts["is_integer_valued"] and value_facts["is_nonnegative"]:
        return "counts"
    if value_facts["is_nonnegative"] and value_facts["max"] <= 1.0:
        return "bounded_unit_interval"
    return "continuous"


def _feature_statistics(X: Matrix) -> dict[str, Any]:
    """Per-feature spread, which decides whether standardisation is needed."""
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_of_squares = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        variance = np.maximum(mean_of_squares - mean**2, 0.0)
    else:
        variance = np.asarray(X).var(axis=0)
    std = np.sqrt(variance)

    nonzero_std = std[std > 0]
    low, high = (
        np.percentile(nonzero_std, [5, 95]) if nonzero_std.size else (0.0, 0.0)
    )
    return {
        "n_constant": int((std == 0).sum()),
        "n_near_constant": int((std < 1e-8).sum()),
        "std_min": float(nonzero_std.min()) if nonzero_std.size else 0.0,
        "std_median": float(np.median(nonzero_std)) if nonzero_std.size else 0.0,
        "std_max": float(nonzero_std.max()) if nonzero_std.size else 0.0,
        "std_ratio_p95_p05": float(high / low) if low > 0 else None,
    }


def _sample_statistics(X: Matrix) -> dict[str, Any]:
    """Per-sample totals. For count data this is library size, which drives distances."""
    totals = np.asarray(X.sum(axis=1)).ravel()
    positive = totals[totals > 0]
    ratio = float(positive.max() / positive.min()) if positive.size else None

    nonzero_per_sample = (
        np.asarray((X != 0).sum(axis=1)).ravel()
        if sp.issparse(X)
        else (np.asarray(X) != 0).sum(axis=1)
    )
    return {
        "total_min": float(totals.min()),
        "total_median": float(np.median(totals)),
        "total_max": float(totals.max()),
        "total_ratio_max_min": ratio,
        "nonzero_features_median": float(np.median(nonzero_per_sample)),
        "n_all_zero_samples": int((totals == 0).sum()),
        "n_duplicate_rows": _count_duplicate_rows(X),
    }


def _count_duplicate_rows(X: Matrix) -> int:
    """Exact duplicates in a capped sample. Duplicates break neighbour graphs."""
    n_samples = X.shape[0]
    rng = np.random.default_rng(0)
    index = (
        np.arange(n_samples)
        if n_samples <= DUPLICATE_SAMPLE
        else rng.choice(n_samples, size=DUPLICATE_SAMPLE, replace=False)
    )
    block = X[index]
    dense = np.asarray(block.todense() if sp.issparse(block) else block)
    return int(len(index) - np.unique(dense, axis=0).shape[0])


def _label_statistics(labels: np.ndarray | None, meta: dict[str, Any]) -> dict[str, Any]:
    if labels is None:
        return {
            "present": False,
            "kind": meta.get("label_kind"),
            "note": "no labels; supervised evaluation metrics are unavailable unless "
            "reference labels are derived",
        }
    classes, counts = np.unique(labels, return_counts=True)
    return {
        "present": True,
        "kind": meta.get("label_kind", "unknown"),
        "n_classes": int(classes.size),
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
        "names": meta.get("label_names"),
        "balance_ratio": float(counts.max() / counts.min()),
        "smallest_class_size": int(counts.min()),
    }


# ------------------------------------------------------------------- observations


def _observations(profile: dict[str, Any]) -> list[dict[str, str]]:
    """Statements about what the measurements imply, each citing its evidence.

    The `evidence` keys are dotted paths into this same profile, so a rationale that
    quotes an observation can always be traced back to the number behind it.
    """
    notes: list[dict[str, str]] = []
    shape, values = profile["shape"], profile["values"]
    features, samples, labels = (
        profile["features"],
        profile["samples"],
        profile["labels"],
    )

    def note(text: str, *evidence: str) -> None:
        # Evidence keys are rooted at the artefact so they resolve uniformly wherever
        # they are quoted — in the decision log, in the report, or in a test.
        notes.append(
            {"observation": text, "evidence": [f"profile.{key}" for key in evidence]}
        )

    n, d = shape["n_samples"], shape["n_features"]

    if d > 10 * n:
        note(
            f"{d:,} features against {n:,} samples. In this regime pairwise Euclidean "
            "distances concentrate and neighbour graphs become unreliable, so a linear "
            "reduction before any neighbour-based method is close to mandatory.",
            "shape.n_features",
            "shape.n_samples",
            "shape.aspect_ratio",
        )
    elif d > n:
        note(
            f"More features ({d:,}) than samples ({n:,}); the sample covariance is "
            "singular and any method assuming full rank needs care.",
            "shape.aspect_ratio",
        )

    if n > 50_000:
        note(
            f"{n:,} samples. Methods with quadratic memory in the sample count — MDS, "
            "Isomap, dense-kernel spectral methods — are infeasible at this size "
            "without subsampling.",
            "shape.n_samples",
        )
    elif n < 200:
        note(
            f"Only {n:,} samples. Neighbour-based methods have little to work with, "
            "and default hyperparameters tuned for thousands of points will be wrong.",
            "shape.n_samples",
        )

    if values["sparsity"] > 0.5:
        if shape["storage"] == "sparse_csr":
            dense_mb = n * d * np.dtype(shape["dtype"]).itemsize / 1e6
            cost = (
                f" Held sparsely at {shape['memory_mb']:,.0f} MB; densifying it would "
                f"take {dense_mb:,.0f} MB, so stages that require dense input are a "
                "memory decision, not just a formatting one."
            )
        else:
            cost = ""
        note(
            f"{values['sparsity']:.0%} of entries are exactly zero. Zero inflation "
            "makes raw Euclidean distance a poor similarity: two samples can agree on "
            "every measured feature and still sit far apart because they share few "
            "non-zeros." + cost,
            "values.sparsity",
            "shape.storage",
        )

    if values["suspected_kind"] == "counts":
        note(
            "Values are non-negative integers, i.e. count data. Counts are "
            "heteroscedastic — variance grows with the mean — so a variance-stabilising "
            "transform such as log1p belongs before any Euclidean-distance method.",
            "values.is_integer_valued",
            "values.is_nonnegative",
        )

    ratio = samples["total_ratio_max_min"]
    if ratio is not None and ratio > 5:
        note(
            f"Sample totals vary {ratio:,.0f}-fold across the dataset. Without "
            "per-sample normalisation, the leading variation in any distance-based "
            "embedding will be total magnitude rather than profile shape.",
            "samples.total_ratio_max_min",
        )

    scale_ratio = features["std_ratio_p95_p05"]
    if scale_ratio is not None and scale_ratio > 100:
        note(
            f"Feature standard deviations span a factor of {scale_ratio:,.0f} between "
            "the 5th and 95th percentiles. Unstandardised, a handful of high-variance "
            "features will dominate every distance and every principal component.",
            "features.std_ratio_p95_p05",
        )

    if features["n_constant"] > 0:
        note(
            f"{features['n_constant']:,} features are constant and carry no "
            "information; they should be dropped before scaling, which would otherwise "
            "divide by zero.",
            "features.n_constant",
        )

    if samples["n_all_zero_samples"] > 0:
        note(
            f"{samples['n_all_zero_samples']:,} samples are entirely zero. They have no "
            "well-defined direction and will break cosine distance and normalisation.",
            "samples.n_all_zero_samples",
        )

    if samples["n_duplicate_rows"] > 0:
        note(
            f"At least {samples['n_duplicate_rows']:,} duplicate rows (in a capped "
            "sample). Duplicates give zero-distance neighbours, which destabilises "
            "local-scaling methods and can make neighbour graphs degenerate.",
            "samples.n_duplicate_rows",
        )

    if not labels["present"]:
        note(
            "No labels. Label-based evaluation is unavailable unless reference labels "
            "are derived, which would make those metrics a measure of agreement with "
            "the clustering rather than with truth.",
            "labels.present",
        )
    elif labels.get("balance_ratio", 1.0) > 10:
        note(
            f"Class sizes are imbalanced by {labels['balance_ratio']:,.0f}-fold; the "
            f"smallest class has {labels['smallest_class_size']:,} members. Any "
            "subsampling should be stratified or small classes will vanish.",
            "labels.balance_ratio",
            "labels.smallest_class_size",
        )

    return notes
