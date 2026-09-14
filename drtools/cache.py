"""Caching the loaded dataset inside the run directory.

Candidates run in separate processes, and each one would otherwise reload the dataset
from source — tolerable for a 2,700-cell matrix, wasteful for a hundred thousand images
across five candidates. Writing it once into the run directory also makes the run
self-contained: the artefacts describe an analysis of a specific matrix that is still
there to inspect, rather than of whatever the source happens to hold later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from drtools import jsonio
from drtools.contract import ContractError, Matrix
from drtools.digest import content_hash
from drtools.runs import RunDir


def cache_dir(run: RunDir):
    path = run.path / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_cached(run: RunDir) -> bool:
    directory = run.path / "data"
    return (directory / "meta.json").exists()


def write_cache(
    run: RunDir, X: Matrix, labels: np.ndarray | None, meta: dict[str, Any]
) -> dict[str, Any]:
    directory = cache_dir(run)
    sparse = sp.issparse(X)

    if sparse:
        sp.save_npz(directory / "X.npz", sp.csr_matrix(X))
    else:
        np.save(directory / "X.npy", np.ascontiguousarray(X))
    if labels is not None:
        np.save(directory / "labels.npy", labels)

    descriptor = {
        **meta,
        "cached_storage": "sparse_csr" if sparse else "dense",
        "cached_shape": list(X.shape),
        "cached_dtype": str(X.dtype),
        "has_labels": labels is not None,
        "dataset_digest": content_hash(X, labels),
    }
    jsonio.write(directory / "meta.json", descriptor)
    return descriptor


def read_cache(run: RunDir) -> tuple[Matrix, np.ndarray | None, dict[str, Any]]:
    directory = run.path / "data"
    meta = jsonio.read(directory / "meta.json")

    if meta["cached_storage"] == "sparse_csr":
        X: Matrix = sp.load_npz(directory / "X.npz").tocsr()
    else:
        # Read-only memory mapping: several candidate processes share one copy, and
        # every executor builds new arrays rather than writing through to this one.
        X = np.load(directory / "X.npy", mmap_mode="r")

    labels = np.load(directory / "labels.npy") if meta.get("has_labels") else None
    return X, labels, meta


def ensure_cache(
    run: RunDir, X: Matrix, labels: np.ndarray | None, meta: dict[str, Any]
) -> dict[str, Any]:
    """Cache on first contact; on every later contact, verify rather than trust.

    The old behaviour was to return the existing cache untouched whenever one existed,
    which let a repaired loader produce a profile of one matrix and candidates of
    another under a single run id.
    """
    if not is_cached(run):
        return write_cache(run, X, labels, meta)

    cached = jsonio.read(run.path / "data" / "meta.json")
    incoming = content_hash(X, labels)
    if cached.get("dataset_digest") != incoming:
        raise ContractError(
            f"this run was created from a different dataset. The cache holds "
            f"{cached.get('dataset_digest', 'no digest')[:12]} and the data just "
            f"loaded is {incoming[:12]}. A run describes one dataset, so analyse the "
            f"changed data in a new run rather than reusing this one."
        )
    return cached
