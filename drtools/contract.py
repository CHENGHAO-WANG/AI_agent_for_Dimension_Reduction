"""The loader contract.

Data loading is the one place the agent is allowed to write its own code, because
unfamiliar file formats are exactly where an "unseen dataset" actually bites. That
freedom is only safe if whatever comes back is checked before anything else runs, so
every loader — built in or agent written — must satisfy this contract:

    load(spec) -> (X, labels, meta)

    X       (n_samples, n_features), float32/float64, dense ndarray or CSR.
    labels  1-D integer array of length n_samples, or None when unlabelled.
    meta    dict carrying at minimum `name` and `source`.

Complete matrices only. Missing values are out of scope, and the refusal above says
so rather than asking a loader to resolve them: measured against this toolbox, all
ten reductions raise on NaN and no registry op imputes, so the only route a softer
refusal left open was an adapter filling the holes in silently. Imputation changes
results and belongs where the choice is recorded, which is nowhere in this system
today.

Sparse input is admitted deliberately. A 2700-cell scRNA-seq matrix is 707 MB dense
and 8 MB sparse, and the sparsity itself is a fact the profiler must report and the
planner must reason about — densifying at load time would destroy the evidence before
the agent ever sees it. Preprocessing stages densify later, as a recorded decision.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import scipy.sparse as sp

Matrix: TypeAlias = np.ndarray | sp.csr_array | sp.csr_matrix
Dataset: TypeAlias = tuple[Matrix, np.ndarray | None, dict[str, Any]]

ALLOWED_DTYPES = (np.float32, np.float64)
REQUIRED_META_KEYS = frozenset({"name", "source"})


class ContractError(ValueError):
    """A loader returned something the rest of the pipeline cannot trust."""


def check_dataset(X: Any, labels: Any, meta: Any, *, origin: str = "loader") -> None:
    """Raise `ContractError` unless the triple is safe to hand downstream.

    Every message names `origin` and says what was wrong, because the agent reads
    these failures and is expected to repair its own adapter from them.
    """
    _check_matrix(X, origin)
    _check_labels(labels, X.shape[0], origin)
    _check_meta(meta, origin)


def _check_matrix(X: Any, origin: str) -> None:
    if not (isinstance(X, np.ndarray) or sp.issparse(X)):
        raise ContractError(
            f"{origin}: X must be a numpy array or a scipy sparse matrix, "
            f"got {type(X).__name__}"
        )
    if X.ndim != 2:
        raise ContractError(f"{origin}: X must be 2-D, got {X.ndim} dimensions")
    if X.dtype not in ALLOWED_DTYPES:
        raise ContractError(
            f"{origin}: X must be float32 or float64, got {X.dtype}. Cast at load "
            "time rather than letting downstream code guess."
        )
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 1:
        raise ContractError(
            f"{origin}: X has shape {X.shape}; need at least 2 samples and 1 feature"
        )

    values = X.data if sp.issparse(X) else X
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ContractError(
            f"{origin}: X contains {n_bad} non-finite values (NaN or inf). Datasets "
            "with missing values are out of scope for this toolbox: none of the ten "
            "reductions accepts NaN, and no op imputes, so filling them in here would "
            "put fabricated numbers into every metric and figure as though they had "
            "been measured. Supply a complete matrix."
        )


def _check_labels(labels: Any, n_samples: int, origin: str) -> None:
    if labels is None:
        return
    if not isinstance(labels, np.ndarray):
        raise ContractError(
            f"{origin}: labels must be a numpy array or None, "
            f"got {type(labels).__name__}"
        )
    if labels.ndim != 1:
        raise ContractError(f"{origin}: labels must be 1-D, got {labels.ndim} dimensions")
    if labels.shape[0] != n_samples:
        raise ContractError(
            f"{origin}: labels length {labels.shape[0]} does not match X's "
            f"{n_samples} samples — the two are misaligned"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ContractError(
            f"{origin}: labels must be integer-coded, got {labels.dtype}. Map strings "
            "to codes in the loader and put the names in meta['label_names']."
        )


def _check_meta(meta: Any, origin: str) -> None:
    if not isinstance(meta, dict):
        raise ContractError(f"{origin}: meta must be a dict, got {type(meta).__name__}")
    missing = REQUIRED_META_KEYS - set(meta)
    if missing:
        raise ContractError(
            f"{origin}: meta is missing required key(s) {sorted(missing)}"
        )
