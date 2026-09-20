"""One content digest for the matrix a run is analysing.

Identity is what the data *is*, not how it happens to be stored. Two sparse matrices
holding the same values must agree even when their indices arrived unsorted, carry an
explicitly stored zero, or are int64 rather than int32 — index width is a storage
detail, and a loader that switches to it has not produced a different dataset. So
everything is canonicalised before it is hashed, on both the write path and the check
path, which is why this lives in one function rather than at each call site.
"""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix


def content_hash(X: Matrix, labels: np.ndarray | None) -> str:
    """A stable digest of the matrix and its label codes."""
    digest = hashlib.sha256()

    if sp.issparse(X):
        matrix = X.tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix.eliminate_zeros()
        digest.update(b"sparse_csr")
        digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
        digest.update(str(matrix.data.dtype).encode())
        digest.update(np.ascontiguousarray(matrix.data).tobytes())
        # Fixed width and endianness: int32 and int64 indices describe one matrix.
        digest.update(matrix.indices.astype("<i8").tobytes())
        digest.update(matrix.indptr.astype("<i8").tobytes())
    else:
        dense = np.ascontiguousarray(X)
        digest.update(b"dense")
        digest.update(np.asarray(dense.shape, dtype="<i8").tobytes())
        digest.update(str(dense.dtype).encode())
        # Row blocks rather than one .tobytes(): X is often a memmap, and a whole-array
        # copy of a gigabyte to hash it defeats the point of memory mapping it.
        for start in range(0, dense.shape[0], 4096):
            digest.update(np.ascontiguousarray(dense[start : start + 4096]).tobytes())

    if labels is None:
        digest.update(b"no_labels")
    else:
        codes = np.ascontiguousarray(np.asarray(labels))
        digest.update(b"labels")
        digest.update(np.asarray(codes.shape, dtype="<i8").tobytes())
        digest.update(codes.astype("<i8").tobytes())

    return digest.hexdigest()
