"""Plots A and B: how a representation run shows a representation of d > 2 (section 3.11).

The first two columns of an embedding are not a picture of it. For a neighbour method
the axes are arbitrary and unordered, and for metric MDS they have no order at all. So
a representation run draws two plots of each candidate instead.

- **Plot A** rotates the embedding onto its own principal axes. A rotation changes no
  distance, so the plot shows the representation itself, projected onto the two
  directions that carry most of its variance, and it states the share they carry. It
  equals the first two coordinates wherever those are already ordered by importance.
- **Plot B** is UMAP fitted on the representation, with every setting fixed by rule --
  the size-aware `n_neighbors` suggestion, the registry's `min_dist`, the run's seed --
  and identical for every candidate, so that a difference between two candidates' B
  plots comes from their representations and not from settings. It is omitted at
  d = 2, where plot A is already the whole representation.

Neither plot is tuned, scored, ranked or a candidate: it is how the deliverable is
looked at, and the battery has already scored the deliverable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drtools.pipeline import run_pipeline

#: Plot B's method, fixed by rule rather than chosen (section 3.11).
PLOT_B_METHOD = "umap"


def principal_axes(embedding: np.ndarray) -> tuple[np.ndarray, float]:
    """The embedding on its first two principal axes, and the variance share they carry."""
    centred = np.asarray(embedding, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    _, singular, rows = np.linalg.svd(centred, full_matrices=False)
    total = float(np.sum(singular**2))
    share = float(np.sum(singular[:2] ** 2) / total) if total > 0 else 1.0
    return centred @ rows[:2].T, share


def umap_layout(embedding: np.ndarray, *, n_neighbors: int, seed: int) -> np.ndarray:
    """UMAP on the representation, at the fixed settings plot B is drawn with."""
    result = run_pipeline(
        np.asarray(embedding, dtype=np.float64),
        None,
        [{"op": PLOT_B_METHOD,
          "params": {"n_neighbors": int(n_neighbors), "n_components": 2}}],
        seed=seed,
    )
    return result.embedding


def draw_plots(
    embedding: np.ndarray, *, n_neighbors: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Both plots' coordinates, and the record of how each was made."""
    d = int(embedding.shape[1])
    coordinates: dict[str, np.ndarray] = {}
    record: dict[str, Any] = {"d": d}
    coordinates["A"], share = principal_axes(embedding)
    record["A"] = {"variance_share": round(share, 4)}
    if d <= 2:
        record["B"] = None
        record["B_omitted"] = "d = 2: plot A is already the whole representation"
        return coordinates, record
    coordinates["B"] = umap_layout(embedding, n_neighbors=n_neighbors, seed=seed)
    record["B"] = {
        "method": PLOT_B_METHOD, "n_neighbors": int(n_neighbors), "seed": int(seed)
    }
    return coordinates, record
