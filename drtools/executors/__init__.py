"""Stage implementations, and the registry that dispatches to them.

Every executor has the same shape:

    fn(X, ctx, **params) -> (X_out, notes)

`notes` is where an executor says what it actually did — how many features survived a
filter, that a sparse input meant no centring, which bandwidth was chosen when the plan
left it null. Those notes end up in the run's stage records and from there in the
report, which is the only reason a reader can tell "PCA to 50 components" from "PCA to
50 components, uncentred, because the input was sparse".

Nothing here decides *whether* a stage should run. That is the planner's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from drtools.contract import Matrix

MEMORY_BUDGET_GB = 2.0


class ExecutionError(RuntimeError):
    """A stage cannot run on this input, with a message the agent can act on."""


@dataclass
class Context:
    """State that flows alongside the matrix through a pipeline."""

    labels: np.ndarray | None = None
    seed: int = 0
    n_samples_original: int = 0
    sample_index: np.ndarray | None = None
    feature_index: np.ndarray | None = None
    history: list[str] = field(default_factory=list)

    def select_samples(self, index: np.ndarray) -> None:
        """Record that a stage kept only some samples, keeping labels aligned."""
        index = np.asarray(index)
        if self.labels is not None:
            self.labels = self.labels[index]
        self.sample_index = (
            index if self.sample_index is None else self.sample_index[index]
        )

    def select_features(self, index: np.ndarray) -> None:
        index = np.asarray(index)
        self.feature_index = (
            index if self.feature_index is None else self.feature_index[index]
        )


class Executor(Protocol):
    def __call__(
        self, X: Matrix, ctx: Context, **params: Any
    ) -> tuple[Matrix, dict[str, Any]]: ...


EXECUTORS: dict[str, Executor] = {}


def executor(name: str) -> Callable[[Executor], Executor]:
    """Register a stage implementation under the name used in the registry."""

    def register(function: Executor) -> Executor:
        if name in EXECUTORS:
            raise RuntimeError(f"executor {name!r} is already registered")
        EXECUTORS[name] = function
        return function

    return register


def get_executor(name: str) -> Executor:
    try:
        return EXECUTORS[name]
    except KeyError:
        raise ExecutionError(
            f"no executor implements {name!r}; implemented ops are "
            f"{', '.join(sorted(EXECUTORS))}"
        ) from None


def require_dense(X: Matrix, op: str) -> np.ndarray:
    """Reject sparse input rather than densifying behind the planner's back.

    Densification can turn 8 MB into 700 MB, which is a decision with consequences and
    therefore one the plan has to make explicitly.
    """
    import scipy.sparse as sp

    if sp.issparse(X):
        n, d = X.shape
        raise ExecutionError(
            f"{op} requires dense input but received a sparse matrix. Add a densify "
            f"stage before it (cost: {n * d * 8 / 1e9:.2f} GB), or reduce the feature "
            "count first with select_variable_features or a sparse-capable PCA."
        )
    return np.asarray(X, dtype=np.float64)


def require_pairwise_affordable(n_samples: int, op: str) -> None:
    """Refuse to allocate an n-by-n matrix that will not fit."""
    required_gb = n_samples**2 * 8 / 1e9
    if required_gb > MEMORY_BUDGET_GB:
        raise ExecutionError(
            f"{op} needs a dense {n_samples}x{n_samples} matrix "
            f"({required_gb:.1f} GB, over the {MEMORY_BUDGET_GB:.0f} GB budget). "
            "Subsample first, and record that the metrics then describe the subset."
        )


# Importing the modules is what populates EXECUTORS.
from drtools.executors import (  # noqa: E402,F401
    linear,
    manifold,
    neighbour,
    preprocessing,
    spectral,
)

__all__ = [
    "Context",
    "EXECUTORS",
    "ExecutionError",
    "Executor",
    "executor",
    "get_executor",
    "require_dense",
    "require_pairwise_affordable",
]
