"""Running a candidate: an ordered list of stages applied to a dataset.

A candidate is a pipeline, not a method name. `PCA -> UMAP` is the standard scRNA-seq
route and nobody runs t-SNE on thirty thousand raw genes, so a representation that
could only express single algorithms would force the agent into choices that are simply
wrong. Making the pipeline the unit also means the agent can enter `umap(raw)` and
`pca50 -> umap` as two candidates and let the metrics decide whether the intermediate
step helped — an experiment it designs, rather than a recipe it follows.

Every stage leaves a record: the parameters it ran with, where each value came from,
what it did, and how long it took. The stage records are what let the report say what
happened instead of what was planned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix
from drtools.executors import Context, ExecutionError, get_executor
from drtools.registry import Registry, RegistryError, load_registry

Stage = dict[str, Any]


class PipelineError(ValueError):
    """A stage list is malformed or structurally impossible."""


@dataclass
class StageRecord:
    op: str
    params: dict[str, Any]
    param_provenance: dict[str, str]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    duration_s: float
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "params": self.params,
            "param_provenance": self.param_provenance,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "duration_s": round(self.duration_s, 4),
            "notes": self.notes,
        }


@dataclass
class PipelineResult:
    embedding: np.ndarray
    labels: np.ndarray | None
    stages: list[StageRecord]
    context: Context

    @property
    def total_duration_s(self) -> float:
        return sum(stage.duration_s for stage in self.stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "output_shape": list(self.embedding.shape),
            "total_duration_s": round(self.total_duration_s, 4),
            "n_samples_subsampled": (
                int(self.context.sample_index.size)
                if self.context.sample_index is not None
                else None
            ),
        }


def _tag_failure(
    error: ExecutionError,
    op: str,
    params: dict[str, Any],
    underlying: str | None = None,
) -> None:
    """Attach the failing stage to an exception, without overwriting an inner tag."""
    if getattr(error, "op", None) is None:
        error.op = op
        error.params = dict(params)
        error.underlying = underlying


def failure_record(error: ExecutionError) -> dict[str, Any]:
    """The structured form of a failure: what broke, where, and with what settings.

    A traceback tells a developer where in the library the error surfaced. This tells
    the agent which stage of its plan failed and what it was configured with, which is
    what it needs in order to revise and retry.
    """
    return {
        "op": getattr(error, "op", None),
        "error_type": getattr(error, "underlying", None) or "ExecutionError",
        "message": str(error),
        "params": getattr(error, "params", {}),
    }


def normalise_stages(stages: Any) -> list[Stage]:
    """Accept `["pca"]` or `[{"op": "pca", "params": {...}}]` and return the long form."""
    if not isinstance(stages, list):
        raise PipelineError(f"stages must be a list, got {type(stages).__name__}")

    normalised: list[Stage] = []
    for position, stage in enumerate(stages):
        if isinstance(stage, str):
            normalised.append({"op": stage, "params": {}})
            continue
        if not isinstance(stage, dict):
            raise PipelineError(
                f"stage {position}: must be a name or a mapping, "
                f"got {type(stage).__name__}"
            )
        if "op" not in stage:
            raise PipelineError(f"stage {position}: missing required key 'op'")
        params = stage.get("params") or {}
        if not isinstance(params, dict):
            raise PipelineError(f"stage {position}: 'params' must be a mapping")
        normalised.append({"op": str(stage["op"]), "params": dict(params)})
    return normalised


def validate_stages(
    stages: list[Stage],
    registry: Registry | None = None,
    *,
    require_terminal_reduction: bool = True,
) -> list[Stage]:
    """Structural checks that do not need the data: ops exist, order is possible.

    This is deliberately separate from running, so a plan can be rejected before any
    compute is spent on it. Constraints that depend on the data — whether n exceeds a
    method's limit, whether the graph connects — belong to the plan validator, which
    can see the profile.
    """
    registry = registry or load_registry()
    stages = normalise_stages(stages)
    if not stages:
        raise PipelineError("a candidate needs at least one stage")

    for position, stage in enumerate(stages):
        op = stage["op"]
        try:
            spec = registry[op]
            registry.resolve_params(op, stage["params"])
        except RegistryError as error:
            raise PipelineError(f"stage {position} ({op}): {error}") from None

        is_last = position == len(stages) - 1
        if not is_last and not spec.can_be_intermediate():
            following = stages[position + 1]["op"]
            raise PipelineError(
                f"stage {position} ({op}) cannot be followed by {following!r}: "
                f"{op} is a terminal method whose output is an embedding for viewing, "
                "not a representation to reduce further. Its coordinates have no "
                "meaningful metric for a downstream method to consume."
            )

    # Base preprocessing is the exception: it is a stage list by construction made only
    # of preprocessing, since its output is the common representation candidates are
    # measured against rather than an embedding.
    if require_terminal_reduction and not registry[stages[-1]["op"]].is_reduction:
        raise PipelineError(
            f"a candidate must end in a reduction; this one ends in "
            f"{stages[-1]['op']!r}, which is preprocessing and leaves the data in its "
            "original dimensionality"
        )
    return stages


def run_pipeline(
    X: Matrix,
    labels: np.ndarray | None,
    stages: list[Stage],
    *,
    seed: int = 0,
    registry: Registry | None = None,
    require_terminal_reduction: bool = True,
) -> PipelineResult:
    """Apply `stages` in order. Raises `ExecutionError` with an actionable message."""
    registry = registry or load_registry()
    stages = validate_stages(
        stages, registry, require_terminal_reduction=require_terminal_reduction
    )

    context = Context(
        labels=None if labels is None else np.asarray(labels).copy(),
        seed=seed,
        n_samples_original=int(X.shape[0]),
    )
    current: Matrix = X
    records: list[StageRecord] = []

    for stage in stages:
        op = stage["op"]
        params, provenance = registry.resolve_params(op, stage["params"])
        executor = get_executor(op)

        input_shape = tuple(int(v) for v in current.shape)
        started = time.perf_counter()
        try:
            current, notes = executor(current, context, **params)
        except ExecutionError as error:
            # Tag the failure with the stage and the parameters it ran with, so the
            # structured failure record can name them without re-deriving anything.
            _tag_failure(error, op, params)
            raise
        except Exception as error:
            # Library failures are re-raised as ExecutionError so that everything
            # upstream sees one failure type, carrying the original exception's name —
            # "MemoryError" and "LinAlgError" call for different repairs.
            wrapped = ExecutionError(
                f"{op} failed with {type(error).__name__}: {error}"
            )
            _tag_failure(wrapped, op, params, underlying=type(error).__name__)
            raise wrapped from error
        duration = time.perf_counter() - started

        context.history.append(op)
        records.append(
            StageRecord(
                op=op,
                params=params,
                param_provenance=provenance,
                input_shape=input_shape,
                output_shape=tuple(int(v) for v in current.shape),
                duration_s=duration,
                notes=notes,
            )
        )

    embedding = np.asarray(
        current.todense() if sp.issparse(current) else current, dtype=np.float64
    )
    return PipelineResult(
        embedding=embedding,
        labels=context.labels,
        stages=records,
        context=context,
    )
