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

A candidate covers every row (section 3.12). One that subsamples is fitted on the rows
it kept, and every other row is then passed through the stages after the subsample, in
chunks, with the parameters the fit produced. The embedding comes back in the dataset's
own row order, and the record says which rows were fitted and which were projected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix
from drtools.executors import Context, ExecutionError, Projection, get_executor
from drtools.registry import Registry, RegistryError, load_registry

Stage = dict[str, Any]

#: The working memory one chunk of projected rows may take, in bytes. A chunk's
#: heaviest object is its matrix against every fitted row -- a kernel, a distance or a
#: geodesic matrix -- so the rows per chunk are this divided by eight bytes times the
#: larger of the fitted rows and the features entering the projection.
PROJECTION_CHUNK_BYTES = 256 * 1024**2


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
    #: How rows this stage was not fitted on passed through it, and the time that took;
    #: None for a stage that ran on every row.
    projection: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        record = {
            "op": self.op,
            "params": self.params,
            "param_provenance": self.param_provenance,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "duration_s": round(self.duration_s, 4),
            "notes": self.notes,
        }
        if self.projection is not None:
            record["projection"] = {
                **self.projection,
                "duration_s": round(self.projection["duration_s"], 4),
            }
        return record


@dataclass
class Coverage:
    """Which rows a candidate was fitted on, and how the rest were placed."""

    n_rows: int
    fitted_index: np.ndarray | None = None
    chunk_rows: int | None = None
    n_chunks: int = 0

    @property
    def n_fitted(self) -> int:
        return self.n_rows if self.fitted_index is None else int(self.fitted_index.size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_fitted": self.n_fitted,
            "n_projected": self.n_rows - self.n_fitted,
            "fitted_on": "every row" if self.fitted_index is None else "subsample",
            "chunk_rows": self.chunk_rows,
            "n_chunks": self.n_chunks,
        }


@dataclass
class PipelineResult:
    embedding: np.ndarray
    labels: np.ndarray | None
    stages: list[StageRecord]
    context: Context
    coverage: Coverage

    @property
    def total_duration_s(self) -> float:
        """Fitting and projecting together: what the candidate cost to deliver."""
        return sum(
            stage.duration_s
            + (stage.projection["duration_s"] if stage.projection else 0.0)
            for stage in self.stages
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "output_shape": list(self.embedding.shape),
            "total_duration_s": round(self.total_duration_s, 4),
            "rows": self.coverage.as_dict(),
        }


def save_embedding(directory: Path, candidate_id: str, result: PipelineResult) -> None:
    """Write a candidate's arrays: the embedding, and the rows it was fitted on.

    The fitted rows are written only when the candidate subsampled. Evaluation never
    reads them, since every embedding covers every row; they are the record of which
    rows the fit saw.
    """
    np.save(directory / f"{candidate_id}.npy", result.embedding)
    if result.labels is not None:
        np.save(directory / f"{candidate_id}.labels.npy", result.labels)
    if result.coverage.fitted_index is not None:
        np.save(directory / f"{candidate_id}.fitted.npy", result.coverage.fitted_index)


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

    methods: list[str] = []
    for position, stage in enumerate(stages):
        op = stage["op"]
        try:
            spec = registry[op]
            resolved, _ = registry.resolve_params(op, stage["params"])
        except RegistryError as error:
            raise PipelineError(f"stage {position} ({op}): {error}") from None

        # Section 3.9: one pre-step, then one method. A PCA of a PCA is the smaller PCA
        # taken directly, and a third method has no useful occupant.
        if spec.is_reduction or spec.is_visualization:
            if op in methods:
                raise PipelineError(
                    f"stage {position} ({op}) repeats a method this candidate already "
                    "runs, and a candidate's second method must differ from its "
                    "first: a PCA of a PCA is the smaller PCA taken directly, so the "
                    "extra stage spends a place in the chain and changes nothing. "
                    "Drop the repeated stage."
                )
            if len(methods) == 2:
                raise PipelineError(
                    f"stage {position} ({op}) is this candidate's third reduction or "
                    "visualization method, and a candidate holds at most two: one "
                    f"pre-step, then one method ({' -> '.join(methods)} already). "
                    "Drop the surplus stage."
                )
            methods.append(op)

        is_last = position == len(stages) - 1
        if not is_last and not spec.can_be_intermediate():
            following = stages[position + 1]["op"]
            allowed = ", ".join(
                sorted(n for n, s in registry.ops.items() if "intermediate" in s.roles)
            )
            raise PipelineError(
                f"stage {position} ({op}) cannot be followed by {following!r}: {op} is "
                "a terminal method, which may stand only in a candidate's last stage. "
                f"Only {allowed} may come before a reduction or a visualization method."
            )

        # Section 3.11: a visualization method's output is a picture, drawn at two
        # dimensions. Keyed on the class, so a method added to it needs no entry here.
        if spec.is_visualization and resolved.get("n_components") != 2:
            raise PipelineError(
                f"stage {position} ({op}) asks for "
                f"n_components={resolved.get('n_components')}, but {op} is a "
                "visualization method, and a visualization method runs at "
                "n_components = 2: its output is a picture, never a representation of "
                "more dimensions. Set n_components to 2, or choose a reduction if the "
                "analysis needs more dimensions."
            )

    # Base preprocessing is the exception: it is a stage list by construction made only
    # of preprocessing, since its output is the common representation candidates are
    # measured against rather than an embedding.
    final = registry[stages[-1]["op"]]
    if require_terminal_reduction and not (final.is_reduction or final.is_visualization):
        raise PipelineError(
            f"a candidate must end in a reduction or a visualization method; this one "
            f"ends in {stages[-1]['op']!r}, which is preprocessing and leaves the data "
            "in its original dimensionality"
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

    all_labels = None if labels is None else np.asarray(labels).copy()
    context = Context(
        labels=None if all_labels is None else all_labels.copy(),
        seed=seed,
        n_samples_original=int(X.shape[0]),
    )
    current: Matrix = X
    records: list[StageRecord] = []
    # Set at the subsample: every row as it entered it, and which of them were kept.
    # From there on each stage is fitted on the kept rows and records its projection.
    split: tuple[Matrix, np.ndarray] | None = None
    projections: list[Projection] = []

    for stage in stages:
        op = stage["op"]
        params, provenance = registry.resolve_params(op, stage["params"])
        executor = get_executor(op)

        if op == "subsample" and split is not None:
            raise PipelineError(
                "this candidate subsamples twice. It is fitted on the rows its first "
                "subsample kept and every other row is projected, so a second one "
                "would leave rows that are neither fitted nor projected. Drop the "
                "second subsample stage."
            )

        input_shape = tuple(int(v) for v in current.shape)
        entering = current
        context.projection = None
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

        if split is not None:
            if context.projection is None:
                error = ExecutionError(
                    f"{op} follows a subsample but has no way to place rows it was not "
                    "fitted on: it has no transform and no standard extension. A "
                    "candidate that subsamples must cover every row, so this one "
                    f"cannot run. Drop the candidate and reject {op}, citing "
                    "profile.shape.n_samples."
                )
                _tag_failure(error, op, params)
                raise error
            projections.append(context.projection)
        elif op == "subsample" and context.sample_index is not None:
            split = (entering, context.sample_index)

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

    fitted = np.asarray(
        current.todense() if sp.issparse(current) else current, dtype=np.float64
    )
    coverage = Coverage(n_rows=int(X.shape[0]))
    embedding = (
        fitted
        if split is None
        else _project_rest(
            fitted, split, projections, records[len(records) - len(projections):],
            coverage,
        )
    )
    return PipelineResult(
        embedding=embedding,
        labels=all_labels,
        stages=records,
        context=context,
        coverage=coverage,
    )


def _project_rest(
    fitted: np.ndarray,
    split: tuple[Matrix, np.ndarray],
    projections: list[Projection],
    records: list[StageRecord],
    coverage: Coverage,
) -> np.ndarray:
    """Place every row the fit did not see, and return the embedding in row order.

    Each chunk passes through every fitted stage before the next chunk starts, so no
    stage's output is ever held for all the projected rows at once: after `standardise`
    a chunk is dense, and all of them together would be the dense matrix the subsample
    existed to avoid.
    """
    entering, kept = split
    n_rows = int(entering.shape[0])
    rest = np.setdiff1d(np.arange(n_rows), kept, assume_unique=True)
    coverage.fitted_index = np.asarray(kept)

    width = max(int(kept.size), int(entering.shape[1]))
    chunk_rows = max(1, min(int(rest.size), PROJECTION_CHUNK_BYTES // (8 * width)))
    coverage.chunk_rows = int(chunk_rows)

    embedding = np.empty((n_rows, fitted.shape[1]), dtype=np.float64)
    embedding[kept] = fitted
    spent = [0.0] * len(projections)
    for start in range(0, rest.size, chunk_rows):
        rows = rest[start:start + chunk_rows]
        chunk: Matrix = entering[rows]
        for position, (projection, record) in enumerate(zip(projections, records)):
            started = time.perf_counter()
            try:
                chunk = projection.function(chunk)
            except ExecutionError as error:
                _tag_failure(error, record.op, record.params)
                raise
            except Exception as error:
                wrapped = ExecutionError(
                    f"{record.op} failed placing rows it was not fitted on, with "
                    f"{type(error).__name__}: {error}"
                )
                _tag_failure(
                    wrapped, record.op, record.params, underlying=type(error).__name__
                )
                raise wrapped from error
            spent[position] += time.perf_counter() - started
        embedding[rows] = np.asarray(
            chunk.todense() if sp.issparse(chunk) else chunk, dtype=np.float64
        )
        coverage.n_chunks += 1

    for projection, record, seconds in zip(projections, records, spent):
        record.projection = {
            "kind": projection.kind,
            "n_rows": int(rest.size),
            "duration_s": seconds,
        }
    return embedding
