"""Choosing d and tuning: every candidate, before ranking (sections 3.5 and 3.7).

Each reduction and visualization method declares in the registry the one fidelity
parameter tuning scales, what its multiplier scales, and what chooses its d. Tuning
runs inside the candidate's Attempt, on at most 2,000 rows of the Reference, with the
candidate's own stages unchanged, and every cell of the grid is scored by the Battery
under the pre-registered weighting. What comes out is the refit: the candidate's stages
at the chosen d and multiplier, fitted once on every row under the run's own seed and
scored alone, so the ranking never reads the best of several noisy tuning scores.

Four shapes of search, by what the method declares:

- *Nested in d, with its own criterion* -- PCA, kernel PCA, Isomap, Laplacian Eigenmaps,
  Diffusion Maps. One fit per multiplier at the largest d gives the criterion at every
  integer d. The criterion picks d for that multiplier, and the battery scores each
  (multiplier, d) pair; the best pair wins. Diffusion Maps' time t is swept on every
  fit at no cost, since its coordinates are psi * lambda**t from one decomposition.
- *Nested in d, read by the battery* -- standard LLE. One fit per multiplier, truncated
  to every grid d and scored; for each d the best multiplier is kept, and the elbow
  rule reads that profile.
- *Not nested, nothing to tune* -- MDS by its stress, sparse PCA by the battery. One fit
  per grid d.
- *Not nested, with a parameter* -- the other LLE variants. Alternate from
  reconnaissance's estimate of d: the best multiplier at d, then d at that multiplier,
  until a step changes nothing or two cycles have run.

A visualization method runs at d = 2, and only its multiplier is tuned. A PCA before a
method has its output dimension k chosen by PCA's own criterion, as a PCA's d is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from drtools.constraints import RULES
from drtools.contract import Matrix
from drtools.executors import ExecutionError
from drtools.metrics import BatteryScorer, _subsample_index, derive_seed
from drtools.pipeline import PipelineError, run_pipeline
from drtools.rank import weighted_score
from drtools.registry import Registry, load_registry

#: The most rows tuning fits on, the battery's own cap.
TUNING_ROWS = 2000
#: No reduction delivers more than this many dimensions (section 3.7).
D_CAP = 100

DEFAULT_D_GRID = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100)
DEFAULT_MULTIPLIERS = (0.5, 1.0, 2.0)
DEFAULT_T_GRID = (1, 2, 4)
#: A curve whose best value is at most this much above its value at the smallest d is
#: flat, and the smallest d is chosen: an absolute difference on the curve's 0-1 scale.
DEFAULT_FLATNESS = 0.10
#: With no elbow, the smallest d whose value reaches this share of the best: relative.
DEFAULT_FALLBACK_SHARE = 0.90
DEFAULT_MAX_CYCLES = 2

#: Criteria whose curve is read by the shared elbow procedure.
CURVE_CRITERIA = (
    "explained_variance",
    "kernel_variance",
    "residual_variance",
    "diffusion_distance",
)


class TuningError(ExecutionError):
    """No cell of a candidate's grid could run, so there is nothing to refit."""


@dataclass(frozen=True)
class TuningSettings:
    """The plan's `tuning` block: fixed at registration, the same for every candidate."""

    d_grid: tuple[int, ...] = DEFAULT_D_GRID
    multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS
    t_grid: tuple[int, ...] = DEFAULT_T_GRID
    flatness: float = DEFAULT_FLATNESS
    fallback_share: float = DEFAULT_FALLBACK_SHARE
    max_cycles: int = DEFAULT_MAX_CYCLES
    #: Set by the run's purpose, not by the plan: 2 in a visualization run, where every
    #: candidate is a picture and only its multiplier is tuned (section 3.11).
    fixed_d: int | None = None

    @classmethod
    def from_plan(cls, block: dict[str, Any] | None) -> "TuningSettings":
        block = dict(block or {})
        return cls(
            d_grid=tuple(int(d) for d in block.get("d_grid", DEFAULT_D_GRID)),
            multipliers=tuple(float(m) for m in block.get("multipliers", DEFAULT_MULTIPLIERS)),
            t_grid=tuple(int(t) for t in block.get("t_grid", DEFAULT_T_GRID)),
            flatness=float(block.get("flatness", DEFAULT_FLATNESS)),
            fallback_share=float(block.get("fallback_share", DEFAULT_FALLBACK_SHARE)),
            max_cycles=int(block.get("max_cycles", DEFAULT_MAX_CYCLES)),
            fixed_d=None if block.get("fixed_d") is None else int(block["fixed_d"]),
        )


# ------------------------------------------------------------------- the d rules


def choose_d_by_curve(
    curve: dict[int, float], *, flatness: float, fallback_share: float
) -> dict[str, Any]:
    """Section 3.7's rule for a curve Q(d) on a 0-1 scale.

    1. Replace the curve by its running maximum: a larger d never scores worse than
       keeping a smaller one.
    2. Flat -- best minus the value at the smallest d at most `flatness` -- chooses the
       smallest d.
    3. Otherwise the Kneedle elbow on log d: both axes rescaled to [0, 1], the interior
       point furthest above the chord joining the ends. An end is never an elbow.
    4. With no interior point above the chord, the smallest d reaching
       `fallback_share` of the best. Still rising at the last point, that is capped.
    """
    ds = sorted(curve)
    if not ds:
        raise TuningError("no value of d could be measured")
    running = list(np.maximum.accumulate([float(curve[d]) for d in ds]))
    best = running[-1]
    record: dict[str, Any] = {"running_max": dict(zip(ds, [round(v, 6) for v in running]))}
    still_rising = len(ds) > 1 and running[-1] > running[-2]

    if len(ds) == 1:
        return {**record, "d": ds[0], "rule": "single", "capped": False}
    if best - running[0] <= flatness:
        return {**record, "d": ds[0], "rule": "flat", "capped": False}

    x = np.log(np.asarray(ds, dtype=float))
    x = (x - x[0]) / (x[-1] - x[0])
    y = np.asarray(running)
    y = (y - y[0]) / (y[-1] - y[0]) if y[-1] > y[0] else np.zeros_like(y)
    above = y - x
    interior = above[1:-1]
    if interior.size and interior.max() > 1e-12:
        chosen = ds[1 + int(np.argmax(interior))]
        return {**record, "d": chosen, "rule": "elbow", "capped": False}

    threshold = fallback_share * best
    chosen = next(d for d, value in zip(ds, running) if value >= threshold - 1e-12)
    return {
        **record,
        "d": chosen,
        "rule": "fallback",
        "capped": bool(still_rising or chosen == ds[-1]),
    }


def choose_d_by_eigengap(eigenvalues: list[float], ds: list[int]) -> dict[str, Any]:
    """The d in `ds` with the largest gap between its eigenvalue and the next.

    `eigenvalues` are the non-trivial ones, largest first, so the gap after d is
    eigenvalues[d - 1] - eigenvalues[d]. Ties go to the smaller d.
    """
    gaps = {
        d: float(eigenvalues[d - 1] - eigenvalues[d])
        for d in ds
        if d < len(eigenvalues)
    }
    if not gaps:
        raise TuningError("too few eigenvalues to measure a gap")
    largest = max(gaps.values())
    chosen = min(d for d, gap in gaps.items() if gap >= largest - 1e-15)
    return {
        "d": chosen,
        "rule": "eigengap",
        "gaps": {d: round(g, 8) for d, g in gaps.items()},
        "capped": chosen == max(gaps),
    }


def diffusion_curve(eigenvalues: list[float], t: int, d_max: int) -> dict[int, float]:
    """Q(d): the share of squared diffusion distance the first d coordinates keep.

    The squared diffusion distance is sum_i lambda_i**(2t) (psi_i(x) - psi_i(y))**2, so
    keeping d coordinates keeps the first d terms (section 3.7).
    """
    weights = np.power(np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None), 2 * t)
    total = float(weights.sum())
    cumulative = np.cumsum(weights) / total if total > 0 else np.zeros_like(weights)
    return {d: float(cumulative[d - 1]) for d in range(1, min(d_max, len(weights)) + 1)}


# -------------------------------------------------------------------- the search


@dataclass
class Cell:
    """One fit of the grid: a multiplier's value, a d, and what it scored."""

    multiplier: float
    value: Any
    d: int
    t: int | None = None
    score: float | None = None
    values: dict[str, float | None] = field(default_factory=dict)
    feasible: bool = True
    reason: str | None = None
    runtime_s: float = 0.0
    merged: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "multiplier": self.multiplier,
            "value": self.value,
            "d": self.d,
            "score": None if self.score is None else round(self.score, 6),
            "feasible": self.feasible,
            "runtime_s": round(self.runtime_s, 4),
        }
        if self.t is not None:
            out["t"] = self.t
        if self.values:
            out["metrics"] = {k: None if v is None else round(v, 6) for k, v in self.values.items()}
        if self.reason:
            out["reason"] = self.reason
        if self.merged:
            out["merged_multipliers"] = self.merged
        return out


def _nearest_one(multiplier: float) -> float:
    return abs(np.log(multiplier))


def _better(a: Cell, b: Cell | None) -> bool:
    """Whether a beats b: higher score, then smaller d, then multiplier nearest 1, then
    smaller t -- section 3.5's tie-breaks."""
    if b is None or b.score is None:
        return a.score is not None
    if a.score is None:
        return False
    if abs(a.score - b.score) > 1e-12:
        return a.score > b.score
    key_a = (a.d, _nearest_one(a.multiplier), a.t or 0)
    key_b = (b.d, _nearest_one(b.multiplier), b.t or 0)
    return key_a < key_b


@dataclass
class _Candidate:
    """What tuning needs to know about one candidate's stages."""

    stages: list[dict[str, Any]]
    method_position: int
    pca_position: int | None
    subsample_rows: int | None


def _describe(stages: list[dict[str, Any]], registry: Registry) -> _Candidate:
    methods = [
        i for i, s in enumerate(stages)
        if registry[s["op"]].is_reduction or registry[s["op"]].is_visualization
    ]
    subsample = next((s for s in stages if s["op"] == "subsample"), None)
    rows = None
    if subsample is not None:
        rows = int(registry.resolve_params("subsample", subsample.get("params", {}))[0]["n_samples"])
    return _Candidate(
        stages=[{"op": s["op"], "params": dict(s.get("params") or {})} for s in stages],
        method_position=methods[-1],
        pca_position=methods[0] if len(methods) == 2 else None,
        subsample_rows=rows,
    )


class Tuner:
    """Tunes one candidate on the tuning rows and says what to refit."""

    def __init__(
        self,
        reference: Matrix,
        labels: np.ndarray | None,
        stages: list[dict[str, Any]],
        *,
        seed: int,
        weights: dict[str, float],
        k: int,
        settings: TuningSettings,
        suggest: Callable[[str, int, dict[str, Any]], dict[str, Any]],
        intrinsic_dimension: float | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.registry = registry or load_registry()
        self.reference = reference
        self.labels = None if labels is None else np.asarray(labels)
        self.candidate = _describe(stages, self.registry)
        self.seed = int(seed)
        self.weights = dict(weights)
        self.settings = settings
        self.suggest = suggest
        self.intrinsic_dimension = intrinsic_dimension
        self.n_rows = int(reference.shape[0])

        self.rows_seed = derive_seed(seed, "tuning-rows")
        self.fits_seed = derive_seed(seed, "tuning-fits")
        rows = _subsample_index(self.n_rows, TUNING_ROWS, self.labels, self.rows_seed)
        self.rows = rows
        self.X = reference[rows]
        self.y = None if self.labels is None else self.labels[rows]
        wanted = {name for name, weight in self.weights.items() if weight > 0}
        self.scorer = BatteryScorer(self.X, self.y, k=k, wanted=wanted)
        self.record: dict[str, Any] = {
            "rows": {
                "n_rows": int(len(rows)),
                "of": self.n_rows,
                "stratified": self.labels is not None,
            },
            "seeds": {
                "run": self.seed,
                "tuning_rows": self.rows_seed,
                "tuning_fits": self.fits_seed,
            },
            "settings": {
                "d_grid": list(settings.d_grid),
                "multipliers": list(settings.multipliers),
                "t_grid": list(settings.t_grid),
                "flatness": settings.flatness,
                "fallback_share": settings.fallback_share,
                "max_cycles": settings.max_cycles,
            },
            "k": int(k),
        }

    # ------------------------------------------------------------------ helpers

    def _fit_rows(self, n_rows: int) -> int:
        """How many rows the method is fitted on when the stages see `n_rows`."""
        limit = self.candidate.subsample_rows
        return n_rows if limit is None else min(n_rows, limit)

    def _run(self, stages: list[dict[str, Any]], *, criterion: bool = False) -> Any:
        return run_pipeline(
            self.X, self.y, stages, seed=self.fits_seed, registry=self.registry,
            measure_criterion=criterion, require_terminal_reduction=False,
        )

    def _score(self, embedding: np.ndarray) -> tuple[float | None, dict[str, Any]]:
        values = self.scorer.score(embedding)
        return weighted_score(values, self.weights), values

    # ----------------------------------------------------------------- k for PCA

    def _choose_pca_k(self) -> None:
        """PCA's k before a method, by PCA's own criterion at every integer (day 15)."""
        position = self.candidate.pca_position
        if position is None:
            return
        prefix = self.candidate.stages[:position]
        entering = self._run(prefix).embedding if prefix else self.X
        n_fit = self._fit_rows(int(entering.shape[0]))
        n_fit_refit = self._fit_rows(self.n_rows)
        k_max = min(D_CAP, int(entering.shape[1]) - 1, n_fit - 1, n_fit_refit - 1)
        if k_max < 2:
            raise TuningError(
                f"the PCA before the method can keep at most {k_max} component(s) "
                "here, too few to feed another method; drop the PCA stage"
            )
        stages = [dict(s, params=dict(s["params"])) for s in self.candidate.stages[: position + 1]]
        stages[position]["params"]["n_components"] = k_max
        result = self._run(stages, criterion=True)
        curve = result.context.criterion["curve"]
        choice = choose_d_by_curve(
            {k: curve[k - 1] for k in range(2, k_max + 1)},
            flatness=self.settings.flatness,
            fallback_share=self.settings.fallback_share,
        )
        self.candidate.stages[position]["params"]["n_components"] = int(choice["d"])
        self.record["pca_k"] = {
            "criterion": "explained_variance",
            "k": int(choice["d"]),
            "rule": choice["rule"],
            "capped": choice["capped"],
            "k_max": k_max,
            "curve": {k: round(curve[k - 1], 6) for k in range(2, k_max + 1)},
        }

    # ------------------------------------------------------------- the method

    def tune(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Tune, and return the refit's stages with the record of how."""
        started = time.perf_counter()
        self._choose_pca_k()

        position = self.candidate.method_position
        stage = self.candidate.stages[position]
        op = stage["op"]
        spec = self.registry[op]
        resolved, _ = self.registry.resolve_params(op, stage["params"])
        prefix = self.candidate.stages[:position]
        entering = self._run(prefix).embedding if prefix else self.X
        p_in = int(entering.shape[1])
        n_fit, n_fit_refit = self._fit_rows(int(self.X.shape[0])), self._fit_rows(self.n_rows)
        d_max = min(D_CAP, p_in - 1, n_fit - 1, n_fit_refit - 1)
        fixed = 2 if spec.is_visualization else self.settings.fixed_d
        if fixed is not None:
            # A picture is drawn at d = 2 whatever its input's width. The cap above
            # bounds a d that tuning chooses, and here nothing is chosen: a PCA
            # pre-step keeping two components must not leave UMAP asked for one.
            d_max = fixed if n_fit > fixed and n_fit_refit > fixed else 0
        if d_max < 1:
            raise TuningError(f"{op} receives {p_in} feature(s): nothing to reduce")

        param = spec.tuned_param(resolved)
        criterion = spec.tuning.criterion
        nested = spec.is_visualization or spec.holds("nested_in_d", resolved)
        self.method = {
            "op": op, "spec": spec, "resolved": resolved, "param": param,
            "criterion": criterion, "nested": nested, "d_max": d_max,
            "n_fit": n_fit, "n_fit_refit": n_fit_refit,
        }
        values = self._values(op, spec, resolved, param, stage["params"], n_fit)
        self.record["method"] = {
            "op": op,
            "criterion": criterion,
            "nested_in_d": bool(nested),
            "param": param,
            "base": values["base"],
            "d_max": d_max,
            "n_fit": n_fit,
            "n_fit_refit": n_fit_refit,
        }

        if fixed is not None:
            best = self._search_fixed(values, d_max)
        elif nested and criterion in (*CURVE_CRITERIA, "eigengap"):
            best = self._search_own_criterion(values)
        elif nested:
            best = self._search_battery_profile(values)
        elif param is None:
            best = self._search_per_d()
        else:
            best = self._search_alternating(values)

        refit = self._refit_stages(best, values)
        self.record["chosen"] = {
            "d": best.d,
            "multiplier": best.multiplier if param is not None else None,
            "value": best.value if param is not None else None,
            "t": best.t,
            "score": None if best.score is None else round(best.score, 6),
            "at_grid_edge": self._at_edge(best, values),
        }
        if best.t is not None:
            self.record["chosen"]["t_at_grid_edge"] = bool(
                len(self.settings.t_grid) > 1
                and best.t in (min(self.settings.t_grid), max(self.settings.t_grid))
            )
        self.record["duration_s"] = round(time.perf_counter() - started, 4)
        return refit, self.record

    # --------------------------------------------------------- tuned values

    def _values(
        self, op: str, spec: Any, resolved: dict[str, Any], param: str | None,
        given: dict[str, Any], n_fit: int,
    ) -> dict[str, Any]:
        """The multiplier grid turned into this method's parameter values.

        A suggestion is recomputed at the row count the method is fitted on, unless
        the stage overrode it, in which case the stage's value is the centre at every
        size. A width tuned from the executor's rule is passed as `width_multiplier`.
        Whole numbers are rounded, and multipliers that round to one value merge.
        """
        multipliers = list(self.settings.multipliers)
        if param is None:
            return {"base": None, "cells": [(1.0, None, [])], "centre": None}
        if spec.tuning.base == "rule":
            centre = "explicit value" if given.get(param) is not None else "rule"
            return {
                "base": {"source": centre, "value": given.get(param)},
                "cells": [(m, m, []) for m in multipliers],
                "centre": centre,
            }

        full_suggestion = self.suggest(op, self.n_rows, resolved).get(param)
        overridden = param in given and (
            full_suggestion is None or not _close(given[param], full_suggestion)
        )
        integer = spec.params[param].type == "int"
        minimum = spec.params[param].minimum

        def value_at(rows: int, multiplier: float) -> Any:
            if overridden:
                base = given[param]
            else:
                base = self.suggest(op, rows, resolved).get(param, resolved[param])
            value = multiplier * float(base)
            if integer:
                value = int(round(value))
                if minimum is not None:
                    value = max(int(minimum), value)
            return value

        merged: dict[Any, list[float]] = {}
        for m in sorted(multipliers, key=_nearest_one):
            merged.setdefault(value_at(n_fit, m), []).append(m)
        cells = [(ms[0], value, sorted(ms[1:])) for value, ms in merged.items()]
        cells.sort(key=lambda cell: cell[0])
        self._value_at = value_at
        return {
            "base": {
                "source": "override" if overridden else "suggestion",
                "value": given[param] if overridden else self.suggest(op, n_fit, resolved).get(param),
                "at_rows": n_fit,
            },
            "cells": cells,
            "centre": "override" if overridden else "suggestion",
        }

    def _params_for(self, value: Any, d: int, *, t: int | None = None) -> dict[str, Any]:
        method = self.method
        params = dict(self.candidate.stages[self.candidate.method_position]["params"])
        params["n_components"] = int(d)
        if method["param"] is not None:
            if method["spec"].tuning.base == "rule":
                params["width_multiplier"] = float(value)
            else:
                params[method["param"]] = value
        if t is not None:
            params["t"] = int(t)
        return params

    def _infeasible(self, value: Any, d: int, multiplier: float) -> str | None:
        """Why a cell cannot run at the tuning rows or at the refit's, if it cannot."""
        method = self.method
        param = method["param"]
        checks = [("tuning", method["n_fit"], value)]
        if param is not None and method["spec"].tuning.base == "suggestion":
            checks.append(("refit", method["n_fit_refit"], self._value_at(method["n_fit_refit"], multiplier)))
        for where, rows, cell_value in checks:
            params = self._params_for(cell_value, d)
            if param == "n_neighbors" and int(params["n_neighbors"]) >= rows:
                return f"n_neighbors={params['n_neighbors']} is not below the {rows} rows at the {where}"
            if param == "perplexity" and float(params["perplexity"]) >= rows / 3:
                return f"perplexity={params['perplexity']:g} is not below n/3 at the {where}'s {rows} rows"
            for rule in method["spec"].d_limits:
                resolved, _ = self.registry.resolve_params(method["op"], params)
                problem = RULES[rule].violation(resolved)
                if problem:
                    return problem
        return None

    def _fit(self, value: Any, d: int, multiplier: float, *, criterion: bool = False) -> Any:
        """Fit the candidate's stages at one cell; None, with a reason, if it cannot."""
        problem = self._infeasible(value, d, multiplier)
        if problem:
            return None, problem, 0.0
        stages = [dict(s, params=dict(s["params"])) for s in self.candidate.stages]
        stages[self.candidate.method_position]["params"] = self._params_for(value, d)
        started = time.perf_counter()
        try:
            result = self._run(stages, criterion=criterion)
        except (ExecutionError, PipelineError) as error:
            return None, str(error), time.perf_counter() - started
        seconds = time.perf_counter() - started
        if not np.isfinite(result.embedding).all():
            return None, "the fit returned non-finite coordinates", seconds
        return result, None, seconds

    # ------------------------------------------------------------ the shapes

    def _search_fixed(self, values: dict[str, Any], d: int = 2) -> Cell:
        """d fixed -- a visualization method, or any method in a visualization run.

        Only the multiplier is tuned. Diffusion Maps keeps the t it was resolved with:
        the sweep over t belongs to choosing d, and here d is not chosen.
        """
        cells = []
        for multiplier, value, merged in values["cells"]:
            result, problem, seconds = self._fit(value, d, multiplier)
            cell = Cell(multiplier, value, d, runtime_s=seconds, merged=merged)
            if result is None:
                cell.feasible, cell.reason = False, problem
            else:
                cell.score, cell.values = self._score(result.embedding)
            cells.append(cell)
        return self._finish(cells, f"fixed at d = {d}")

    def _search_own_criterion(self, values: dict[str, Any]) -> Cell:
        """Nested in d: one fit per multiplier at the largest d, the criterion picks d."""
        method = self.method
        criterion = method["criterion"]
        d_max = method["d_max"]
        top = d_max + 1 if criterion == "eigengap" else d_max
        top = min(top, method["n_fit"] - 1, method["n_fit_refit"] - 1)
        ds = list(range(2, d_max + 1)) or [d_max]
        cells, curves = [], []
        for multiplier, value, merged in values["cells"]:
            result, problem, seconds = self._fit(value, top, multiplier, criterion=True)
            if result is None:
                cells.append(Cell(multiplier, value, d_max, feasible=False, reason=problem,
                                  runtime_s=seconds, merged=merged))
                continue
            measured = result.context.criterion or {}
            embedding = result.embedding
            for t, choice in self._own_choices(measured, ds, criterion):
                d = int(choice["d"])
                block = embedding[:, :d]
                if t is not None:
                    fitted_t = int(method["resolved"].get("t") or 1)
                    lam = np.asarray(measured["eigenvalues"][:d])
                    block = block * np.power(np.clip(lam, 1e-300, None), t - fitted_t)
                cell = Cell(multiplier, value, d, t=t, runtime_s=seconds, merged=merged)
                cell.score, cell.values = self._score(block)
                cells.append(cell)
                curves.append({"multiplier": multiplier, "t": t, **_rounded(choice)})
        self.record["criterion_choices"] = curves
        return self._finish(cells, f"{criterion}, then the battery across multipliers")

    def _own_choices(
        self, measured: dict[str, Any], ds: list[int], criterion: str
    ) -> list[tuple[int | None, dict[str, Any]]]:
        settings = self.settings
        if criterion == "eigengap":
            return [(None, choose_d_by_eigengap(measured["eigenvalues"], ds))]
        if criterion == "diffusion_distance":
            out = []
            for t in settings.t_grid:
                curve = diffusion_curve(measured["eigenvalues"], t, max(ds))
                choice = choose_d_by_curve(
                    {d: curve[d] for d in ds if d in curve},
                    flatness=settings.flatness, fallback_share=settings.fallback_share,
                )
                choice["curve"] = {d: curve[d] for d in ds if d in curve}
                out.append((t, choice))
            return out
        curve = measured["curve"]
        points = {d: curve[d - 1] for d in ds if d - 1 < len(curve)}
        choice = choose_d_by_curve(
            points, flatness=settings.flatness, fallback_share=settings.fallback_share
        )
        choice["curve"] = points
        return [(None, choice)]

    def _search_battery_profile(self, values: dict[str, Any]) -> Cell:
        """Nested in d, no criterion of its own: truncate, score, profile, elbow."""
        method = self.method
        grid = self._grid()
        cells: list[Cell] = []
        for multiplier, value, merged in values["cells"]:
            feasible_ds = []
            for d in grid:
                problem = self._infeasible(value, d, multiplier)
                if problem is None:
                    feasible_ds.append(d)
                else:
                    cells.append(Cell(multiplier, value, d, feasible=False, reason=problem,
                                      merged=merged))
            if not feasible_ds:
                continue
            top = max(feasible_ds)
            result, problem, seconds = self._fit(value, top, multiplier)
            if result is None:
                cells.append(Cell(multiplier, value, top, feasible=False, reason=problem,
                                  runtime_s=seconds, merged=merged))
                continue
            for d in feasible_ds:
                cell = Cell(multiplier, value, d, runtime_s=seconds, merged=merged)
                cell.score, cell.values = self._score(result.embedding[:, :d])
                cells.append(cell)
        return self._by_profile(cells, "the battery, profiled over multipliers")

    def _search_per_d(self) -> Cell:
        """Not nested and nothing to tune: one fit per grid d."""
        method = self.method
        own = method["criterion"] == "stress"
        cells, curve = [], {}
        for d in self._grid():
            result, problem, seconds = self._fit(None, d, 1.0, criterion=own)
            cell = Cell(1.0, None, d, runtime_s=seconds)
            if result is None:
                cell.feasible, cell.reason = False, problem
            else:
                cell.score, cell.values = self._score(result.embedding)
                if own:
                    curve[d] = float(result.context.criterion["value"])
            cells.append(cell)
        if not own:
            return self._by_profile(cells, "the battery")
        if not curve:
            raise TuningError(self._nothing_ran(cells))
        choice = choose_d_by_curve(
            curve, flatness=self.settings.flatness, fallback_share=self.settings.fallback_share
        )
        self.record["criterion_choices"] = [{**_rounded(choice), "curve": _round_map(curve)}]
        chosen = next(c for c in cells if c.d == choice["d"])
        self._record_cells(cells, "stress", choice)
        return chosen

    def _search_alternating(self, values: dict[str, Any]) -> Cell:
        """Not nested, with a parameter: alternate from reconnaissance's d (section 3.5)."""
        grid = self._grid()
        memo: dict[tuple[float, int], Cell] = {}
        cells_by_multiplier = {m: (m, v, merged) for m, v, merged in values["cells"]}

        def cell(multiplier: float, d: int) -> Cell:
            key = (multiplier, d)
            if key not in memo:
                _, value, merged = cells_by_multiplier[multiplier]
                result, problem, seconds = self._fit(value, d, multiplier)
                c = Cell(multiplier, value, d, runtime_s=seconds, merged=merged)
                if result is None:
                    c.feasible, c.reason = False, problem
                else:
                    c.score, c.values = self._score(result.embedding)
                memo[key] = c
            return memo[key]

        start = self._starting_d(grid)
        # Reconnaissance's d can be infeasible for every multiplier -- Hessian LLE at
        # d = 8 needs 45 neighbours -- while other points of the grid run. Move to the
        # nearest grid point where one does, the smaller d on a tie, rather than
        # refuse a candidate that has feasible cells.
        estimated = start
        by_distance = sorted(grid, key=lambda g: (abs(grid.index(g) - grid.index(start)), g))
        start = next(
            (g for g in by_distance if any(cell(m, g).feasible for m in cells_by_multiplier)),
            start,
        )
        multiplier = min(cells_by_multiplier, key=_nearest_one)
        d = start
        steps: list[dict[str, Any]] = []
        outcome = "stopped at the cap"
        for cycle in range(self.settings.max_cycles):
            best = None
            for m in cells_by_multiplier:
                candidate = cell(m, d)
                if candidate.feasible and _better(candidate, best):
                    best = candidate
            if best is None:
                raise TuningError(self._nothing_ran(list(memo.values())))
            steps.append({"step": "multiplier", "at_d": d, "chose": best.multiplier})
            if cycle > 0 and best.multiplier == multiplier:
                outcome = "converged"
                break
            multiplier = best.multiplier
            curve = {
                g: c.score for g in grid
                if (c := cell(multiplier, g)).feasible and c.score is not None
            }
            if not curve:
                raise TuningError(self._nothing_ran(list(memo.values())))
            choice = choose_d_by_curve(
                curve, flatness=self.settings.flatness,
                fallback_share=self.settings.fallback_share,
            )
            steps.append({"step": "d", "at_multiplier": multiplier, "chose": choice["d"],
                          "rule": choice["rule"], "capped": choice["capped"]})
            if choice["d"] == d:
                outcome = "converged"
                break
            d = int(choice["d"])
        chosen = cell(multiplier, d)
        self.record["alternating"] = {"start_d": start, "steps": steps, "outcome": outcome}
        if start != estimated:
            self.record["alternating"]["estimated_d"] = estimated
            self.record["alternating"]["start_moved"] = (
                f"no multiplier could run at reconnaissance's d = {estimated}"
            )
        if outcome == "stopped at the cap":
            self.record["alternating"]["neighbours"] = self._neighbour_check(
                chosen, grid, sorted(cells_by_multiplier), cell
            )
        self._record_cells(list(memo.values()), "alternating", None)
        return chosen

    def _neighbour_check(
        self, chosen: Cell, grid: list[int], multipliers: list[float], cell: Callable
    ) -> dict[str, Any]:
        """At the cap, score the cells around the choice, diagonals included, as a check."""
        di, mi = grid.index(chosen.d), multipliers.index(chosen.multiplier)
        scored = []
        for dj in (di - 1, di, di + 1):
            for mj in (mi - 1, mi, mi + 1):
                if 0 <= dj < len(grid) and 0 <= mj < len(multipliers) and (dj, mj) != (di, mi):
                    c = cell(multipliers[mj], grid[dj])
                    scored.append({"multiplier": c.multiplier, "d": c.d,
                                   "score": None if c.score is None else round(c.score, 6)})
        better = [s for s in scored if s["score"] is not None and chosen.score is not None
                  and s["score"] > chosen.score + 1e-12]
        return {"cells": scored, "any_better": bool(better)}

    # ------------------------------------------------------------- choosing

    def _grid(self) -> list[int]:
        d_max = self.method["d_max"]
        grid = [d for d in self.settings.d_grid if d <= d_max]
        return grid or [d_max]

    def _starting_d(self, grid: list[int]) -> int:
        estimate = self.intrinsic_dimension
        if estimate is None or not np.isfinite(estimate):
            return grid[0]
        return min(grid, key=lambda d: (abs(np.log(d) - np.log(max(estimate, 1.0))), d))

    def _by_profile(self, cells: list[Cell], how: str) -> Cell:
        """For each d the best cell, the elbow rule on that profile, then the best at it."""
        best_at: dict[int, Cell] = {}
        for c in cells:
            if c.feasible and c.score is not None and _better(c, best_at.get(c.d)):
                best_at[c.d] = c
        if not best_at:
            raise TuningError(self._nothing_ran(cells))
        choice = choose_d_by_curve(
            {d: c.score for d, c in best_at.items()},
            flatness=self.settings.flatness, fallback_share=self.settings.fallback_share,
        )
        self.record["criterion_choices"] = [
            {**_rounded(choice), "curve": _round_map({d: c.score for d, c in best_at.items()})}
        ]
        self._record_cells(cells, how, choice)
        return best_at[choice["d"]]

    def _finish(self, cells: list[Cell], how: str) -> Cell:
        best = None
        for c in cells:
            if c.feasible and _better(c, best):
                best = c
        if best is None:
            raise TuningError(self._nothing_ran(cells))
        self._record_cells(cells, how, None)
        return best

    def _record_cells(self, cells: list[Cell], how: str, choice: dict[str, Any] | None) -> None:
        self.record["search"] = how
        self.record["cells"] = [c.as_dict() for c in cells]
        if choice is not None:
            self.record["d_rule"] = {"rule": choice["rule"], "capped": choice["capped"]}

    def _nothing_ran(self, cells: list[Cell]) -> str:
        reasons = sorted({c.reason for c in cells if c.reason})
        return (
            f"no cell of {self.method['op']}'s tuning grid could run: "
            + "; ".join(reasons[:3])
            + ". Revise the candidate's stages, or reject the method with this reason."
        )

    def _at_edge(self, best: Cell, values: dict[str, Any]) -> bool:
        """A multiplier at either end of the grid: the optimum may lie outside it."""
        if self.method["param"] is None:
            return False
        grid = self.settings.multipliers
        if len(grid) < 2:
            return False
        ends = {min(grid), max(grid)}
        return best.multiplier in ends or bool(set(best.merged) & ends)

    # --------------------------------------------------------------- refit

    def _refit_stages(self, best: Cell, values: dict[str, Any]) -> list[dict[str, Any]]:
        """The candidate's stages at the chosen cell, with the suggestion recomputed at
        the rows the method is fitted on in the refit (section 3.5, settled day 15)."""
        method = self.method
        value = best.value
        if method["param"] is not None and method["spec"].tuning.base == "suggestion":
            value = self._value_at(method["n_fit_refit"], best.multiplier)
        stages = [dict(s, params=dict(s["params"])) for s in self.candidate.stages]
        stages[self.candidate.method_position]["params"] = self._params_for(
            value, best.d, t=best.t
        )
        self.record["refit_params"] = stages[self.candidate.method_position]["params"]
        return stages


def _close(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except (TypeError, ValueError):
        return a == b


def _round_map(curve: dict[int, float]) -> dict[int, float]:
    return {d: round(float(v), 6) for d, v in curve.items()}


def _rounded(choice: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in choice.items():
        if isinstance(value, dict):
            out[key] = {k: round(float(v), 6) for k, v in value.items()}
        else:
            out[key] = value
    return out


# ------------------------------------------------------------ one whole Attempt


def tune_and_refit(
    X: Matrix,
    labels: np.ndarray | None,
    stages: list[dict[str, Any]],
    *,
    n_base: int,
    seed: int,
    weights: dict[str, float],
    settings: TuningSettings,
    profile: dict[str, Any],
    recon: dict[str, Any] | None,
    registry: Registry | None = None,
) -> tuple[Any, dict[str, Any]]:
    """The Base once on every row, tuning on the Reference, then the refit.

    The Base preprocessing is fitted on every row, as the Reference is, and never on
    the tuning rows alone: a z-score or a median total fitted on 2,000 rows would make
    tuning describe a slightly different Reference. Its output stays in its own storage,
    so a sparse Reference reaches the candidate's stages sparse. The refit runs the
    candidate's stages at the chosen values on every row under the run's own seed; its
    record carries the Base's stage records and its own, as an untuned run's did.
    """
    from drtools.heuristics import suggest
    from drtools.metrics import METRIC_SAMPLE_CAP, neighbourhood_size
    from drtools.pipeline import PipelineResult

    registry = registry or load_registry()
    base_stages, own = stages[:n_base], stages[n_base:]
    if base_stages:
        base = run_pipeline(
            X, labels, base_stages, seed=seed, registry=registry,
            require_terminal_reduction=False, densify=False,
        )
        reference, base_records, base_arrays = base.embedding, base.stages, base.arrays
    else:
        reference, base_records, base_arrays = X, [], {}

    def suggester(op: str, n_rows: int, params: dict[str, Any]) -> dict[str, Any]:
        shape = {**profile.get("shape", {}), "n_samples": int(n_rows)}
        entries = suggest(op, {**profile, "shape": shape}, recon, registry, params=params)
        return {name: entry["value"] for name, entry in entries.items()}

    intrinsic = ((recon or {}).get("intrinsic_dimension") or {}).get("twonn")
    k = neighbourhood_size(min(int(reference.shape[0]), METRIC_SAMPLE_CAP))
    tuner = Tuner(
        reference, labels, own, seed=seed, weights=weights, k=k, settings=settings,
        suggest=suggester, intrinsic_dimension=intrinsic, registry=registry,
    )
    refit_stages, record = tuner.tune()

    refit = run_pipeline(reference, labels, refit_stages, seed=seed, registry=registry)
    combined = PipelineResult(
        embedding=refit.embedding,
        labels=refit.labels,
        stages=[*base_records, *refit.stages],
        context=refit.context,
        coverage=refit.coverage,
        arrays=_combined_arrays(base_arrays, refit.arrays, len(base_records), X.shape[1]),
    )
    return combined, record


def _combined_arrays(
    base: dict[str, np.ndarray], own: dict[str, np.ndarray], n_base: int, n_columns: int
) -> dict[str, np.ndarray]:
    """The refit's arrays renumbered after the Base's, its columns mapped to the cache's."""
    kept = base.get("features_kept", np.arange(n_columns))
    combined = {k: v for k, v in base.items() if k != "features_kept"}
    for key, value in own.items():
        if key == "features_kept":
            combined[key] = kept[value]
            continue
        position, name = key.split(".", 1)
        combined[f"{int(position) + n_base}.{name}"] = kept[value] if name == "features" else value
    return combined


def d_curve(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """The curve that chose a candidate's d, at the multiplier chosen, for the report.

    None for a candidate whose d was fixed -- a visualization run's -- or never tuned.
    """
    if not record or not record.get("criterion_choices") or "chosen" not in record:
        return None
    criterion = record["method"]["criterion"]
    if criterion == "fixed":
        return None
    chosen = record["chosen"]
    choices = record["criterion_choices"]
    choice = next(
        (c for c in choices
         if c.get("multiplier") == chosen.get("multiplier") and c.get("t") == chosen.get("t")),
        choices[0],
    )
    return {
        "points": {int(d): float(v) for d, v in choice["curve"].items()},
        "chosen": int(chosen["d"]),
        "rule": choice.get("rule"),
        "label": CURVE_LABELS.get(criterion, criterion.replace("_", " ")),
    }


#: What each criterion's stored curve holds. Each is stored so that higher is better,
#: which for residual variance and stress means one minus the quantity.
CURVE_LABELS = {
    "explained_variance": "cumulative explained variance",
    "kernel_variance": "cumulative kernel variance",
    "residual_variance": "1 - residual variance",
    "stress": "1 - Kruskal stress",
    "diffusion_distance": "share of diffusion distance kept",
    "battery": "weighted battery score",
}
