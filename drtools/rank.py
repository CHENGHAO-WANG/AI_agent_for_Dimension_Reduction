"""Turning metrics into a ranking, under weights declared in advance.

The mechanism is deliberately dull: normalise each metric onto its own absolute scale,
multiply by the declared weight, add them up. All of the judgment lives in the weights,
and the weights were fixed in `plan.json` before any embedding existed. That ordering is
the whole point. Choosing a weighting after seeing the results would let the agent pick
whichever emphasis crowned the candidate that happened to win, and the rationale would
read exactly the same in the report as an honest one.

Runtime is the one metric with no absolute scale — two seconds is fast or slow only
relative to the alternatives — so it is scaled within the cohort, and that difference is
recorded rather than glossed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drtools.metrics import METRIC_SPECS

WEIGHT_TOLERANCE = 1e-6


class RankingError(ValueError):
    """The weighting is unusable, or there is nothing to rank."""


def rank_candidates(
    metrics_by_id: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    justification: str = "",
    failures: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score and order candidates. `metrics_by_id` holds only successful candidates."""
    if not metrics_by_id:
        raise RankingError(
            "no candidate produced an embedding to rank; every one failed or timed out"
        )
    _check_weights(weights)

    available = _available_metrics(metrics_by_id)
    effective, dropped = _effective_weights(weights, available)

    runtime_scale = _runtime_scale(metrics_by_id)

    scored = []
    for candidate_id, metrics in metrics_by_id.items():
        contributions: dict[str, dict[str, Any]] = {}
        total = 0.0
        for metric, weight in effective.items():
            raw = metrics["values"].get(metric)
            normalised = (
                _normalise_runtime(raw, runtime_scale)
                if metric == "runtime_s"
                else METRIC_SPECS[metric].normalise(raw)
            )
            if normalised is None:
                continue
            total += weight * normalised
            contributions[metric] = {
                "raw": raw,
                "normalised": round(normalised, 4),
                "weight": round(weight, 4),
                "contribution": round(weight * normalised, 4),
            }
        scored.append(
            {
                "id": candidate_id,
                "score": round(total, 4),
                "contributions": contributions,
            }
        )

    scored.sort(key=lambda row: row["score"], reverse=True)
    for position, row in enumerate(scored, start=1):
        row["rank"] = position

    return {
        "ranking": scored,
        "winner": scored[0]["id"],
        "weights_declared": weights,
        "weights_applied": {k: round(v, 4) for k, v in effective.items()},
        "weights_dropped": dropped,
        "justification": justification,
        "failed_candidates": sorted(failures or {}),
        "notes": _notes(scored, dropped, runtime_scale, failures or {}),
    }


def _check_weights(weights: dict[str, float]) -> None:
    if not weights:
        raise RankingError("no weights were declared")

    unknown = sorted(set(weights) - set(METRIC_SPECS))
    if unknown:
        raise RankingError(
            f"unknown metric(s) in the weighting: {unknown}. The battery is "
            f"{', '.join(sorted(METRIC_SPECS))}."
        )
    negative = sorted(name for name, value in weights.items() if value < 0)
    if negative:
        raise RankingError(
            f"negative weight(s) on {negative}; direction is already declared by each "
            "metric, so a negative weight would invert a metric's meaning silently"
        )
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-3:
        raise RankingError(
            f"weights sum to {total:.4f} rather than 1.0, so scores would not be "
            "comparable across runs"
        )


def _available_metrics(metrics_by_id: dict[str, dict[str, Any]]) -> set[str]:
    """Metrics that produced a value for every candidate.

    A metric present for some candidates and absent for others cannot be part of a fair
    comparison, so it is dropped for all of them rather than scored where convenient.
    """
    per_candidate = [
        {name for name, value in metrics["values"].items() if value is not None}
        for metrics in metrics_by_id.values()
    ]
    return set.intersection(*per_candidate) if per_candidate else set()


def _effective_weights(
    weights: dict[str, float], available: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Renormalise the declared weights over the metrics that could be computed."""
    usable = {name: value for name, value in weights.items() if name in available}
    dropped = {name: value for name, value in weights.items() if name not in available}

    total = sum(usable.values())
    if total <= 0:
        raise RankingError(
            "none of the weighted metrics could be computed for every candidate; "
            f"missing: {sorted(dropped)}"
        )
    return {name: value / total for name, value in usable.items()}, dropped


def _runtime_scale(metrics_by_id: dict[str, dict[str, Any]]) -> tuple[float, float]:
    times = [
        metrics["values"].get("runtime_s")
        for metrics in metrics_by_id.values()
        if metrics["values"].get("runtime_s") is not None
    ]
    return (min(times), max(times)) if times else (0.0, 0.0)


def _normalise_runtime(value: float | None, scale: tuple[float, float]) -> float | None:
    """Fastest candidate scores 1, slowest 0. Cohort-relative by necessity."""
    if value is None:
        return None
    fastest, slowest = scale
    if slowest <= fastest:
        return 1.0
    return float(np.clip(1.0 - (value - fastest) / (slowest - fastest), 0.0, 1.0))


def _notes(
    scored: list[dict[str, Any]],
    dropped: dict[str, float],
    runtime_scale: tuple[float, float],
    failures: dict[str, dict[str, Any]],
) -> list[str]:
    notes: list[str] = []

    if dropped:
        share = sum(dropped.values())
        notes.append(
            f"{share:.0%} of the declared weight was on {sorted(dropped)}, which could "
            "not be computed for every candidate and was redistributed across the rest. "
            "The ranking therefore answers a slightly narrower question than the "
            "weighting intended, and the comparison should be read in that light."
        )

    if len(scored) > 1:
        first, second = scored[0], scored[1]
        margin = first["score"] - second["score"]
        if margin < 0.02:
            notes.append(
                f"{first['id']} and {second['id']} are separated by {margin:.4f}, which "
                "is inside the noise of stochastic methods and repeated subsampling. "
                "They should be treated as tied rather than ranked."
            )

    fastest, slowest = runtime_scale
    if slowest > 0 and slowest / max(fastest, 1e-9) > 10:
        notes.append(
            f"runtimes span {fastest:.2f}s to {slowest:.2f}s, so any weight on runtime "
            "dominates the comparison between the extremes; it is scaled within this "
            "cohort because runtime has no absolute best value."
        )

    if failures:
        notes.append(
            f"{len(failures)} candidate(s) produced no embedding and are excluded from "
            f"the ranking rather than scored as zero: {sorted(failures)}. Exclusion is "
            "not a judgement on the method, only on this configuration of it."
        )

    return notes
