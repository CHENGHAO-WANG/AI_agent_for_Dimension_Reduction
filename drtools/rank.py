"""Turning metrics into a ranking, under weights declared in advance.

The mechanism is deliberately dull: normalise each metric onto its own absolute scale,
multiply by the declared weight, add them up. All of the judgment lives in the weights,
and the weights were fixed in `plan.json` before any embedding existed. That ordering is
the whole point. Choosing a weighting after seeing the results would let the agent pick
whichever emphasis crowned the candidate that happened to win, and the rationale would
read exactly the same in the report as an honest one.

Runtime is measured and reported but never weighted (section 2.4): scaled within the
cohort it let a candidate that could not win reverse the order of two that could, and by
the time ranking happens its cost is already paid. So every weighted metric is on an
absolute scale.
"""

from __future__ import annotations

from typing import Any


from drtools.metrics import METRIC_SPECS

WEIGHT_TOLERANCE = 1e-6

#: Section 2.4's two defaults. A plan whose weights match one needs no evidence for the
#: emphasis, unless the data has labels: then which default applies says whether the
#: labels are trusted, and that is a decision. Anything else is a departure and must
#: cite evidence. The focus weightings join on day 17, with the question that sets them.
DEFAULT_WEIGHTINGS: dict[str, dict[str, float]] = {
    "default": {
        "trustworthiness": 0.25,
        "continuity": 0.25,
        "shepard_correlation": 0.5,
    },
    # Seventy per cent stays on unsupervised fidelity in its own 1 : 1 : 2 proportions;
    # the labels are an outside check on the representation, not its goal.
    "default_trusted_labels": {
        "trustworthiness": 0.175,
        "continuity": 0.175,
        "shepard_correlation": 0.35,
        "knn_label_preservation": 0.20,
        "silhouette": 0.10,
    },
}


def matching_default(weights: dict[str, float]) -> str | None:
    """The default these weights are, to within rounding, or None for a departure.

    Matched rather than declared: a field naming the default would restate the weights
    and could disagree with them. A zero weight is the same as an absent one.
    """
    given = {name: value for name, value in weights.items() if value != 0}
    for name, default in DEFAULT_WEIGHTINGS.items():
        if set(given) == set(default) and all(
            abs(given[metric] - value) <= WEIGHT_TOLERANCE
            for metric, value in default.items()
        ):
            return name
    return None


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

    scored = []
    for candidate_id, metrics in metrics_by_id.items():
        contributions: dict[str, dict[str, Any]] = {}
        total = 0.0
        for metric, weight in effective.items():
            raw = metrics["values"].get(metric)
            normalised = METRIC_SPECS[metric].normalise(raw)
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
        "notes": _notes(scored, dropped, failures or {}),
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
    unweightable = sorted(name for name in weights if not METRIC_SPECS[name].weightable)
    if unweightable:
        raise RankingError(
            f"{unweightable} cannot carry weight. Runtime is measured and reported for "
            "every candidate, never weighted: by the time ranking happens the cost is "
            "already paid, it varies on replay, and it rewards a candidate for looking "
            "at less data. Put its weight on the fidelity metrics."
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


def _notes(
    scored: list[dict[str, Any]],
    dropped: dict[str, float],
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

    if failures:
        notes.append(
            f"{len(failures)} candidate(s) produced no embedding and are excluded from "
            f"the ranking rather than scored as zero: {sorted(failures)}. Exclusion is "
            "not a judgement on the method, only on this configuration of it."
        )

    return notes
