"""The comparison a visualization run makes instead of a ranking (section 3.11, day 17).

A visualization run computes no weighted total: under a weighting heavy on local
structure t-SNE or UMAP would win, under one heavy on global structure PCA or MDS, so
the total would decide the method by itself. What still compares is each metric on its
own, since every one is on an absolute scale and computed identically for every
candidate. So this states, metric by metric, who is best and who lies within the margin
of the best, and nothing more. Which candidates suit the run's focus is the agent's
judgment, recorded by `recommend` and citing these facts.
"""

from __future__ import annotations

from typing import Any

from drtools.metrics import METRIC_SPECS

#: Differences below this are equal, as in `rank`.
_EQUAL = 1e-9


def compare_candidates(
    metrics_by_id: dict[str, dict[str, Any]],
    *,
    margin: float,
    order: list[str],
) -> dict[str, Any]:
    """Every weightable metric, candidate by candidate, with its best and near-best.

    `order` is the fixed order candidates are listed in: the linear baseline first, then
    the plan's. Values are compared rounded to four places, as the report prints them.
    A metric no candidate has a value for is left out.
    """
    candidates = [cid for cid in order if cid in metrics_by_id]
    metrics: dict[str, Any] = {}
    for name, spec in METRIC_SPECS.items():
        if not spec.weightable:
            continue
        values = {
            cid: round(float(value), 4)
            for cid in candidates
            if (value := metrics_by_id[cid]["values"].get(name)) is not None
        }
        if not values:
            continue
        sign = 1 if spec.higher_is_better else -1
        ranked = sorted(values, key=lambda cid: (-sign * values[cid], candidates.index(cid)))
        best = values[ranked[0]]
        metrics[name] = {
            "values": values,
            "order": ranked,
            "best": ranked[0],
            "within_margin": [
                cid for cid in ranked if sign * (best - values[cid]) <= margin + _EQUAL
            ],
            "measures": spec.measures,
        }
    return {"candidates": candidates, "margin": margin, "metrics": metrics}
