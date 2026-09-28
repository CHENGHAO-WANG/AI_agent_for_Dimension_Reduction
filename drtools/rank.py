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

Fidelity rises with d, so the score alone would reward whichever candidate demanded the
most dimensions (section 3.7). The comparison is therefore by a non-inferiority margin:
the candidates scoring within the margin of the leader are close competitors, and among
them the fewest dimensions win. The margin is declared in the plan, 0.02 by default,
and frozen at registration. Every comparison reads the scores as rounded here, to four
places, so what a reader checks against the table is what was decided.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from drtools.metrics import METRIC_SPECS

WEIGHT_TOLERANCE = 1e-6

#: Section 3.7's non-inferiority margin on the weighted score: "2 points out of 100".
DEFAULT_MARGIN = 0.02
#: The widest margin registration allows, whatever the argument: the flatness threshold
#: at which one candidate's curve is called flat. A wider margin would treat as
#: negligible between candidates more than is negligible within one.
MAX_MARGIN = 0.10
#: Above this many close competitors the ranking has not discriminated, and the note
#: says so in one sentence rather than listing them (section 3.8).
COMPETITOR_CAP = 3
#: Scores are compared as rounded to four places; differences below this are equal.
_EQUAL = 1e-9

#: Section 2.4's defaults, one pair per focus (settled on day 17). Trustworthiness and
#: continuity carry the local share equally, the Shepard correlation the global one;
#: each focus moves 0.2 from one to the other. A plan whose weights match its recorded
#: focus's default needs no evidence for the emphasis, unless the data has labels: then
#: which default applies says whether the labels are trusted, and that is a decision.
#: Anything else is a departure and must cite evidence.
FOCUS_SHARES: dict[str, tuple[float, float, float]] = {
    "local": (0.35, 0.35, 0.30),
    "balanced": (0.25, 0.25, 0.50),
    "global": (0.15, 0.15, 0.70),
}
#: With trusted labels seventy per cent stays on unsupervised fidelity, in the focus's
#: own proportions; the labels are an outside check on the representation, not its goal.
UNSUPERVISED_SHARE_WITH_LABELS = 0.70
LABEL_WEIGHTS = {"knn_label_preservation": 0.20, "silhouette": 0.10}


def default_weightings(focus: str = "balanced") -> dict[str, dict[str, float]]:
    """The two defaults for a focus: without trusted labels, and with them."""
    trust, cont, shepard = FOCUS_SHARES[focus]
    share = UNSUPERVISED_SHARE_WITH_LABELS
    return {
        "default": {
            "trustworthiness": trust,
            "continuity": cont,
            "shepard_correlation": shepard,
        },
        "default_trusted_labels": {
            "trustworthiness": round(share * trust, 6),
            "continuity": round(share * cont, 6),
            "shepard_correlation": round(share * shepard, 6),
            **LABEL_WEIGHTS,
        },
    }


#: The balanced defaults, the weighting a run with no stated focus is held to.
DEFAULT_WEIGHTINGS = default_weightings("balanced")


def matching_default(weights: dict[str, float], focus: str = "balanced") -> str | None:
    """The default for `focus` these weights are, to within rounding, or None.

    Matched rather than declared: a field naming the default would restate the weights
    and could disagree with them. A zero weight is the same as an absent one. Another
    focus's default does not match: the focus answer would otherwise move nothing.
    """
    given = {name: value for name, value in weights.items() if value != 0}
    for name, default in default_weightings(focus).items():
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
    margin: float = DEFAULT_MARGIN,
    justification: str = "",
    failures: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score, compare and order candidates. `metrics_by_id` holds only successful ones.

    Each metrics record carries its Embedding's `d` and, where `evaluate` could compute
    them, the jackknife replicates the standard errors are built from.
    """
    if not metrics_by_id:
        raise RankingError(
            "no candidate produced an embedding to rank; every one failed or timed out"
        )
    _check_weights(weights)
    missing_d = sorted(cid for cid, record in metrics_by_id.items() if "d" not in record)
    if missing_d:
        raise RankingError(
            f"{missing_d} carry no d, so the comparison cannot weigh their dimensions. "
            "Their metrics predate the rule; run `drtools evaluate` again for each."
        )

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
                "d": int(metrics["d"]),
                "score": round(total, 4),
                "contributions": contributions,
            }
        )

    leader = min(scored, key=lambda row: (-row["score"], row["d"], row["id"]))
    close = [row for row in scored if leader["score"] - row["score"] <= margin + _EQUAL]
    rest = [row for row in scored if row not in close]
    # The close set in the order of the rule that chose the winner, so rank k is the
    # candidate the rule would choose were ranks 1 to k-1 withdrawn: the leader comes
    # last among them and so stays in the comparison throughout.
    close.sort(key=lambda row: (row["d"], -row["score"], row["id"]))
    rest.sort(key=lambda row: (-row["score"], row["d"], row["id"]))
    ordered = close + rest
    winner = ordered[0]

    replicates, unavailable = _replicate_scores(metrics_by_id, effective)
    for position, row in enumerate(ordered, start=1):
        row["rank"] = position
        row["close_competitor"] = position > 1 and position <= len(close)
        row["difference_from_winner"] = round(row["score"] - winner["score"], 4)
        row["se"] = _jackknife_se(replicates.get(row["id"]))
        row["se_difference"] = (
            None
            if position == 1
            else _jackknife_se(
                _paired(replicates.get(row["id"]), replicates.get(winner["id"]))
            )
        )

    competitors = [
        {
            "id": row["id"],
            "d": row["d"],
            "score": row["score"],
            "difference_from_winner": row["difference_from_winner"],
            "se_difference": row["se_difference"],
        }
        for row in close[1:]
    ]
    path = path_of_winners(ordered)

    return {
        "ranking": ordered,
        "winner": winner["id"],
        "leader": leader["id"],
        "margin": margin,
        "close_competitors": competitors,
        "discriminated": len(competitors) <= COMPETITOR_CAP,
        "path": path["path"],
        "wins_at_no_rate": path["wins_at_no_rate"],
        "path_sentences": path_sentences(path, ordered, winner["id"]),
        "weights_declared": weights,
        "weights_applied": {k: round(v, 4) for k, v in effective.items()},
        "weights_dropped": dropped,
        "justification": justification,
        "failed_candidates": sorted(failures or {}),
        "notes": _notes(
            winner, leader, competitors, margin, dropped, failures or {}, replicates,
            unavailable,
        ),
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


# ------------------------------------------------------------- standard errors


def _replicate_scores(
    metrics_by_id: dict[str, dict[str, Any]], effective: dict[str, float]
) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Each candidate's weighted score in every jackknife replicate, or why there is none.

    The replicates are scored under the weights the ranking applies, so a metric
    dropped from the ranking is dropped from them too. Candidates are paired only if
    their groups are the same, which `rank` checks before it gets here.
    """
    scores: dict[str, list[float]] = {}
    unavailable: dict[str, str] = {}
    for candidate_id, record in metrics_by_id.items():
        jackknife = record.get("jackknife") or {}
        replicates = jackknife.get("replicates")
        if not replicates:
            unavailable[candidate_id] = jackknife.get(
                "unavailable", "no jackknife replicates were recorded"
            )
            continue
        values = []
        for replicate in replicates:
            normalised = {
                name: METRIC_SPECS[name].normalise(replicate.get(name))
                for name in effective
            }
            lost = sorted(name for name, value in normalised.items() if value is None)
            if lost:
                unavailable[candidate_id] = (
                    f"{lost} could not be computed with a group left out"
                )
                break
            values.append(sum(effective[name] * normalised[name] for name in effective))
        else:
            scores[candidate_id] = values
    return scores, unavailable


def _paired(first: list[float] | None, second: list[float] | None) -> list[float] | None:
    """Replicate-by-replicate differences; each pair left out the same rows."""
    if first is None or second is None or len(first) != len(second):
        return None
    return [a - b for a, b in zip(first, second)]


def _jackknife_se(values: list[float] | None) -> float | None:
    """sqrt((G - 1) / G * sum_g (S_g - mean)^2), the delete-a-group standard error."""
    if not values or len(values) < 2:
        return None
    array = np.asarray(values, dtype=np.float64)
    groups = array.size
    spread = float(np.sum((array - array.mean()) ** 2))
    return round(math.sqrt((groups - 1) / groups * spread), 4)


# ------------------------------------------------------------ the path of winners


def path_of_winners(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Who wins at every value lambda >= 0 of one dimension, maximising S - lambda * d.

    The winner changes at the slopes of the upper convex hull of the points (d, S), as
    along a lasso path. At lambda = 0 it is the leader; as lambda grows it moves to
    fewer dimensions, ending at the best score among the fewest. A tie at a breakpoint
    goes to the fewer dimensions, so each range of rates includes its lower end and
    excludes its upper one. A candidate under the hull, or on an edge between two
    corners, wins at no rate.
    """
    leader = min(ranked, key=lambda row: (-row["score"], row["d"], row["id"]))
    best_at: dict[int, dict[str, Any]] = {}
    for row in sorted(ranked, key=lambda row: (-row["score"], row["id"])):
        if row["d"] <= leader["d"] and row["d"] not in best_at:
            best_at[row["d"]] = row
    points = [best_at[d] for d in sorted(best_at)]

    hull: list[dict[str, Any]] = []
    for point in points:
        # Drop the last corner while it lies on or under the chord to the new point.
        while (
            len(hull) >= 2
            and _slope(hull[-2], hull[-1]) <= _slope(hull[-1], point) + _EQUAL
        ):
            hull.pop()
        hull.append(point)

    # From the leader at lambda = 0 towards the fewest dimensions.
    corners = list(reversed(hull))
    path = []
    for position, corner in enumerate(corners):
        lower = 0.0 if position == 0 else _slope(corner, corners[position - 1])
        upper = (
            None
            if position == len(corners) - 1
            else _slope(corners[position + 1], corner)
        )
        path.append(
            {
                "id": corner["id"],
                "d": corner["d"],
                "score": corner["score"],
                "from_rate": round(lower, 6),
                "to_rate": None if upper is None else round(upper, 6),
            }
        )
    on_path = {entry["id"] for entry in path}
    return {
        "path": path,
        "wins_at_no_rate": [row["id"] for row in ranked if row["id"] not in on_path],
    }


def _slope(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Score gained per dimension from `left` to `right`, which has more of them."""
    return (right["score"] - left["score"]) / (right["d"] - left["d"])


def _rate(value: float) -> str:
    """A rate to two significant figures: the scores are precise to about 0.005."""
    return np.format_float_positional(
        value, precision=2, unique=False, fractional=False, trim="-"
    )


def _names(ids: list[str], conjunction: str = "and") -> str:
    if len(ids) == 1:
        return ids[0]
    return ", ".join(ids[:-1]) + f" {conjunction} " + ids[-1]


def path_sentences(
    path: dict[str, Any], ranked: list[dict[str, Any]], winner: str
) -> list[str]:
    """The path of winners in words, and what the margin rule's choice amounts to."""
    steps = path["path"]
    sentences = []
    if len(steps) == 1:
        sentences.append(
            f"{steps[0]['id']} has the highest score, and no candidate has fewer "
            "dimensions, so it wins at every value of a dimension."
        )
    else:
        sentences.append(
            f"With no value placed on a dimension, {steps[0]['id']} "
            f"(d = {steps[0]['d']}) scores highest."
        )
        clauses = []
        for position, step in enumerate(steps[1:], start=1):
            who = f"{step['id']} (d = {step['d']})"
            rate = _rate(step["from_rate"])
            clauses.append(
                f"if one dimension were worth more than {rate} of score, {who} would win"
                if position == 1
                else f"above {rate}, {who} would"
            )
        sentences.append(clauses[0][0].upper() + "; ".join(clauses)[1:] + ".")
    if path["wins_at_no_rate"]:
        verb = "wins" if len(path["wins_at_no_rate"]) == 1 else "win"
        sentences.append(f"{_names(path['wins_at_no_rate'])} {verb} at no rate.")

    chosen = next((step for step in steps if step["id"] == winner), None)
    if chosen is None:
        sentences.append(
            f"The margin rule chose {winner}, which wins at no rate: at every value of "
            f"a dimension, {_names(_beaten_by(steps, ranked, winner), 'or')} scores "
            "higher net of its dimensions."
        )
        return sentences
    lower, upper = chosen["from_rate"], chosen["to_rate"]
    if len(steps) == 1:
        where = "at every value of a dimension"
    elif upper is None:
        where = f"whenever one dimension is worth more than {_rate(lower)} of score"
    elif lower == 0:
        where = f"while one dimension is worth less than {_rate(upper)} of score"
    else:
        where = (
            f"when one dimension is worth between {_rate(lower)} and {_rate(upper)} "
            "of score"
        )
    sentences.append(f"The margin rule chose {winner}, which wins {where}.")
    return sentences


def _beaten_by(
    steps: list[dict[str, Any]], ranked: list[dict[str, Any]], winner: str
) -> list[str]:
    """The hull corners that beat an off-hull candidate at every rate.

    Under the hull edge from corner L to corner R, of slope s, R scores higher net of
    its dimensions at every rate up to s, and L at every rate from s. A candidate with
    more dimensions than the leader is beaten by the leader alone, and one at a
    corner's d by that corner.
    """
    row = next(r for r in ranked if r["id"] == winner)
    corners = sorted(steps, key=lambda step: step["d"])
    if row["d"] >= corners[-1]["d"]:
        return [corners[-1]["id"]]
    same = [corner["id"] for corner in corners if corner["d"] == row["d"]]
    if same:
        return same
    left = max((c for c in corners if c["d"] < row["d"]), key=lambda c: c["d"])
    right = min((c for c in corners if c["d"] > row["d"]), key=lambda c: c["d"])
    return [right["id"], left["id"]]


# ------------------------------------------------------------------ the notes


#: Every kind of note `rank` writes. The report places each one by its kind, and a
#: test holds the two lists together, so a new kind cannot fall out of the report
#: unnoticed -- which is what selecting notes by their wording allowed (defect 8).
NOTE_KINDS = (
    "weights_dropped",
    "failed_candidates",
    "close_competitors",
    "not_discriminated",
    "standard_error_scope",
    "standard_error_unavailable",
)


def _note(kind: str, text: str) -> dict[str, str]:
    """A ranking note: the report places it by `kind` and prints `text` as written."""
    assert kind in NOTE_KINDS, kind
    return {"kind": kind, "text": text}


def _notes(
    winner: dict[str, Any],
    leader: dict[str, Any],
    competitors: list[dict[str, Any]],
    margin: float,
    dropped: dict[str, float],
    failures: dict[str, dict[str, Any]],
    replicates: dict[str, list[float]],
    unavailable: dict[str, str],
) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []

    if dropped:
        share = sum(dropped.values())
        notes.append(_note(
            "weights_dropped",
            f"{share:.0%} of the declared weight was on {sorted(dropped)}, which could "
            "not be computed for every candidate and was redistributed across the rest. "
            "The ranking therefore answers a slightly narrower question than the "
            "weighting intended, and the comparison should be read in that light.",
        ))

    if len(competitors) > COMPETITOR_CAP:
        notes.append(_note(
            "not_discriminated",
            f"{len(competitors) + 1} candidates lie within {margin} of the leader, "
            f"{leader['id']}, so the ranking did not discriminate among them. "
            f"{winner['id']} is named because it has the fewest dimensions "
            f"(d = {winner['d']}). That is a finding about the portfolio or the "
            "weighting more than about any one candidate.",
        ))
    elif competitors:
        listed = "; ".join(
            f"{c['id']} (d = {c['d']}, {c['difference_from_winner']:+.4f}"
            + (f", SE {c['se_difference']:.4f}" if c["se_difference"] is not None else "")
            + ")"
            for c in competitors
        )
        if winner is leader:
            text = (
                f"{winner['id']} leads, and no candidate within {margin} of it has fewer "
                f"dimensions (d = {winner['d']}). Its close competitors, each with its "
                f"score minus the winner's: {listed}. Each lies within the margin, so "
                f"the ranking does not separate it from {winner['id']}."
            )
        else:
            text = (
                f"Among the candidates within {margin} of the leader, {leader['id']}, "
                f"the rule chose {winner['id']}, which has the fewest dimensions "
                f"(d = {winner['d']}). Its close competitors, each with its score minus "
                f"the winner's: {listed}. The ranking separates none of them from "
                f"{winner['id']} by more than the margin, so the choice among them "
                "rests on dimensions, not on fidelity."
            )
        notes.append(_note("close_competitors", text))

    if replicates:
        notes.append(_note(
            "standard_error_scope",
            "Standard errors come from a grouped jackknife over the scored rows, in ten "
            "groups that are the same for every candidate, so each difference from the "
            "winner has its own paired standard error. They hold each fitted Embedding "
            "fixed and measure only which rows were scored, so they exclude seed and "
            "refit variability and are lower bounds. They are reported and do not enter "
            "the choice.",
        ))
    by_reason: dict[str, list[str]] = {}
    for candidate_id, reason in sorted(unavailable.items()):
        by_reason.setdefault(reason, []).append(candidate_id)
    for reason, ids in by_reason.items():
        whose = "Its score is" if len(ids) == 1 else "Their scores are"
        notes.append(_note(
            "standard_error_unavailable",
            f"No standard error for {_names(ids)}: {reason}. {whose} unaffected.",
        ))

    if failures:
        notes.append(_note(
            "failed_candidates",
            f"{len(failures)} candidate(s) produced no embedding and are excluded from "
            f"the ranking rather than scored as zero: {sorted(failures)}. Exclusion is "
            "not a judgement on the method, only on this configuration of it.",
        ))

    return notes


def weighted_score(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """One embedding's score under a weighting, as `rank_candidates` computes it.

    The weights are renormalised over the metrics that could be computed, and a
    metric that could not is left out. Tuning scores its cells with this (section
    3.5); every cell of one candidate is scored on the same rows, so the same metrics
    are available to all of them.
    """
    usable = {
        name: weight
        for name, weight in weights.items()
        if weight > 0 and METRIC_SPECS[name].normalise(values.get(name)) is not None
    }
    total = sum(usable.values())
    if total <= 0:
        return None
    return sum(
        weight / total * METRIC_SPECS[name].normalise(values.get(name))
        for name, weight in usable.items()
    )
