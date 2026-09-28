"""Profile-derived starting values for hyperparameters.

Library defaults are written for no dataset in particular, which is why t-SNE's
perplexity of 30 is offered just as readily for sixty points as for six hundred
thousand. These suggestions are derived from the profile and the reconnaissance probes
instead, and each one carries its reasoning and the keys it rests on.

They are suggestions, not decisions. `suggest-params` persists each one in the run, and
registration compares every value a plan gives against it: a value equal to it is
recorded as `suggested`, a different one as `overridden` -- refused unless the stage
gives a reason -- a value nothing was suggested for as `specified`, and one left unset
as `registry_default`. That is the distinction the report needs in order to say whether
a value followed the data, departed from it, or was merely inherited.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drtools.constraints import lle_neighbour_minimum
from drtools.decision import base_rule, recorded_decision, rule_reason
from drtools.registry import Registry, load_registry


def suggest_base(
    profile: dict[str, Any], recon: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The base preprocessing section 3.10's rule gives for the recorded data decision.

    The Probe representation is this same rule applied, and the two remain different
    things: the probe is fixed by the rule and discarded once the measuring is done,
    while the base preprocessing is the agent's, may depart from the rule with a
    reason and evidence, and survives as the Reference.
    """
    decision = recorded_decision(recon)
    if decision is None:
        raise ValueError("no data decision is recorded; run recon with --decision first")
    return {
        "stages": base_rule(decision),
        "rationale": rule_reason(decision),
        "evidence": ["recon.data_decision", *decision.evidence],
        "note": (
            "A suggestion: the rule's base for the recorded data decision, not the "
            "Probe representation itself, which is discarded once the measuring is "
            "done. Adopt it, or depart from it with base_departure, a reason and "
            "evidence; drop_constant stays first either way."
        ),
    }


def suggest(
    op: str,
    profile: dict[str, Any],
    recon: dict[str, Any] | None = None,
    registry: Registry | None = None,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Suggested parameters for `op` on this data, each with a rationale.

    `params` are the other settings the stage will run with, for a suggestion that
    depends on them -- LLE's neighbour minimum grows with its variant and with d.
    Unset ones take their registry defaults.
    """
    registry = registry or load_registry()
    spec = registry.ops.get(op)
    n = int(profile.get("shape", {}).get("n_samples", 0))
    suggestions: dict[str, dict[str, Any]] = {}

    if op == "tsne":
        perplexity = float(np.clip(n / 100, 5, 50))
        suggestions["perplexity"] = _entry(
            perplexity,
            f"perplexity is an effective neighbourhood size, so it should scale with "
            f"the data: n/100 clipped to [5, 50] gives {perplexity:g} at n = {n:,}. "
            "The library default of 30 is independent of n and is badly wrong at both "
            "ends of the range.",
            ["profile.shape.n_samples"],
        )

    # Any op with a neighbourhood size gets the size-aware suggestion, starting from its
    # own registry default, so a new neighbour-graph method is offered one unedited.
    # LLE has its own rule: the shared one raises k for connectivity and density, which
    # collides with the range LLE works in and produced a clamped value whose rationale
    # still named the raised one.
    if op == "lle" and spec is not None:
        suggestions.update(_lle_suggestion(spec, params or {}, recon))
    elif spec is not None and "n_neighbors" in spec.params:
        base = int(spec.params["n_neighbors"].default)
        suggestions.update(_neighbour_suggestion(base, n, recon))

    if op == "pca":
        suggestions.update(_component_suggestion(recon))

    if op == "diffusion_maps":
        suggestions["alpha"] = _entry(
            1.0,
            "alpha = 1 divides out the sampling density, so the operator approximates "
            "the Laplace-Beltrami operator of the manifold and the result describes the "
            "geometry rather than how densely it happened to be sampled.",
            [],
        )

    return suggestions


def _neighbour_suggestion(
    base: int, n: int, recon: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    value = int(np.clip(base if n >= 1000 else max(5, n // 50), 5, 50))
    reasons = [f"a neighbourhood of {value} is a reasonable starting point at n = {n:,}"]
    evidence = ["profile.shape.n_samples"]

    if recon is not None:
        graph = recon.get("neighbourhood", {})
        probe_k = graph.get("k")
        parts = graph.get("n_connected_components")
        density = graph.get("density_ratio_p95_p05")

        if parts and parts > 1 and probe_k:
            value = int(min(50, max(value, probe_k * 3)))
            reasons = [
                f"reconnaissance found the k = {probe_k} graph split into {parts} "
                f"components, and graph-based methods need it connected, so this is "
                f"raised to {value}. If it still fails, the data may genuinely be "
                "disconnected rather than under-connected."
            ]
            evidence.append("recon.neighbourhood.n_connected_components")
        elif density is not None and density > 10:
            value = int(min(50, round(value * 1.5)))
            reasons.append(
                f"neighbourhood radii vary {density:,.0f}-fold, so density is strongly "
                f"non-uniform and a larger k ({value}) stabilises the graph at some cost "
                "in local detail"
            )
            evidence.append("recon.neighbourhood.density_ratio_p95_p05")

    return {"n_neighbors": _entry(value, " ".join(reasons), evidence)}


#: Where LLE recovers the manifold: measured on a Swiss roll, it does at k between 6 and
#: 12 and collapses by k = 24.
LLE_RANGE = (6, 12)


def _lle_suggestion(
    spec: Any, params: dict[str, Any], recon: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """The larger of LLE's neighbour minimum and its default, kept within its range.

    Never raised for connectivity or density: a larger k is what makes LLE collapse.
    A disconnected graph is said in the rationale and warned of by the validator, and
    the value suggested is the value the rationale names.
    """
    method = str(params.get("method") or spec.params["method"].default)
    d = int(params.get("n_components") or spec.params["n_components"].default)
    minimum = lle_neighbour_minimum(method, d)
    low, high = LLE_RANGE
    value = int(np.clip(max(minimum, int(spec.params["n_neighbors"].default)), low, high))
    evidence: list[str] = []

    if minimum > high:
        value = minimum
        reasons = [
            f"lle(method={method!r}) needs at least {minimum} neighbours for "
            f"{d} components, so {minimum} is suggested. That is outside the range "
            f"measured to work, k between {low} and {high} on a Swiss roll, with "
            "collapse by k = 24: a lower d, or another variant, is the safer choice."
        ]
    else:
        reasons = [
            f"{value} neighbours: LLE recovers the manifold at k between {low} and "
            f"{high} on a Swiss roll and collapses by k = 24, so it stays at the low "
            f"end, above its minimum of {minimum} for method={method!r} at d = {d}."
        ]

    graph = (recon or {}).get("neighbourhood", {})
    probe_k, parts = graph.get("k"), graph.get("n_connected_components")
    if parts and parts > 1 and probe_k:
        reasons.append(
            f"Reconnaissance found the k = {probe_k} graph split into {parts} "
            "components, and LLE needs it connected. It is not raised to connect it, "
            "since a larger k is what makes LLE collapse: on this data LLE is likely "
            "unsuitable, and a rejection citing the disconnection is the honest record."
        )
        evidence.append("recon.neighbourhood.n_connected_components")

    return {"n_neighbors": _entry(value, " ".join(reasons), evidence)}


def _component_suggestion(recon: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """For a PCA pre-step: the rule that chooses its k, read on the probe spectrum.

    Since day 15 a PCA before a method has its k chosen inside the Attempt by PCA's own
    criterion, the shared elbow procedure on cumulative variance. This reports what the
    same rule gives on reconnaissance's probe, so the suggestion and the choice cannot
    disagree by construction; the choice is made again on the candidate's own input.
    """
    from drtools.tuning import (
        D_CAP,
        DEFAULT_FALLBACK_SHARE,
        DEFAULT_FLATNESS,
        choose_d_by_curve,
    )

    if recon is None:
        return {}
    cumulative = recon.get("spectrum", {}).get("probe", {}).get("cumulative") or []
    points = {k: float(cumulative[k - 1]) for k in range(2, min(len(cumulative), D_CAP) + 1)}
    if not points:
        return {}
    choice = choose_d_by_curve(
        points, flatness=DEFAULT_FLATNESS, fallback_share=DEFAULT_FALLBACK_SHARE
    )
    value = int(choice["d"])
    return {
        "n_components": _entry(
            value,
            f"PCA's own criterion on the probe's cumulative variance gives {value} "
            f"components (rule: {choice['rule']}), which keeps "
            f"{points[value]:.0%} of the variance. A PCA before a method has its k "
            "chosen by this rule on the candidate's own input when it runs, so this "
            "value is a preview, not a setting.",
            ["recon.spectrum.probe.cumulative"],
            applies_to="intermediate",
        )
    }


def _entry(
    value: Any, rationale: str, evidence: list[str], applies_to: str | None = None
) -> dict[str, Any]:
    """One suggestion. `applies_to: intermediate` limits it to a stage with a method after
    it, so registration does not read a terminal stage's own value as overriding it."""
    entry = {"value": value, "rationale": rationale, "evidence": evidence}
    if applies_to is not None:
        entry["applies_to"] = applies_to
    return entry
