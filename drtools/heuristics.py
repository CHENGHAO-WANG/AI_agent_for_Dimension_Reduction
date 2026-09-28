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
from drtools.registry import Registry, load_registry


def suggest_base(
    profile: dict[str, Any], recon: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Base preprocessing the planner may adopt, from reconnaissance's own rule.

    This is a suggestion crossing a boundary, not an identification. A Probe
    representation exists so that measurements describe the data rather than an
    artefact of scale; it is discarded once the measuring is done and never produces an
    Embedding. Base preprocessing is part of the analysis, is chosen by the agent, and
    its output survives as the Reference. Day 6 settled that these are genuinely two
    things and that collapsing them would have been the wrong fix.

    What they share is the question. Both answer "what transform makes distances on
    this data meaningful", so the rule that settles one is the honest default for the
    other -- and an override the report can describe needs something concrete to
    override.
    """
    from drtools.recon import choose_probe_representation

    transform, reason = choose_probe_representation(profile)
    evidence = ["profile.values.suspected_kind", "profile.features.std_ratio_p95_p05"]
    if recon is not None:
        evidence.append("recon.probe_representation.reason")

    return {
        "stages": [{"op": step, "params": {}} for step in transform],
        "rationale": reason,
        "evidence": evidence,
        "note": (
            "A suggestion, derived from the same rule reconnaissance used to choose its "
            "probe representation -- not that representation itself, which is discarded "
            "once the measuring is done. Adopt it, or override it with a logged reason."
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
    """For a PCA stage feeding a neighbour embedding, take the spectrum's elbow."""
    if recon is None:
        return {}
    spectrum = recon.get("spectrum", {}).get("probe", {})
    elbow = spectrum.get("elbow")
    ninety = spectrum.get("n_components_for_90pct")
    if elbow is None:
        return {}

    value = int(elbow if ninety is None else max(elbow, min(ninety, 50)))
    return {
        "n_components": _entry(
            value,
            f"the cumulative variance curve bends at {elbow} components"
            + (f" and reaches 90% by {ninety}" if ninety else "")
            + f", so {value} retains the structure without carrying the noise tail. "
            "This is the value for an intermediate reduction; a terminal PCA for "
            "plotting needs 2.",
            ["recon.spectrum.probe.elbow", "recon.spectrum.probe.n_components_for_90pct"],
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
