"""Profile-derived starting values for hyperparameters.

Library defaults are written for no dataset in particular, which is why t-SNE's
perplexity of 30 is offered just as readily for sixty points as for six hundred
thousand. These suggestions are derived from the profile and the reconnaissance probes
instead, and each one carries its reasoning and the keys it rests on.

They are suggestions, not decisions. The planner may override any of them, and when it
does the override is recorded as `specified` while these show as `registry_default` —
which is exactly the distinction the report needs in order to say whether a value was
chosen or merely inherited.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def suggest(
    op: str, profile: dict[str, Any], recon: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Suggested parameters for `op` on this data, each with a rationale."""
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

    if op in {"umap", "laplacian_eigenmaps", "isomap", "lle"}:
        suggestions.update(_neighbour_suggestion(op, n, recon))

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
    op: str, n: int, recon: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    base = {"lle": 10, "isomap": 10, "laplacian_eigenmaps": 15, "umap": 15}[op]
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

    if op == "lle":
        reasons.append(
            "LLE is the most sensitive of these to this parameter: measured on a Swiss "
            "roll it recovers the manifold at k between 6 and 12 and collapses by k = 24, "
            "so it should stay at the low end"
        )
        value = int(min(value, 12))

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
        )
    }


def _entry(value: Any, rationale: str, evidence: list[str]) -> dict[str, Any]:
    return {"value": value, "rationale": rationale, "evidence": evidence}
