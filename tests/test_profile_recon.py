"""Profiling and reconnaissance, graded against fixtures of known structure.

These are the first agent-behaviour tests rather than unit tests. The fixtures were
built with their answers known, so the question here is not "does the code run" but
"does the evidence the planner will read actually describe the data": a Swiss roll has
to come back as a connected 2-D manifold, and separated blobs have to come back as a
fragmented graph, or every decision downstream is made on false premises.
"""

from __future__ import annotations

import numpy as np
import pytest

from drtools.loaders import load
from drtools.profile import profile_dataset
from drtools.recon import reconnaissance
from drtools.runs import MISSING, resolve_evidence

FIXTURES = ["swiss_roll", "s_curve", "blobs", "linear_subspace", "sparse_counts"]


def build(name: str, **kwargs):
    X, labels, meta = load(name, **kwargs)
    profile = profile_dataset(X, labels, meta)
    return X, labels, meta, profile


# ----------------------------------------------------------------------- profile


@pytest.mark.parametrize("name", FIXTURES)
def test_every_observation_cites_evidence_that_resolves(name: str) -> None:
    """A citation that does not resolve is worse than no citation: it reads as proof."""
    _, _, _, profile = build(name, n_samples=300) if name != "sparse_counts" else build(
        name, n_cells=200
    )

    resolved = resolve_evidence(
        [key for note in profile["observations"] for key in note["evidence"]],
        {"profile": profile},
    )

    unresolved = [key for key, value in resolved.items() if value is MISSING]
    assert not unresolved, f"{name}: evidence keys do not resolve: {unresolved}"
    assert all(note["evidence"] for note in profile["observations"])


def test_count_data_is_recognised_as_count_data() -> None:
    _, _, _, profile = build("sparse_counts", n_cells=200)

    assert profile["values"]["suspected_kind"] == "counts"
    assert profile["values"]["sparsity"] > 0.5
    assert profile["samples"]["total_ratio_max_min"] > 3

    text = " ".join(note["observation"] for note in profile["observations"])
    assert "log1p" in text
    assert "normalisation" in text


def test_continuous_data_is_not_mistaken_for_counts() -> None:
    _, _, _, profile = build("swiss_roll", n_samples=300)

    assert profile["values"]["suspected_kind"] == "continuous"
    assert profile["values"]["is_nonnegative"] is False


def test_constant_features_are_reported_so_scaling_can_avoid_them() -> None:
    X, labels, meta = load("blobs", n_samples=200)
    X = np.hstack([X, np.zeros((X.shape[0], 3))])

    profile = profile_dataset(X, labels, meta)

    assert profile["features"]["n_constant"] == 3
    assert any("constant" in n["observation"] for n in profile["observations"])


def test_unlabelled_data_says_so_rather_than_inventing_labels() -> None:
    X, _, meta = load("swiss_roll", n_samples=200)

    profile = profile_dataset(X, None, meta)

    assert profile["labels"]["present"] is False
    assert any("No labels" in n["observation"] for n in profile["observations"])


# --------------------------------------------------------------------- recon


@pytest.mark.parametrize("name", ["swiss_roll", "s_curve"])
def test_two_dimensional_manifolds_are_measured_as_low_dimensional(name: str) -> None:
    X, labels, _, profile = build(name, n_samples=800)

    recon = reconnaissance(X, labels, profile)

    assert recon["intrinsic_dimension"]["twonn"] < 4.0
    assert recon["neighbourhood"]["n_connected_components"] == 1
    assert recon["spectrum"]["probe"]["elbow"] == 2


def test_separated_clusters_fragment_the_neighbourhood_graph() -> None:
    """The condition that makes spectral methods fail, caught before they are chosen."""
    X, labels, _, profile = build("blobs", n_samples=600, n_clusters=5)

    recon = reconnaissance(X, labels, profile)

    assert recon["neighbourhood"]["n_connected_components"] == 5
    assert recon["neighbourhood"]["largest_component_fraction"] < 0.5
    assert any(
        "Isomap" in note["observation"] for note in recon["observations"]
    ), "a fragmented graph must warn against graph-based spectral methods"


def test_low_rank_data_shows_its_rank_in_the_elbow() -> None:
    X, labels, _, profile = build("linear_subspace", n_samples=600, rank=5)

    recon = reconnaissance(X, labels, profile)

    assert recon["spectrum"]["probe"]["elbow"] == 5
    assert recon["spectrum"]["probe"]["n_components_for_95pct"] <= 6


def test_count_data_is_probed_after_normalisation_not_before() -> None:
    X, labels, _, profile = build("sparse_counts", n_cells=400)

    recon = reconnaissance(X, labels, profile)

    assert recon["probe_representation"]["transform"] == ["normalise_total", "log1p"]
    assert "raw" in recon["spectrum"], "both representations are needed to compare them"


@pytest.mark.parametrize("name", FIXTURES)
def test_recon_observations_cite_evidence_that_resolves(name: str) -> None:
    X, labels, _, profile = (
        build(name, n_samples=400) if name != "sparse_counts" else build(name, n_cells=300)
    )
    recon = reconnaissance(X, labels, profile)

    resolved = resolve_evidence(
        [key for note in recon["observations"] for key in note["evidence"]],
        {"recon": recon, "profile": profile},
    )

    unresolved = [key for key, value in resolved.items() if value is MISSING]
    assert not unresolved, f"{name}: evidence keys do not resolve: {unresolved}"


def test_subsampling_is_stratified_and_declared() -> None:
    """Small classes must survive the probe, and the report must say it happened."""
    X, labels, _, profile = build("blobs", n_samples=2000, n_clusters=5)

    recon = reconnaissance(X, labels, profile, max_samples=500)

    assert recon["subsample"]["n_used"] <= 505
    assert recon["subsample"]["stratified"] is True
    assert any("subsampl" in n["observation"] for n in recon["observations"])


def test_recon_is_deterministic_under_a_fixed_seed() -> None:
    X, labels, _, profile = build("blobs", n_samples=400)

    first = reconnaissance(X, labels, profile, seed=3)
    second = reconnaissance(X, labels, profile, seed=3)

    assert first["intrinsic_dimension"] == second["intrinsic_dimension"]
    assert first["spectrum"]["probe"]["cumulative"] == second["spectrum"]["probe"]["cumulative"]
