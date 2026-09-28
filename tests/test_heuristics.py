"""Suggested starting values, and which ops receive them."""

from __future__ import annotations

import textwrap

from drtools.heuristics import suggest
from drtools.registry import load_registry

LARGE = {"shape": {"n_samples": 5000}}


def test_any_op_with_n_neighbors_gets_the_neighbour_suggestion(tmp_path) -> None:
    """A new neighbour-graph method needs no edit here to be offered one."""
    path = tmp_path / "registry.yaml"
    path.write_text(textwrap.dedent("""\
        version: 1
        ops:
          toy_graph:
            kind: visualization
            summary: a toy neighbour-graph method
            tuning: {param: n_neighbors, base: suggestion, criterion: fixed}
            params:
              n_components: {type: int, default: 2, min: 1}
              n_neighbors: {type: int, default: 20, min: 2}
    """), encoding="utf-8")
    suggested = suggest("toy_graph", LARGE, None, registry=load_registry(path))
    assert suggested["n_neighbors"]["value"] == 20


def test_lle_starts_from_its_registry_default_and_stays_capped() -> None:
    assert suggest("lle", LARGE)["n_neighbors"]["value"] == 12


DISCONNECTED = {"neighbourhood": {"k": 15, "n_connected_components": 3}}
UNEVEN = {"neighbourhood": {"k": 15, "n_connected_components": 1,
                            "density_ratio_p95_p05": 40.0}}


def test_lle_is_not_raised_for_a_disconnected_graph_and_its_rationale_says_so() -> None:
    """Defect 11: the shared rule raised k to 45, the cap clamped it to 12, and the
    rationale still said 45 -- a configuration already known to fail."""
    entry = suggest("lle", LARGE, DISCONNECTED)["n_neighbors"]
    assert entry["value"] == 12
    assert "45" not in entry["rationale"]
    assert "disconnected" in entry["rationale"] or "components" in entry["rationale"]
    assert "recon.neighbourhood.n_connected_components" in entry["evidence"]


def test_lle_is_not_raised_for_uneven_density() -> None:
    assert suggest("lle", LARGE, UNEVEN)["n_neighbors"]["value"] == 12


def test_lle_respects_the_hessian_minimum_for_the_d_it_is_asked_about() -> None:
    entry = suggest("lle", LARGE, params={"method": "hessian", "n_components": 4})
    assert entry["n_neighbors"]["value"] == 15
    assert "outside" in entry["n_neighbors"]["rationale"]


def test_lle_small_d_keeps_its_default() -> None:
    entry = suggest("lle", LARGE, params={"method": "hessian", "n_components": 2})
    assert entry["n_neighbors"]["value"] == 12


def test_other_graph_methods_are_still_raised_for_a_disconnected_graph() -> None:
    assert suggest("isomap", LARGE, DISCONNECTED)["n_neighbors"]["value"] == 45


def test_an_op_without_n_neighbors_gets_no_neighbour_suggestion() -> None:
    assert "n_neighbors" not in suggest("pca", LARGE)


def test_an_unknown_op_gets_nothing() -> None:
    assert suggest("not_an_op", LARGE) == {}
