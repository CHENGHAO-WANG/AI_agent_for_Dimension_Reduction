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
            params:
              n_components: {type: int, default: 2, min: 1}
              n_neighbors: {type: int, default: 20, min: 2}
    """), encoding="utf-8")
    suggested = suggest("toy_graph", LARGE, None, registry=load_registry(path))
    assert suggested["n_neighbors"]["value"] == 20


def test_lle_starts_from_its_registry_default_and_stays_capped() -> None:
    assert suggest("lle", LARGE)["n_neighbors"]["value"] == 12


def test_an_op_without_n_neighbors_gets_no_neighbour_suggestion() -> None:
    assert "n_neighbors" not in suggest("pca", LARGE)


def test_an_unknown_op_gets_nothing() -> None:
    assert suggest("not_an_op", LARGE) == {}
