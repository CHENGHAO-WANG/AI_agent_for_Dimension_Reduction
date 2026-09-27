"""What the registry accepts, and what it refuses at load.

Each test writes a small registry of its own, so the rules are pinned independently of
what today's methods happen to declare.
"""

from __future__ import annotations

import textwrap

import pytest

from drtools.registry import RegistryError, load_registry

TOY_PARAMS = """\
    params:
      n_components: {type: int, default: 2, min: 1}
      kernel: {type: str, default: rbf, choices: [rbf, poly]}
"""


def _registry(tmp_path, body: str, name: str = "registry.yaml"):
    path = tmp_path / name
    path.write_text(
        "version: 1\nops:\n" + textwrap.indent(textwrap.dedent(body), "  "),
        encoding="utf-8",
    )
    return load_registry(path)


def _toy(extra: str = "", kind: str = "reduction") -> str:
    return (
        f"toy:\n  kind: {kind}\n  summary: a toy method\n"
        + textwrap.indent(textwrap.dedent(extra), "  ")
        + textwrap.indent(textwrap.dedent(TOY_PARAMS), "  ")
    )


def test_a_visualization_method_is_its_own_kind(tmp_path) -> None:
    registry = _registry(tmp_path, _toy(kind="visualization"))
    spec = registry["toy"]
    assert spec.is_visualization and not spec.is_reduction
    assert list(registry.visualization_methods()) == ["toy"]
    assert registry.reductions() == {}


def test_an_unknown_kind_is_refused_naming_the_three(tmp_path) -> None:
    with pytest.raises(RegistryError, match="preprocessing, reduction, visualization"):
        _registry(tmp_path, _toy(kind="embedding"))


def test_a_property_may_hold_always_never_or_under_a_condition(tmp_path) -> None:
    spec = _registry(tmp_path, _toy("""\
        euclidean: {when: {kernel: [rbf]}}
        nested_in_d: true
        requires_connected_graph: false
    """))["toy"]
    assert spec.holds("nested_in_d") is True
    assert spec.holds("requires_connected_graph") is False
    assert spec.holds("euclidean", {"kernel": "rbf"}) is True
    assert spec.holds("euclidean", {"kernel": "poly"}) is False


def test_a_condition_reads_the_default_when_the_stage_leaves_the_parameter_unset(tmp_path) -> None:
    spec = _registry(tmp_path, _toy("euclidean: {when: {kernel: [rbf]}}\n"))["toy"]
    assert spec.holds("euclidean", {}) is True


def test_asking_for_an_undeclared_property_is_an_error_not_false(tmp_path) -> None:
    spec = _registry(tmp_path, _toy())["toy"]
    with pytest.raises(RegistryError, match="does not declare 'euclidean'"):
        spec.holds("euclidean")


@pytest.mark.parametrize(
    "condition, message",
    [
        ("{when: {metric: [euclidean]}}", "does not declare"),
        ("{when: {n_components: [2]}}", "declares no choices"),
        ("{when: {kernel: [gaussian]}}", "not among"),
        ("{when: {kernel: []}}", "non-empty list"),
        ("{kernel: [rbf]}", "expected true, false"),
        ("sometimes", "expected true, false"),
    ],
)
def test_a_malformed_condition_stops_the_registry_loading(tmp_path, condition, message) -> None:
    with pytest.raises(RegistryError, match=message):
        _registry(tmp_path, _toy(f"euclidean: {condition}\n"))


def test_a_null_emphasis_needs_a_reason(tmp_path) -> None:
    with pytest.raises(RegistryError, match="emphasis_reason"):
        _registry(tmp_path, _toy("emphasis: null\n"))
    spec = _registry(
        tmp_path, _toy("emphasis: null\nemphasis_reason: depends on the width\n"), "b.yaml"
    )["toy"]
    assert spec.raw["emphasis"] is None


def test_an_unknown_emphasis_or_route_to_new_rows_is_refused(tmp_path) -> None:
    with pytest.raises(RegistryError, match="emphasis"):
        _registry(tmp_path, _toy("emphasis: medium\n"))
    with pytest.raises(RegistryError, match="new_rows"):
        _registry(tmp_path, _toy("new_rows: approximate\n"), "b.yaml")


def test_an_unknown_limit_on_d_is_refused_and_a_known_one_is_described(tmp_path) -> None:
    with pytest.raises(RegistryError, match="constraints.py"):
        _registry(tmp_path, _toy("d_limits: [no_such_rule]\n"))
    registry = _registry(tmp_path, _toy("d_limits: [lle_neighbour_minimum]\n"), "b.yaml")
    rules = registry.describe("toy")["d_limit_rules"]
    assert rules[0]["name"] == "lle_neighbour_minimum"
    assert "n_neighbors" in rules[0]["sentence"]


def _toy_with_gamma(rule: str) -> str:
    return (
        "toy:\n  kind: reduction\n  summary: a toy method\n  params:\n"
        "    n_components: {type: int, default: 2, min: 1}\n"
        "    kernel: {type: str, default: rbf, choices: [rbf, poly, cosine]}\n"
        "    gamma:\n      type: float\n      default: null\n"
        + textwrap.indent(textwrap.dedent(rule), "      ")
    )


def test_a_default_rule_names_the_label_each_branch_records(tmp_path) -> None:
    spec = _registry(tmp_path, _toy_with_gamma("""\
        default_rule:
          - {when: {kernel: [rbf]}, rule: median heuristic}
          - {when: {kernel: [poly]}, rule: library default}
          - {when: {kernel: [cosine]}, rule: not applicable}
    """))["toy"]
    assert spec.default_rule("gamma", {"kernel": "poly"}) == "library default"
    assert spec.default_rule("gamma", {}) == "median heuristic"


def test_a_plain_default_rule_applies_everywhere(tmp_path) -> None:
    spec = _registry(tmp_path, _toy_with_gamma("default_rule: median heuristic\n"))["toy"]
    assert spec.default_rule("gamma", {"kernel": "cosine"}) == "median heuristic"


def test_a_default_rule_that_misses_a_choice_stops_the_registry_loading(tmp_path) -> None:
    """Review focus 5: a kernel added to the choices without a branch is caught at load."""
    with pytest.raises(RegistryError, match="cosine"):
        _registry(tmp_path, _toy_with_gamma("""\
            default_rule:
              - {when: {kernel: [rbf]}, rule: median heuristic}
              - {when: {kernel: [poly]}, rule: library default}
        """))


def test_a_default_rule_that_names_a_choice_twice_is_refused(tmp_path) -> None:
    with pytest.raises(RegistryError, match="more than once"):
        _registry(tmp_path, _toy_with_gamma("""\
            default_rule:
              - {when: {kernel: [rbf, poly]}, rule: median heuristic}
              - {when: {kernel: [poly, cosine]}, rule: library default}
        """))


def test_a_null_default_without_a_rule_stops_the_registry_loading(tmp_path) -> None:
    body = (
        "toy:\n  kind: reduction\n  summary: a toy method\n  params:\n"
        "    width: {type: float, default: null}\n"
    )
    with pytest.raises(RegistryError, match="default_rule"):
        _registry(tmp_path, body)
