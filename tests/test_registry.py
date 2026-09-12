"""The method capability records, and how declared parameters become concrete values.

The registry is the agent's only knowledge of what the methods do, so the invariant
that matters most is that it describes something real: an entry with no executor is a
method the planner can select and never run.
"""

from __future__ import annotations

import pytest

from drtools.executors import EXECUTORS
from drtools.registry import RegistryError, load_registry


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_every_declared_op_has_an_implementation(registry) -> None:
    """An entry with no executor is a method the planner can pick and never run."""
    assert sorted(registry.ops) == sorted(EXECUTORS)


def test_every_op_declares_what_it_is_for(registry) -> None:
    for name, spec in registry.ops.items():
        assert spec.summary, f"{name} has no summary for the planner to read"


def test_reductions_declare_the_properties_selection_depends_on(registry) -> None:
    """These are the fields the planner reasons over; a missing one is a blind spot."""
    required = {"preserves", "assumes", "scales_to", "handles_sparse", "roles"}
    for name, spec in registry.reductions().items():
        missing = required - set(spec.raw)
        assert not missing, f"{name} does not declare {sorted(missing)}"


def test_defaults_are_used_and_recorded_as_defaults(registry) -> None:
    params, provenance = registry.resolve_params("pca", {})

    assert params == {"n_components": 2, "whiten": False}
    assert provenance == {
        "n_components": "registry_default",
        "whiten": "registry_default",
    }


def test_supplied_values_are_recorded_as_specified(registry) -> None:
    """The report's claim that the agent chose a value rests on this distinction."""
    params, provenance = registry.resolve_params("pca", {"n_components": 50})

    assert params["n_components"] == 50
    assert provenance["n_components"] == "specified"
    assert provenance["whiten"] == "registry_default"


def test_json_floats_are_coerced_to_the_declared_integer_type(registry) -> None:
    """JSON has one number type; scikit-learn does not."""
    params, _ = registry.resolve_params("pca", {"n_components": 10.0})

    assert params["n_components"] == 10
    assert isinstance(params["n_components"], int)


def test_a_fractional_value_for_an_integer_parameter_is_rejected(registry) -> None:
    with pytest.raises(RegistryError, match="whole number"):
        registry.resolve_params("pca", {"n_components": 2.5})


def test_unknown_parameters_are_rejected_rather_than_ignored(registry) -> None:
    """Silently dropping it would leave the report describing a setting never applied."""
    with pytest.raises(RegistryError, match="perplexity"):
        registry.resolve_params("pca", {"perplexity": 30})


def test_values_outside_declared_bounds_are_rejected(registry) -> None:
    with pytest.raises(RegistryError, match="<= 1"):
        registry.resolve_params("diffusion_maps", {"alpha": 5.0})


def test_values_outside_declared_choices_are_rejected(registry) -> None:
    with pytest.raises(RegistryError, match="must be one of"):
        registry.resolve_params("kernel_pca", {"kernel": "gaussian"})


def test_unknown_ops_name_what_is_available(registry) -> None:
    with pytest.raises(RegistryError, match="pca"):
        registry["definitely_not_a_method"]


def test_terminal_methods_are_marked_as_terminal(registry) -> None:
    assert registry["pca"].can_be_intermediate()
    assert not registry["diffusion_maps"].can_be_intermediate()
    assert not registry["laplacian_eigenmaps"].can_be_intermediate()


def test_sparse_capability_matches_what_the_executors_accept(registry) -> None:
    assert registry["pca"].handles_sparse
    assert not registry["standardise"].handles_sparse
    assert not registry["diffusion_maps"].handles_sparse
