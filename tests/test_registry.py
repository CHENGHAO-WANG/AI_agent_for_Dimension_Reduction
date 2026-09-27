"""The method capability records, and how declared parameters become concrete values.

The registry is the agent's only knowledge of what the methods do, so the invariant
that matters most is that it describes something real: an entry with no executor is a
method the planner can select and never run.
"""

from __future__ import annotations

import openTSNE
import pytest
import umap
from sklearn.decomposition import PCA, KernelPCA, MiniBatchSparsePCA, TruncatedSVD
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding

from drtools.constraints import RULES
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


METHOD_FIELDS = {
    "preserves", "assumes", "scales_to", "handles_sparse", "roles",
    "euclidean", "nested_in_d", "requires_connected_graph", "emphasis", "new_rows",
}


def _methods(registry):
    return {**registry.reductions(), **registry.visualization_methods()}


def test_every_method_declares_the_properties_checks_read(registry) -> None:
    """A missing one is a question a check asks and the registry cannot answer."""
    for name, spec in _methods(registry).items():
        missing = METHOD_FIELDS - set(spec.raw)
        assert not missing, f"{name} does not declare {sorted(missing)}"
        assert "out_of_sample" not in spec.raw, f"{name} still declares out_of_sample"


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


def test_t_sne_and_umap_are_visualization_methods(registry) -> None:
    """Section 3.11: their output is a picture, never a representation."""
    for name in ("tsne", "umap"):
        assert registry[name].is_visualization
        assert not registry[name].is_reduction


def test_methods_lists_the_visualization_class(cli) -> None:
    result = cli("methods", "--kind", "visualization")
    assert result.code == 0
    listed = result.payload["ops"]
    assert {"tsne", "umap"} <= set(listed)
    assert all(record["kind"] == "visualization" for record in listed.values())


def test_only_a_deterministic_op_may_stand_before_another_method(registry) -> None:
    """Section 3.9. Vacuous today; it catches granting the role to LLE or Laplacian
    Eigenmaps, which are cheap and look like reasonable pre-steps."""
    for name, spec in registry.ops.items():
        if "intermediate" in spec.roles:
            assert not spec.stochastic, f"{name} is stochastic but may be intermediate"


def test_every_limit_on_d_is_named_by_some_method(registry) -> None:
    named = {rule for spec in registry.ops.values() for rule in spec.d_limits}
    assert named == set(RULES)


def test_lle_is_described_with_its_neighbour_minimum(registry) -> None:
    rules = registry.describe("lle")["d_limit_rules"]
    assert [r["name"] for r in rules] == ["lle_neighbour_minimum"]


# The library class each executor fits, for every method whose record says it can
# place new rows by `transform`. PCA's executor uses TruncatedSVD on sparse input.
TRANSFORM_CLASSES = {
    "pca": (PCA, TruncatedSVD),
    "kernel_pca": (KernelPCA,),
    "sparse_pca": (MiniBatchSparsePCA,),
    "isomap": (Isomap,),
    "lle": (LocallyLinearEmbedding,),
    "tsne": (openTSNE.TSNEEmbedding,),
    "umap": (umap.UMAP,),
}


def test_every_method_declaring_a_transform_has_one_in_its_library(registry) -> None:
    declared = {n for n, s in _methods(registry).items() if s.raw["new_rows"] == "transform"}
    assert declared == set(TRANSFORM_CLASSES)
    for name, classes in TRANSFORM_CLASSES.items():
        for cls in classes:
            assert hasattr(cls, "transform"), f"{name}: {cls.__name__} has no transform"


def test_the_methods_declared_without_a_transform_really_lack_one(registry) -> None:
    """Diffusion Maps is written in the toolbox and has none; its Nystrom extension is day 14's."""
    assert registry["laplacian_eigenmaps"].raw["new_rows"] == "nystrom"
    assert registry["diffusion_maps"].raw["new_rows"] == "nystrom"
    assert registry["mds"].raw["new_rows"] == "none"
    assert not hasattr(SpectralEmbedding, "transform")
    assert not hasattr(MDS, "transform")
