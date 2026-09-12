"""Contract tests for the synthetic fixtures.

These fixtures are the daily smoke test for the whole pipeline, so their own
guarantees — the loader contract, determinism, and the structural properties the
agent-behaviour tests will later assert against — have to hold first.
"""

from __future__ import annotations

import numpy as np
import pytest

from drtools.synthetic import GENERATORS, generate

NAMES = sorted(GENERATORS)


@pytest.mark.parametrize("name", NAMES)
def test_satisfies_loader_contract(name: str) -> None:
    X, labels, meta = generate(name, seed=0)

    assert X.ndim == 2
    assert X.dtype == np.float64
    assert np.isfinite(X).all()
    assert labels is not None and labels.shape == (X.shape[0],)
    assert meta["name"] == name
    assert set(meta["expected"]) >= {"structure", "manifold_methods_beat_linear"}


@pytest.mark.parametrize("name", NAMES)
def test_is_deterministic_given_a_seed(name: str) -> None:
    first, first_labels, _ = generate(name, seed=7)
    second, second_labels, _ = generate(name, seed=7)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_labels, second_labels)


@pytest.mark.parametrize("name", NAMES)
def test_seed_actually_changes_the_data(name: str) -> None:
    baseline, _, _ = generate(name, seed=0)
    other, _, _ = generate(name, seed=1)

    assert not np.array_equal(baseline, other)


def test_count_fixture_looks_like_count_data() -> None:
    """Non-negative integers, mostly zero, heavy-tailed library sizes.

    These are the properties that make raw Euclidean distances misleading, which is
    the decision the agent is supposed to get right on this fixture.
    """
    X, _, meta = generate("sparse_counts", seed=0)

    assert (X >= 0).all()
    np.testing.assert_array_equal(X, np.round(X))
    assert (X == 0).mean() > 0.5
    assert meta["expected"]["requires_normalisation"] is True

    library_sizes = X.sum(axis=1)
    assert library_sizes.max() / library_sizes.min() > 3.0


def test_low_rank_fixture_has_the_advertised_spectrum() -> None:
    """The recon pass should find a clean elbow here; check one exists to find."""
    X, _, meta = generate("linear_subspace", seed=0)
    rank = meta["expected"]["intrinsic_dim"]

    spectrum = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    explained = np.cumsum(spectrum**2) / np.sum(spectrum**2)

    assert explained[rank - 1] > 0.95
    assert spectrum[rank - 1] / spectrum[rank] > 3.0


def test_unknown_dataset_names_are_rejected_helpfully() -> None:
    with pytest.raises(KeyError, match="unknown synthetic dataset"):
        generate("not_a_dataset")
