"""Every method claiming nested_in_d is nested, checked rather than assumed.

Section 3.5: a method is nested in d when its d-dimensional embedding equals the first
d columns of its embedding at a larger d. Slicing a method that is not nested understates
its fidelity at the smaller d, so the claim is tested for every op that makes it,
including a claim that holds only under a condition (LLE's standard variant).
"""

from __future__ import annotations

import numpy as np
import pytest

from drtools.loaders import load
from drtools.pipeline import run_pipeline
from drtools.registry import load_registry


@pytest.fixture(scope="module")
def padded_roll():
    """A noisy Swiss roll with five low-variance columns added, so d = 5 is possible."""
    X, labels, _ = load("swiss_roll", n_samples=400, noise=0.3)
    padding = 0.1 * np.random.default_rng(0).normal(size=(X.shape[0], 5))
    return np.hstack([X, padding]), labels


def _claims() -> list[tuple[str, dict]]:
    registry = load_registry()
    methods = {**registry.reductions(), **registry.visualization_methods()}
    claims: list[tuple[str, dict]] = []
    for op, spec in methods.items():
        condition = spec.conditions["nested_in_d"]
        if condition is True:
            claims.append((op, {}))
        elif isinstance(condition, dict):
            claims += [(op, {p: v}) for p, values in condition.items() for v in values]
    return claims


def _leading_agreement(X, labels, op: str, params: dict) -> float:
    small = run_pipeline(X, labels, [{"op": op, "params": {"n_components": 2, **params}}], seed=0)
    large = run_pipeline(X, labels, [{"op": op, "params": {"n_components": 5, **params}}], seed=0)
    return min(
        abs(np.corrcoef(small.embedding[:, j], large.embedding[:, j])[0, 1]) for j in range(2)
    )


@pytest.mark.parametrize("op, params", _claims())
def test_a_method_claiming_nested_in_d_is_nested(padded_roll, op, params) -> None:
    assert _leading_agreement(*padded_roll, op, params) > 0.9999


def test_the_comparison_can_fail(padded_roll) -> None:
    """Negative control: modified LLE, declared not nested, scored 0.91 on day 11."""
    assert _leading_agreement(*padded_roll, "lle", {"method": "modified"}) < 0.99
