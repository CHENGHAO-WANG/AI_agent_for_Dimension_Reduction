"""Limits on d that a capability record cannot hold as a number.

Measured on day 11: scikit-learn refuses Hessian LLE below its minimum, and fits the
other three variants one below the toolbox's d + 1. So Hessian's minimum is the
library's and the others' is mathematical, and the tests say which is which.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

from drtools.constraints import RULES, lle_neighbour_minimum

VARIANTS = ["standard", "modified", "hessian", "ltsa"]


@pytest.fixture(scope="module")
def roll() -> np.ndarray:
    X, _ = make_swiss_roll(300, noise=0.05, random_state=0)
    return np.hstack([X, np.random.default_rng(0).normal(size=(300, 7))])


def _fit(X: np.ndarray, method: str, d: int, k: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LocallyLinearEmbedding(
            n_components=d, n_neighbors=k, method=method,
            eigen_solver="dense", random_state=0,
        ).fit(X)


def test_the_minimum_grows_with_d_and_depends_on_the_variant() -> None:
    assert [lle_neighbour_minimum(m, 2) for m in VARIANTS] == [3, 3, 6, 3]
    assert lle_neighbour_minimum("hessian", 8) == 45


@pytest.mark.parametrize("d", [2, 4])
@pytest.mark.parametrize("method", VARIANTS)
def test_every_variant_fits_in_the_library_at_its_minimum(roll, method, d) -> None:
    """Our minimum is never below the library's, or the rule would pass a doomed stage."""
    _fit(roll, method, d, lle_neighbour_minimum(method, d))


@pytest.mark.parametrize("d", [2, 4])
def test_hessians_minimum_is_the_librarys_own(roll, d) -> None:
    with pytest.raises(ValueError, match="hessian"):
        _fit(roll, "hessian", d, lle_neighbour_minimum("hessian", d) - 1)


def test_the_rule_refuses_below_the_minimum_and_names_it() -> None:
    message = RULES["lle_neighbour_minimum"].violation(
        {"method": "hessian", "n_components": 4, "n_neighbors": 10}
    )
    assert message is not None
    assert "15" in message and "n_neighbors" in message


def test_the_rule_accepts_a_stage_at_its_minimum() -> None:
    rule = RULES["lle_neighbour_minimum"]
    assert rule.violation({"method": "hessian", "n_components": 4, "n_neighbors": 15}) is None
    assert rule.violation({"method": "standard", "n_components": 2, "n_neighbors": 3}) is None


def test_every_rule_has_a_sentence_for_the_agent() -> None:
    for name, rule in RULES.items():
        assert rule.name == name
        assert len(rule.sentence) > 40
