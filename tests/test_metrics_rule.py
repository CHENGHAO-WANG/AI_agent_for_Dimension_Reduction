import numpy as np
import pytest

from drtools.metrics import evaluate_embedding, neighbourhood_size

LOCAL = ("trustworthiness", "continuity", "shepard_correlation")


def _pair(n, d_ref=5, d_emb=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d_ref)), rng.normal(size=(n, d_emb))


@pytest.mark.parametrize("n", [3, 4, 5, 6, 19, 20, 30, 31, 32, 1000])
def test_the_rule_satisfies_sklearns_constraint(n):
    assert neighbourhood_size(n) < n / 2


def test_the_rule_saturates_at_fifteen():
    assert neighbourhood_size(1000) == 15
    assert neighbourhood_size(31) == 15
    assert neighbourhood_size(30) == 14


@pytest.mark.parametrize("n", [20, 31, 200])
def test_local_metrics_are_produced_above_the_floor(n):
    reference, embedding = _pair(n)
    values = evaluate_embedding(reference, embedding)["values"]
    assert all(values[name] is not None for name in LOCAL)


@pytest.mark.parametrize("n", [2, 3, 4, 19])
def test_local_metrics_are_absent_below_the_floor(n):
    reference, embedding = _pair(n)
    values = evaluate_embedding(reference, embedding)["values"]
    assert all(values[name] is None for name in LOCAL)


def test_a_small_dataset_scores_instead_of_raising():
    """The reproduction: k=15 against 20 rows used to raise ValueError."""
    reference, embedding = _pair(20)
    result = evaluate_embedding(reference, embedding)
    assert result["values"]["trustworthiness"] is not None


def test_silhouette_is_absent_when_every_row_is_its_own_class():
    reference, embedding = _pair(4)
    values = evaluate_embedding(reference, embedding, np.arange(4))["values"]
    assert values["silhouette"] is None


def test_silhouette_is_present_when_classes_are_valid():
    reference, embedding = _pair(60)
    values = evaluate_embedding(reference, embedding, np.arange(60) % 3)["values"]
    assert values["silhouette"] is not None
