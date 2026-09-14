"""The published neighbourhood rule, and what the battery does at the edges of it.

`neighbourhood_size` stays: it is how `prepare-reference` picks the number once, from
the reference's row count. What left is the battery deriving its own k per candidate —
`evaluate_embedding` now requires the caller to say which neighbourhood this cohort is
measured at, so these tests pass the k the rule would have given for their row count
rather than letting the battery pick one behind their backs.
"""

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


def test_the_battery_has_no_neighbourhood_of_its_own():
    """k is required and has no default: an omitted one is a TypeError, not a guess.

    The old signature derived k from whatever rows the candidate happened to keep,
    which is how two candidates that subsampled differently were scored at different
    neighbourhoods and ranked together. An optional override would have left that
    available to the next caller; required-and-explicit does not.
    """
    reference, embedding = _pair(60)
    with pytest.raises(TypeError):
        evaluate_embedding(reference, embedding)


@pytest.mark.parametrize("n", [20, 31, 200])
def test_local_metrics_are_produced_above_the_floor(n):
    reference, embedding = _pair(n)
    values = evaluate_embedding(reference, embedding, k=neighbourhood_size(n))["values"]
    assert all(values[name] is not None for name in LOCAL)


@pytest.mark.parametrize("n", [2, 3, 4, 19])
def test_local_metrics_are_absent_below_the_floor(n):
    reference, embedding = _pair(n)
    values = evaluate_embedding(reference, embedding, k=neighbourhood_size(n))["values"]
    assert all(values[name] is None for name in LOCAL)


def test_a_small_dataset_scores_instead_of_raising():
    """The reproduction: k=15 against 20 rows used to raise ValueError."""
    reference, embedding = _pair(20)
    result = evaluate_embedding(reference, embedding, k=neighbourhood_size(20))
    assert result["values"]["trustworthiness"] is not None


def test_the_k_the_caller_gave_is_the_k_that_is_recorded():
    """`rank` and the report both read this back, so it has to be the k actually used."""
    reference, embedding = _pair(200)
    result = evaluate_embedding(reference, embedding, k=7)
    assert result["settings"]["k"] == 7


def test_silhouette_is_absent_when_every_row_is_its_own_class():
    reference, embedding = _pair(4)
    values = evaluate_embedding(
        reference, embedding, np.arange(4), k=neighbourhood_size(4)
    )["values"]
    assert values["silhouette"] is None


def test_silhouette_is_present_when_classes_are_valid():
    reference, embedding = _pair(60)
    values = evaluate_embedding(
        reference, embedding, np.arange(60) % 3, k=neighbourhood_size(60)
    )["values"]
    assert values["silhouette"] is not None


def test_label_agreement_is_absent_when_k_exceeds_the_rows_available():
    """The second label guard, separate from the silhouette one and from the floor.

    kNN agreement needs k + 1 rows, so that each point has k neighbours besides itself;
    silhouette needs 1 < n_classes < n_used and does not care about k at all. Here the
    first is unsatisfiable and the second is satisfied, which is what keeps them from
    being collapsed into one check.
    """
    reference, embedding = _pair(19)
    values = evaluate_embedding(
        reference, embedding, np.arange(19) % 3, k=19
    )["values"]
    assert values["knn_label_preservation"] is None
    assert values["silhouette"] is not None
