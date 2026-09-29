"""Diffusion Maps' connectivity floor and the refusal behind it (day 20).

On PBMC3k the bandwidth rule chose a kernel narrower than the typical distance to a
nearest neighbour, so almost every cell was isolated: each took a coordinate of its own,
every leading eigenvalue was 1, and the candidate won at d = 74 of 83. A kernel that
isolates samples is now refused, and a bandwidth left to the rule is doubled until it
does not.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import pairwise_distances

from drtools.executors import ExecutionError
from drtools.executors.spectral import (
    _bandwidth_by_kernel_scaling,
    _connectivity_floor,
    _diffusion_spectrum,
    _isolates,
    _support,
)
from drtools.pipeline import run_pipeline
from drtools.tuning import d_curve


@pytest.fixture(scope="module")
def roll():
    return make_swiss_roll(n_samples=600, noise=0.05, random_state=0)


def test_a_kernel_that_isolates_samples_is_refused(roll):
    X, _ = roll

    with pytest.raises(ExecutionError, match="isolates individual samples"):
        run_pipeline(X, None, [{"op": "diffusion_maps", "params": {"epsilon": 1e-4}}])


def test_the_floor_widens_a_bandwidth_that_isolates_samples(roll):
    X, _ = roll
    distances = pairwise_distances(X, squared=True)
    rule, _ = _bandwidth_by_kernel_scaling(distances, 0)

    width, doublings = _connectivity_floor(distances, rule / 256, 1.0)

    assert doublings > 0
    assert not _isolates(*_values_and_support(distances, width))
    assert _isolates(*_values_and_support(distances, width / 2))
    # The fit agrees with the search: the same solver reads the same kernel.
    result = run_pipeline(X, None, [{"op": "diffusion_maps", "params": {"epsilon": width}}])
    assert result.stages[-1].notes["localised_leading_coordinates"] <= 5


def _values_and_support(distances, width):
    values, vectors, _, _ = _diffusion_spectrum(distances, width, 1.0)
    return values, _support(vectors)


def test_a_kernel_split_into_pieces_is_widened_until_it_joins():
    """Found on day 21: three far outliers each owned an eigenvalue of exactly 1 and
    both coordinates, yet only 5 of 20 leading coordinates read as localised."""
    blob = np.random.default_rng(0).normal(size=(300, 3))
    X = np.vstack([blob, np.eye(3) * 20])

    result = run_pipeline(X, None, [{"op": "diffusion_maps", "params": {}}])
    notes = result.stages[-1].notes
    outliers_share = np.abs(result.embedding[-3:]).sum(0) / np.abs(result.embedding).sum(0)

    assert notes["eigenvalues"][0] - notes["eigenvalues"][1] > 1e-10
    assert outliers_share.max() < 0.5
    with pytest.raises(ExecutionError, match="falls apart into separate pieces"):
        run_pipeline(X, None, [{"op": "diffusion_maps", "params": {"epsilon": 0.36}}])


def test_the_floor_leaves_a_resolved_manifold_alone(roll):
    X, t = roll

    result = run_pipeline(X, None, [{"op": "diffusion_maps", "params": {}}])
    notes = result.stages[-1].notes

    assert notes["connectivity_floor_doublings"] == 0
    assert max(abs(spearmanr(result.embedding[:, j], t).statistic) for j in range(2)) > 0.95


def test_the_d_curve_reads_an_eigengap_record():
    """Found on day 20: the eigengap criterion records `gaps`, and `figures` crashed."""
    record = {
        "method": {"criterion": "eigengap"},
        "chosen": {"d": 4, "multiplier": 1.0, "t": None},
        "criterion_choices": [
            {"multiplier": 1.0, "t": None, "d": 4, "rule": "eigengap",
             "gaps": {"2": 0.04, "3": 0.008, "4": 0.1}},
        ],
    }

    curve = d_curve(record)

    assert curve["points"] == {2: 0.04, 3: 0.008, 4: 0.1}
    assert curve["label"] == "eigengap after d"
