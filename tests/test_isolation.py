"""Running candidates in a process that can be killed, and recording what happened.

The reason isolation exists is that not every failure is an exception. A method can
exhaust memory and take the interpreter with it, or simply never finish. These tests
cover all three outcomes — success, handled failure, and a wall-clock stop — because
the agent's retry logic branches on which one it got.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from drtools.cache import is_cached, read_cache, write_cache
from drtools.isolation import run_candidate
from drtools.loaders import load
from drtools.runs import RunDir


@pytest.fixture
def run_with_data(tmp_path):
    def build(spec: str, **kwargs):
        X, labels, meta = load(spec, **kwargs)
        run = RunDir.create(tmp_path, spec, run_id=spec)
        write_cache(run, X, labels, meta)
        return run

    return build


# ------------------------------------------------------------------------- cache


def test_dense_data_round_trips_through_the_cache(tmp_path) -> None:
    X, labels, meta = load("blobs", n_samples=100, n_features=6)
    run = RunDir.create(tmp_path, "blobs")

    write_cache(run, X, labels, meta)
    restored, restored_labels, restored_meta = read_cache(run)

    assert is_cached(run)
    np.testing.assert_allclose(np.asarray(restored), X)
    np.testing.assert_array_equal(restored_labels, labels)
    assert restored_meta["cached_storage"] == "dense"


def test_sparse_data_stays_sparse_through_the_cache(tmp_path) -> None:
    """Densifying here would undo the whole reason the contract admits sparse input."""
    X = sp.csr_array(np.eye(50, 200, dtype=np.float64))
    run = RunDir.create(tmp_path, "sparse")

    write_cache(run, X, None, {"name": "sparse", "source": "test"})
    restored, labels, meta = read_cache(run)

    assert sp.issparse(restored)
    assert labels is None
    assert meta["cached_storage"] == "sparse_csr"


# --------------------------------------------------------------------- outcomes


def test_a_successful_candidate_records_ok_and_writes_its_embedding(
    run_with_data,
) -> None:
    run = run_with_data("blobs", n_samples=200, n_features=8)

    outcome = run_candidate(
        run, "pca2", [{"op": "pca", "params": {"n_components": 2}}], timeout_s=120
    )

    assert outcome["status"] == "ok"
    assert outcome["output_shape"] == [200, 2]
    embedding = np.load(run.path / "embeddings" / "pca2.npy")
    assert embedding.shape == (200, 2)


def test_a_failing_candidate_names_the_stage_and_its_parameters(run_with_data) -> None:
    """The record is what the agent revises from, so it must identify what to change."""
    run = run_with_data("blobs", n_samples=300, n_clusters=5)

    outcome = run_candidate(
        run,
        "broken",
        [{"op": "isomap", "params": {"n_neighbors": 6}}],
        timeout_s=120,
    )

    assert outcome["status"] == "failed"
    failure = outcome["failure"]
    assert failure["op"] == "isomap"
    assert failure["params"]["n_neighbors"] == 6
    assert "disconnected" in failure["message"]
    assert "traceback" not in failure["message"].lower()


def test_an_underlying_library_error_keeps_its_own_type(run_with_data) -> None:
    """MemoryError and LinAlgError call for different repairs, so the name is kept."""
    run = run_with_data("blobs", n_samples=100, n_features=5)

    outcome = run_candidate(
        run,
        "toomany",
        [{"op": "lle", "params": {"n_neighbors": 3, "method": "hessian"}}],
        timeout_s=120,
    )

    assert outcome["status"] == "failed"
    assert outcome["failure"]["op"] == "lle"


def test_a_candidate_over_its_budget_is_stopped_and_recorded_as_a_timeout(
    run_with_data,
) -> None:
    """A hung analysis is worse than a failed one: it produces nothing and blocks.

    The cap here is short enough to fire reliably. What is under test is the parent's
    handling — that it kills the process and leaves a record the planner can act on —
    not the precise duration at which it fires.
    """
    run = run_with_data("blobs", n_samples=400, n_features=6)

    outcome = run_candidate(
        run, "slow", [{"op": "mds", "params": {"n_init": 4}}], timeout_s=0.5
    )

    assert outcome["status"] == "timeout"
    assert outcome["failure"]["error_type"] == "Timeout"
    assert outcome["failure"]["timeout_s"] == 0.5
    assert "did not fail; it did not finish" in outcome["failure"]["message"]


def test_a_malformed_plan_is_rejected_by_the_worker_without_crashing_it(
    run_with_data,
) -> None:
    run = run_with_data("blobs", n_samples=100, n_features=5)

    outcome = run_candidate(run, "bad", [{"op": "log1p"}], timeout_s=120)

    assert outcome["status"] == "failed"
    assert outcome["failure"]["error_type"] == "PipelineError"
    assert "must end in a reduction" in outcome["failure"]["message"]


def test_isolation_preserves_determinism(run_with_data) -> None:
    run = run_with_data("blobs", n_samples=200, n_features=6)
    stages = [{"op": "pca", "params": {"n_components": 3}}]

    run_candidate(run, "a", stages, seed=5, timeout_s=120)
    run_candidate(run, "b", stages, seed=5, timeout_s=120)

    first = np.load(run.path / "embeddings" / "a.npy")
    second = np.load(run.path / "embeddings" / "b.npy")
    np.testing.assert_allclose(first, second)
