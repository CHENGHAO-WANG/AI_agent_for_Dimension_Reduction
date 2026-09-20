"""The loader contract and the run directory.

The contract is the boundary that makes the agent's one code-writing privilege safe,
so its rejections are tested as carefully as its acceptances: an adapter that returns
misaligned labels must be stopped here, not three stages later.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from drtools.contract import ContractError, check_dataset
from drtools.runs import MISSING, RunDir, resolve_evidence

VALID_META = {"name": "example", "source": "test"}


def test_accepts_a_well_formed_dense_dataset() -> None:
    X = np.zeros((10, 3), dtype=np.float64)
    check_dataset(X, np.arange(10), dict(VALID_META))


def test_accepts_sparse_input() -> None:
    """Sparse input is admitted on purpose; see the module docstring for why."""
    X = sp.csr_array(np.eye(10, 4, dtype=np.float32))
    check_dataset(X, None, dict(VALID_META))


def test_rejects_integer_dtype_with_a_repairable_message() -> None:
    X = np.zeros((10, 3), dtype=np.int64)
    with pytest.raises(ContractError, match="float32 or float64"):
        check_dataset(X, None, dict(VALID_META))


def test_rejects_non_finite_values() -> None:
    X = np.zeros((10, 3))
    X[2, 1] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        check_dataset(X, None, dict(VALID_META))


def test_rejects_labels_misaligned_with_x() -> None:
    with pytest.raises(ContractError, match="misaligned"):
        check_dataset(np.zeros((10, 3)), np.arange(9), dict(VALID_META))


def test_rejects_string_labels_and_says_where_names_belong() -> None:
    labels = np.array(["a"] * 10)
    with pytest.raises(ContractError, match="label_names"):
        check_dataset(np.zeros((10, 3)), labels, dict(VALID_META))


def test_rejects_meta_missing_required_keys() -> None:
    with pytest.raises(ContractError, match="missing required key"):
        check_dataset(np.zeros((10, 3)), None, {"name": "example"})


def test_rejects_a_one_dimensional_x() -> None:
    with pytest.raises(ContractError, match="2-D"):
        check_dataset(np.zeros(10), None, dict(VALID_META))


# ------------------------------------------------------------------ run directory


def test_run_directory_round_trips_artifacts_and_decisions(tmp_path) -> None:
    run = RunDir.create(tmp_path, "swiss_roll", run_id="fixed")

    run.write_artifact("profile.json", {"shape": {"n_samples": 42}})
    run.log_decision(
        stage="plan",
        question="Which methods?",
        chosen="pca, umap",
        rationale="spectrum decays fast",
        evidence=["profile.shape.n_samples"],
    )

    assert run.id == "fixed"
    assert run.read_artifact("profile.json")["shape"]["n_samples"] == 42

    logged = run.decisions()
    assert len(logged) == 1
    assert logged[0]["chosen"] == "pca, umap"
    assert logged[0]["actor"] == "agent"


def test_manifest_records_what_reproduction_needs(tmp_path) -> None:
    run = RunDir.create(tmp_path, "example")
    run.write_manifest(dataset="example", seed=7)

    manifest = run.read_artifact("run.json")
    assert manifest["seed"] == 7
    assert manifest["python"].startswith("3.")
    assert "numpy" in manifest["package_versions"]


def test_evidence_resolution_distinguishes_present_from_missing(tmp_path) -> None:
    """A citation that does not resolve must surface as None, not vanish."""
    artifacts = {"profile": {"shape": {"n_samples": 42}, "observations": [{"a": 1}]}}

    resolved = resolve_evidence(
        ["profile.shape.n_samples", "profile.observations.0.a", "profile.nope.deep"],
        artifacts,
    )

    assert resolved["profile.shape.n_samples"] == 42
    assert resolved["profile.observations.0.a"] == 1
    assert resolved["profile.nope.deep"] is MISSING


def test_the_non_finite_refusal_does_not_ask_the_loader_to_resolve_it() -> None:
    """The contract used to instruct the very fabrication it exists to prevent.

    "Missing data must be resolved or explicitly encoded by the loader" is an
    instruction, not a note: an agent-written loader is the one place that could act
    on it, and `nan_to_num` satisfies the contract with nothing recorded anywhere.
    """
    X = np.zeros((10, 3))
    X[2, 1] = np.nan

    with pytest.raises(ContractError) as error:
        check_dataset(X, None, dict(VALID_META))

    message = str(error.value)
    assert "out of scope" in message
    assert "resolved or explicitly encoded by the loader" not in message
