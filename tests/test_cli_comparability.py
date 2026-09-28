"""The comparability contract: one neighbourhood and one set of scored rows per run.

The reproduction is two candidates, one of which subsamples. Before day 7 each was
scored at a k derived from its own post-subsample row count -- k=13 over 28 rows against
k=15 over 400 -- and `rank` put them in one table with no caveat. k is now fixed once,
when the reference is. Since day 14 a candidate that subsamples is fitted on the rows it
kept and has the rest projected, so it covers every row, and the battery scores it on the
same rows as every other candidate (section 3.12).

Since day 12 a subsample registers only in front of a method whose limit the data
exceeds, so the reproduction subsamples Isomap on a dataset above its 5,000 rows.
"""

import json

import numpy as np
import pandas as pd
import pytest
from plans import complete, reconnoitre

PCA = [{"op": "pca", "params": {"n_components": 2}}]
SUBSAMPLED = [
    {"op": "subsample", "params": {"n_samples": 30}},
    {"op": "isomap", "params": {"n_components": 2}},
]
ROWS = 6000


@pytest.fixture
def wide_csv(tmp_path):
    """Enough rows that the rule saturates at k=15 and Isomap's limit is exceeded."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(ROWS, 8)), columns=[f"f{i}" for i in range(8)]
    )
    path = tmp_path / "wide.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def run_with_two_candidates(cli, wide_csv, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", wide_csv, "--runs-root", runs, "--run-id", "r1")
    reconnoitre(cli, runs / "r1")
    plan = complete({
        "dataset": "d",
        "candidates": [
            {"id": "full", "stages": PCA},
            {"id": "small", "stages": SUBSAMPLED},
        ],
        "evaluation": {
            "weights": {"trustworthiness": 1.0},
            "justification": "declared up front",
        },
    })
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", runs / "r1").code == 0
    return runs / "r1"


def test_a_subsampled_candidate_is_scored_on_the_same_rows_as_the_rest(
    cli, run_with_two_candidates
):
    """Section 3.12: fitted on 30 rows, it still covers 6,000, and is paired with PCA."""
    run = run_with_two_candidates
    for candidate in ("full", "small"):
        assert cli("embed", "--run-dir", run, "--id", candidate, "--in-process").code == 0

    # The record says which rows were fitted and which projected.
    record = json.loads((run / "embeddings" / "small.json").read_text(encoding="utf-8"))
    assert record["rows"]["n_rows"] == ROWS
    assert record["rows"]["n_fitted"] == 30
    assert record["rows"]["n_projected"] == ROWS - 30
    assert np.load(run / "embeddings" / "small.fitted.npy").size == 30
    assert np.load(run / "embeddings" / "small.npy").shape[0] == ROWS
    assert not (run / "embeddings" / "full.fitted.npy").exists()

    cli("prepare-reference", "--run-dir", run)
    full = cli("evaluate", "--run-dir", run, "--id", "full")
    small = cli("evaluate", "--run-dir", run, "--id", "small")

    # Both at the reference's k, on the same rows drawn from the same row count.
    assert full.code == 0 and small.code == 0
    assert full.payload["settings"] == small.payload["settings"]
    assert full.payload["settings"]["k"] == 15
    assert full.payload["n_total"] == small.payload["n_total"] == ROWS
    assert full.payload["scored_rows"] == small.payload["scored_rows"]

    ranking = cli("rank", "--run-dir", run)
    assert ranking.code == 0
    assert sorted(row["id"] for row in ranking.payload["ranking"]) == ["full", "small"]


def test_rank_refuses_candidates_scored_on_different_rows(
    cli, run_with_two_candidates
):
    """A metrics record from before the rule, or edited, cannot rank beside the rest."""
    run = run_with_two_candidates
    for candidate in ("full", "small"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
    cli("prepare-reference", "--run-dir", run)
    for candidate in ("full", "small"):
        cli("evaluate", "--run-dir", run, "--id", candidate)

    path = run / "metrics" / "small.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["scored_rows"] = "0" * 16
    path.write_text(json.dumps(record), encoding="utf-8")

    ranking = cli("rank", "--run-dir", run)
    assert ranking.code == 2
    assert "not all scored on the same rows" in ranking.stderr
    assert "evaluate" in ranking.stderr


def test_an_embedding_short_of_rows_is_refused_at_evaluation(
    cli, run_with_two_candidates
):
    run = run_with_two_candidates
    cli("embed", "--run-dir", run, "--id", "small", "--in-process")
    cli("prepare-reference", "--run-dir", run)
    path = run / "embeddings" / "small.npy"
    np.save(path, np.load(path)[:30])

    result = cli("evaluate", "--run-dir", run, "--id", "small")
    assert result.code == 2
    assert "has 30 rows against the reference's 6000" in result.stderr
    assert not (run / "metrics" / "small.json").exists()


def test_evaluate_scores_at_the_k_prepare_reference_recorded(
    cli, run_with_two_candidates
):
    """Not at one derived from the candidate: the recorded k is the one that binds."""
    run = run_with_two_candidates
    cli("prepare-reference", "--run-dir", run)
    recorded = json.loads(
        (run / "data" / "reference.json").read_text(encoding="utf-8")
    )["settings"]["k"]

    cli("embed", "--run-dir", run, "--id", "full", "--in-process")
    result = cli("evaluate", "--run-dir", run, "--id", "full")

    assert result.code == 0
    assert result.payload["settings"]["k"] == recorded
