"""The comparability contract: one neighbourhood per run, or no ranking.

The reproduction is two candidates that subsample differently. Before this, each was
scored at a k derived from its own post-subsample row count — k=13 over 28 rows against
k=15 over 400 — and `rank` put them in one table with no caveat, the subsampled one
winning on a trustworthiness measured over a few dozen points. k is now fixed once, when
the reference is, and a candidate whose rows cannot carry it is refused rather than
quietly rescored at a smaller one.
"""

import json

import numpy as np
import pandas as pd
import pytest

PCA = [{"op": "pca", "params": {"n_components": 2}}]
SUBSAMPLED = [
    {"op": "subsample", "params": {"n_samples": 30}},
    {"op": "pca", "params": {"n_components": 2}},
]


@pytest.fixture
def wide_csv(tmp_path):
    """400 rows, enough that the rule saturates at k=15 and a subsample cannot carry it."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(400, 8)), columns=[f"f{i}" for i in range(8)]
    )
    path = tmp_path / "wide.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def run_with_two_candidates(cli, wide_csv, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", wide_csv, "--runs-root", runs, "--run-id", "r1")
    plan = {
        "dataset": "d",
        "candidates": [
            {"id": "full", "stages": PCA},
            {"id": "small", "stages": SUBSAMPLED},
        ],
        "evaluation": {
            "weights": {"trustworthiness": 1.0},
            "justification": "declared up front",
        },
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", runs / "r1").code == 0
    return runs / "r1"


def test_candidates_that_subsample_differently_cannot_be_ranked_together(
    cli, run_with_two_candidates
):
    run = run_with_two_candidates
    for candidate in ("full", "small"):
        assert cli("embed", "--run-dir", run, "--id", candidate, "--in-process").code == 0

    full = cli("evaluate", "--run-dir", run, "--id", "full")
    small = cli("evaluate", "--run-dir", run, "--id", "small")

    # The one that kept every row is scored at the reference's k, and says so.
    assert full.code == 0
    assert full.payload["settings"]["k"] == 15

    # The one that subsampled is refused, and the refusal names the subsample and the
    # way out rather than leaving the agent to infer either.
    assert small.code == 2
    assert "subsampled to 30 of the reference's 400 rows" in small.stderr
    assert "k=15" in small.stderr
    assert "new run" in small.stderr

    # Nothing was written for it, so `rank` has nothing to put beside the other.
    assert not (run / "metrics" / "small.json").exists()
    ranking = cli("rank", "--run-dir", run)
    assert ranking.code == 0
    assert [row["id"] for row in ranking.payload["ranking"]] == ["full"]


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
