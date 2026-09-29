"""PHATE, PaCMAP and TriMap, added on day 18 (section 7).

Each runs at d = 2, is reproducible under the run's seed, keeps well-separated clusters
apart, and refuses a neighbourhood the samples cannot supply. Each is offered the shared
size-aware neighbourhood suggestion, which is what tuning's multipliers scale.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from plans import complete, reconnoitre
from sklearn.neighbors import KNeighborsClassifier

from drtools.executors import ExecutionError
from drtools.heuristics import suggest
from drtools.loaders import load
from drtools.pipeline import run_pipeline

METHODS = ["phate", "pacmap", "trimap"]


@pytest.fixture(scope="module")
def blobs():
    X, labels, _ = load("blobs", n_samples=300, n_features=10)
    return X, labels


@pytest.mark.parametrize("op", METHODS)
def test_each_keeps_clusters_apart_and_is_reproducible(blobs, op) -> None:
    X, labels = blobs
    stages = [{"op": op, "params": {}}]

    first = run_pipeline(X, labels, stages, seed=3)
    again = run_pipeline(X, labels, stages, seed=3)

    assert first.embedding.shape == (300, 2)
    assert np.array_equal(first.embedding, again.embedding)
    agreement = KNeighborsClassifier(5).fit(first.embedding, labels).score(first.embedding, labels)
    assert agreement > 0.95
    assert first.stages[0].notes["metric"] == "euclidean"


@pytest.mark.parametrize("op, allowed", [("phate", 20), ("pacmap", 20), ("trimap", 19)])
def test_each_refuses_a_neighbourhood_the_samples_cannot_supply(op, allowed) -> None:
    X, labels, _ = load("blobs", n_samples=20, n_features=4)

    with pytest.raises(ExecutionError, match=f"not below the {allowed}"):
        run_pipeline(X, labels, [{"op": op, "params": {"n_neighbors": allowed}}])


@pytest.mark.parametrize("op, default", [("phate", 5), ("pacmap", 10), ("trimap", 12)])
def test_the_neighbourhood_suggestion_scales_with_n(op, default) -> None:
    def at(n: int) -> int:
        return suggest(op, {"shape": {"n_samples": n}})["n_neighbors"]["value"]

    assert at(5_000) == default
    assert at(400) == max(5, 400 // 50)


def test_a_visualization_run_tunes_and_compares_all_three(cli, csv_dataset, tmp_path) -> None:
    """The daily smoke test for today's methods: registration, tuning at d = 2, scoring."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8), "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    reconnoitre(cli, run, purpose="visualization")
    candidates = [("pca", "pca")] + [(op, op) for op in METHODS]
    document = complete({
        "dataset": "d",
        "candidates": [{"id": cid, "stages": [{"op": op, "params": {}}]} for cid, op in candidates],
        "evaluation": {
            "weights": {"trustworthiness": 0.25, "continuity": 0.25, "shepard_correlation": 0.5},
            "justification": "declared up front",
        },
    })
    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", run).code == 0
    cli("prepare-reference", "--run-dir", run)

    for cid, _ in candidates:
        assert cli("embed", "--run-dir", run, "--id", cid, "--in-process").code == 0
        assert cli("evaluate", "--run-dir", run, "--id", cid).code == 0
        assert np.load(run / "embeddings" / f"{cid}.npy").shape == (60, 2)

    assert cli("compare", "--run-dir", run).code == 0
