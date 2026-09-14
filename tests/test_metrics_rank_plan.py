"""Evaluation, ranking, and the plan gate.

The ranking tests are mostly about what the mechanism refuses to let the agent do:
weight a metric it invented, redistribute weight without saying so, or win by being the
best of a uniformly bad field. The plan tests are about the simulation — the validator
walks the stage list tracking what each method will actually receive, which is what
lets it tell "Isomap on 100,000 points" from "subsample to 3,000, then Isomap".
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from drtools.loaders import load
from drtools.metrics import METRIC_SPECS, evaluate_embedding
from drtools.plan import Plan, validate_plan
from drtools.profile import profile_dataset
from drtools.rank import RankingError, rank_candidates

BASE_WEIGHTS = {"trustworthiness": 0.5, "continuity": 0.3, "shepard_correlation": 0.2}

#: The neighbourhood these scenarios are measured at. `evaluate_embedding` has no
#: default for it and never derives one: the caller says which k the cohort is being
#: compared at. 15 is what the published rule gives for every fixture here (n=300 and
#: n=200 both saturate the cap), so this is the number the CLI would pass too.
K = 15


def make_plan(**overrides):
    plan = {
        "dataset": "blobs",
        "candidates": [
            {"id": "pca2", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
        ],
        "evaluation": {"weights": dict(BASE_WEIGHTS), "justification": "a reason"},
    }
    plan.update(overrides)
    return plan


@pytest.fixture(scope="module")
def blobs():
    X, labels, meta = load("blobs", n_samples=300, n_features=10, n_clusters=4)
    return X, labels, profile_dataset(X, labels, meta)


# ---------------------------------------------------------------------- metrics


def test_a_faithful_embedding_scores_well_on_both_local_metrics(blobs) -> None:
    from drtools.pipeline import run_pipeline

    X, labels, _ = blobs
    embedding = run_pipeline(X, labels, [{"op": "pca", "params": {"n_components": 5}}]).embedding

    metrics = evaluate_embedding(X, embedding, labels, k=K)

    assert metrics["values"]["trustworthiness"] > 0.9
    assert metrics["values"]["continuity"] > 0.9
    assert metrics["values"]["shepard_correlation"] > 0.8


def test_a_random_embedding_scores_badly(blobs) -> None:
    """The battery has to be able to say no, or a good score means nothing."""
    X, labels, _ = blobs
    noise = np.random.default_rng(0).standard_normal((X.shape[0], 2))

    metrics = evaluate_embedding(X, noise, labels, k=K)

    assert metrics["values"]["trustworthiness"] < 0.7
    assert abs(metrics["values"]["shepard_correlation"]) < 0.2


def test_misaligned_rows_are_refused_rather_than_silently_compared(blobs) -> None:
    X, labels, _ = blobs

    with pytest.raises(ValueError, match="row-aligned"):
        evaluate_embedding(X, np.zeros((X.shape[0] - 5, 2)), labels, k=K)


def test_label_metrics_report_the_reference_value_as_context(blobs) -> None:
    """0.58 means nothing until you know the data itself only reaches 0.62."""
    from drtools.pipeline import run_pipeline

    X, labels, _ = blobs
    embedding = run_pipeline(X, labels, [{"op": "pca", "params": {"n_components": 2}}]).embedding

    metrics = evaluate_embedding(X, embedding, labels, k=K)

    assert metrics["reference_values"]["knn_label_preservation"] is not None
    assert metrics["reference_values"]["silhouette"] is not None


def test_unlabelled_data_leaves_the_supervised_metrics_absent_not_zero(blobs) -> None:
    X, _, _ = blobs
    embedding = np.asarray(X)[:, :2]

    metrics = evaluate_embedding(X, embedding, None, k=K)

    assert metrics["values"]["knn_label_preservation"] is None
    assert metrics["values"]["silhouette"] is None


def test_metrics_work_on_a_sparse_reference_without_densifying_it() -> None:
    rng = np.random.default_rng(0)
    dense = (rng.random((200, 60)) < 0.2) * rng.random((200, 60))
    X = sp.csr_array(dense.astype(np.float64))

    metrics = evaluate_embedding(X, np.asarray(dense[:, :2]), None, k=K)

    assert 0.0 <= metrics["values"]["trustworthiness"] <= 1.0


# ---------------------------------------------------------------------- ranking


def scored(**values):
    return {"values": {**{k: None for k in METRIC_SPECS}, **values}}


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(RankingError, match="sum to"):
        rank_candidates({"a": scored(trustworthiness=0.9)}, {"trustworthiness": 0.5})


def test_an_invented_metric_is_rejected() -> None:
    with pytest.raises(RankingError, match="unknown metric"):
        rank_candidates({"a": scored(trustworthiness=0.9)}, {"elegance": 1.0})


def test_a_negative_weight_is_rejected() -> None:
    """Direction is already declared per metric; a negative weight would invert it."""
    with pytest.raises(RankingError, match="negative weight"):
        rank_candidates(
            {"a": scored(trustworthiness=0.9)},
            {"trustworthiness": 1.5, "continuity": -0.5},
        )


def test_scores_are_absolute_so_a_bad_field_does_not_produce_a_good_winner() -> None:
    """Min-max normalisation would award the best of these 1.0. It should not."""
    result = rank_candidates(
        {
            "bad": scored(trustworthiness=0.30, continuity=0.30, shepard_correlation=0.10),
            "worse": scored(trustworthiness=0.20, continuity=0.20, shepard_correlation=0.05),
        },
        dict(BASE_WEIGHTS),
    )

    assert result["winner"] == "bad"
    assert result["ranking"][0]["score"] < 0.4


def test_a_metric_missing_for_every_candidate_is_redistributed_and_declared() -> None:
    result = rank_candidates(
        {
            "a": scored(trustworthiness=0.9, continuity=0.8),
            "b": scored(trustworthiness=0.7, continuity=0.6),
        },
        {"trustworthiness": 0.4, "continuity": 0.3, "knn_label_preservation": 0.3},
    )

    assert "knn_label_preservation" in result["weights_dropped"]
    assert abs(sum(result["weights_applied"].values()) - 1.0) < 1e-9
    assert any("redistributed" in note for note in result["notes"])


def test_a_metric_missing_for_only_some_candidates_is_dropped_for_all() -> None:
    """Scoring it where convenient would make the comparison unfair."""
    result = rank_candidates(
        {
            "a": scored(trustworthiness=0.9, silhouette=0.8),
            "b": scored(trustworthiness=0.7),
        },
        {"trustworthiness": 0.6, "silhouette": 0.4},
    )

    assert "silhouette" in result["weights_dropped"]


def test_a_near_tie_is_reported_as_a_tie() -> None:
    result = rank_candidates(
        {
            "a": scored(trustworthiness=0.800, continuity=0.800, shepard_correlation=0.80),
            "b": scored(trustworthiness=0.799, continuity=0.799, shepard_correlation=0.80),
        },
        dict(BASE_WEIGHTS),
    )

    assert any("treated as tied" in note for note in result["notes"])


def test_failed_candidates_are_excluded_rather_than_scored_as_zero() -> None:
    """Scoring a crash as zero would rank the method, not the configuration that broke."""
    result = rank_candidates(
        {"a": scored(trustworthiness=0.9, continuity=0.8, shepard_correlation=0.7)},
        dict(BASE_WEIGHTS),
        failures={"isomap": {"op": "isomap", "message": "disconnected"}},
    )

    assert result["failed_candidates"] == ["isomap"]
    assert any("not a judgement on the method" in note for note in result["notes"])


def test_ranking_nothing_is_an_error_not_an_empty_result() -> None:
    with pytest.raises(RankingError, match="every one failed"):
        rank_candidates({}, dict(BASE_WEIGHTS))


# ------------------------------------------------------------------------- plan


def test_a_coherent_plan_is_accepted(blobs) -> None:
    _, _, profile = blobs

    report = validate_plan(make_plan(), profile)

    assert report["valid"] is True
    assert report["summary"] == "accepted."


def test_a_plan_without_a_linear_baseline_is_rejected(blobs) -> None:
    """Without PCA there is nothing to measure the nonlinear methods against."""
    _, _, profile = blobs

    report = validate_plan(
        make_plan(
            candidates=[{"id": "u", "stages": [{"op": "umap"}]}]
        ),
        profile,
    )

    codes = {f["code"] for f in report["findings"]}
    assert report["valid"] is False
    assert "no_linear_baseline" in codes


def test_a_method_beyond_its_scale_limit_is_rejected_with_the_limit_named() -> None:
    profile = {
        "shape": {"n_samples": 107_000, "n_features": 2352, "storage": "dense"},
        "values": {"suspected_kind": "bounded_unit_interval"},
    }

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "iso", "stages": [{"op": "isomap"}]},
            ]
        ),
        profile,
    )

    finding = next(f for f in report["findings"] if f["code"] == "exceeds_scale_limit")
    assert finding["candidate"] == "iso"
    assert "107,000" in finding["message"]
    assert "subsample" in finding["fix"]


def test_subsampling_first_makes_the_same_method_acceptable() -> None:
    """The point of simulating the plan rather than pattern-matching on the dataset.

    A rule keyed on the dataset's size would reject both of these. Walking the stages
    while tracking the sample count distinguishes the hopeless one from the correct way
    to do it.
    """
    profile = {
        "shape": {"n_samples": 107_000, "n_features": 2352, "storage": "dense"},
        "values": {"suspected_kind": "bounded_unit_interval"},
    }

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {
                    "id": "iso",
                    "stages": [
                        {"op": "subsample", "params": {"n_samples": 3000}},
                        {"op": "isomap"},
                    ],
                },
            ]
        ),
        profile,
    )

    assert report["valid"] is True


def test_raw_counts_into_a_euclidean_method_are_rejected() -> None:
    profile = {
        "shape": {"n_samples": 2700, "n_features": 32000, "storage": "sparse_csr"},
        "values": {"suspected_kind": "counts"},
    }

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "t", "stages": [{"op": "densify"}, {"op": "tsne", "params": {"perplexity": 30}}]},
            ]
        ),
        profile,
    )

    finding = next(
        f for f in report["findings"] if f["code"] == "raw_counts_into_euclidean_method"
    )
    assert "sequencing depth" in finding["message"]
    assert "log1p" in finding["fix"]


def test_normalising_first_clears_the_raw_counts_objection() -> None:
    profile = {
        "shape": {"n_samples": 2700, "n_features": 32000, "storage": "sparse_csr"},
        "values": {"suspected_kind": "counts"},
    }

    report = validate_plan(
        make_plan(
            base_preprocessing=[{"op": "normalise_total"}, {"op": "log1p"}],
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "t", "stages": [{"op": "densify"}, {"op": "tsne", "params": {"perplexity": 30}}]},
            ],
        ),
        profile,
    )

    codes = {f["code"] for f in report["findings"]}
    assert "raw_counts_into_euclidean_method" not in codes


def test_a_dense_only_method_on_sparse_data_is_rejected_with_the_memory_cost() -> None:
    profile = {
        "shape": {"n_samples": 2700, "n_features": 32000, "storage": "sparse_csr"},
        "values": {"suspected_kind": "continuous"},
    }

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "m", "stages": [{"op": "mds"}]},
            ]
        ),
        profile,
    )

    finding = next(
        f for f in report["findings"] if f["code"] == "sparse_into_dense_method"
    )
    assert "GB" in finding["fix"]


def test_a_perplexity_too_large_for_the_sample_count_is_rejected() -> None:
    profile = {
        "shape": {"n_samples": 60, "n_features": 5, "storage": "dense"},
        "values": {"suspected_kind": "continuous"},
    }

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "t", "stages": [{"op": "tsne", "params": {"perplexity": 30}}]},
            ]
        ),
        profile,
    )

    assert any(f["code"] == "perplexity_too_large" for f in report["findings"])


def test_a_rejection_without_evidence_is_flagged_but_does_not_block(blobs) -> None:
    _, _, profile = blobs

    report = validate_plan(
        make_plan(rejected=[{"method": "isomap", "reason": "too slow"}]), profile
    )

    assert report["valid"] is True
    assert any(f["code"] == "unevidenced_rejection" for f in report["findings"])


def test_an_unjustified_weighting_is_flagged(blobs) -> None:
    """Pre-registering weights constrains nothing if the reasoning is not recorded."""
    _, _, profile = blobs

    report = validate_plan(
        make_plan(evaluation={"weights": dict(BASE_WEIGHTS), "justification": "  "}),
        profile,
    )

    assert any(f["code"] == "unjustified_weighting" for f in report["findings"])


def test_duplicate_candidate_ids_are_rejected(blobs) -> None:
    _, _, profile = blobs

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "pca2", "stages": [{"op": "umap"}]},
            ]
        ),
        profile,
    )

    assert any(f["code"] == "duplicate_candidate_id" for f in report["findings"])


def test_a_disconnected_graph_found_in_recon_warns_before_the_method_runs(blobs) -> None:
    _, _, profile = blobs
    recon = {"neighbourhood": {"k": 15, "n_connected_components": 5}}

    report = validate_plan(
        make_plan(
            candidates=[
                {"id": "pca2", "stages": [{"op": "pca"}]},
                {"id": "iso", "stages": [{"op": "isomap", "params": {"n_neighbors": 10}}]},
            ]
        ),
        profile,
        recon,
    )

    finding = next(
        f for f in report["findings"] if f["code"] == "likely_disconnected_graph"
    )
    assert finding["severity"] == "warning"
    assert "raise n_neighbors" in finding["fix"]


def test_the_plan_schema_rejects_unknown_fields() -> None:
    """A typo'd key that was silently ignored would leave the plan lying about itself."""
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        Plan.model_validate(make_plan(unexpected_field="surprise"))
