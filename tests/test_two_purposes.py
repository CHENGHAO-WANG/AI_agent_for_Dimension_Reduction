"""Day 17: two purposes, a representation or a picture (section 3.11).

The checkpoint's purpose and focus, recorded once and frozen by registration; the focus
weightings; which methods each purpose makes eligible; a visualization run at d = 2,
compared metric by metric, with the agent's recommendation and the user's adoption in
place of a ranking; plots A and B of a representation; what each method's picture
means.
"""

import json

import numpy as np
import pytest
from plans import CHECKPOINT, RECON, checkpoint, complete, reconnoitre

from drtools.plan import validate_plan
from drtools.plots import draw_plots, principal_axes
from drtools.rank import default_weightings, matching_default
from drtools.registry import READING_FIELDS, load_registry

BALANCED = {"trustworthiness": 0.25, "continuity": 0.25, "shepard_correlation": 0.5}
LOCAL = {"trustworthiness": 0.35, "continuity": 0.35, "shepard_correlation": 0.3}
PCA = [{"op": "pca", "params": {}}]
ISOMAP = [{"op": "isomap", "params": {}}]
TSNE = [{"op": "tsne", "params": {"perplexity": 5}}]
PROFILE = {
    "shape": {"n_samples": 300, "n_features": 10, "storage": "dense"},
    "values": {"suspected_kind": "continuous"},
    "labels": {"present": False},
}


def _plan(candidates, weights=BALANCED, **fields):
    return {
        "dataset": "d",
        "candidates": [{"id": cid, "stages": stages} for cid, stages in candidates],
        "evaluation": {"weights": weights, "justification": "declared up front"},
        **fields,
    }


def _codes(report):
    return {f["code"] for f in report["findings"] if f["severity"] == "error"}


def _validate(document, **answers):
    return validate_plan(document, PROFILE, RECON, checkpoint=checkpoint(**answers))


# ------------------------------------------------------------ the focus weightings


def test_each_focus_moves_a_fifth_of_the_weight_between_local_and_global():
    for focus, local in (("local", 0.7), ("balanced", 0.5), ("global", 0.3)):
        weights = default_weightings(focus)["default"]
        assert weights["trustworthiness"] == weights["continuity"]
        assert weights["trustworthiness"] + weights["continuity"] == pytest.approx(local)
        assert sum(weights.values()) == pytest.approx(1.0)


def test_with_trusted_labels_seventy_per_cent_stays_in_the_focus_proportions():
    weights = default_weightings("global")["default_trusted_labels"]

    assert weights == {
        "trustworthiness": 0.105, "continuity": 0.105, "shepard_correlation": 0.49,
        "knn_label_preservation": 0.20, "silhouette": 0.10,
    }


def test_another_focus_default_is_not_this_focus_default():
    assert matching_default(LOCAL, "local") == "default"
    assert matching_default(BALANCED, "local") is None


def test_the_balanced_default_under_a_local_focus_needs_evidence():
    document = complete(_plan([("pca", PCA)]))
    document["evaluation"]["evidence"] = []

    report = _validate(document, focus="local")

    assert "uncited_weighting" in _codes(report)


def test_the_local_default_under_a_local_focus_registers():
    document = complete(_plan([("pca", PCA)], weights=LOCAL))
    document["evaluation"]["evidence"] = []

    report = _validate(document, focus="local")

    assert report["valid"] and report["weighting"] == "default"
    assert report["focus"] == "local"


# ------------------------------------------------------- what each purpose admits


def test_registration_refuses_a_plan_with_no_checkpoint():
    report = validate_plan(complete(_plan([("pca", PCA)])), PROFILE, RECON)

    assert "no_checkpoint" in _codes(report)


def test_a_representation_run_refuses_a_visualization_method():
    document = complete(_plan([("pca", PCA), ("tsne", TSNE)]), purpose="representation")

    report = _validate(document, purpose="representation")

    assert "visualization_method_in_representation_run" in _codes(report)


def test_a_representation_run_does_not_reject_a_visualization_method():
    document = complete(_plan([("pca", PCA)]))
    document["rejected"].append({"method": "umap", "reason": "not wanted",
                                 "evidence": ["profile.shape.n_samples"]})

    report = _validate(document)

    assert "excluded_by_purpose" in _codes(report)


def test_a_representation_run_accounts_for_the_reductions_alone():
    report = _validate(complete(_plan([("pca", PCA)])))

    assert report["valid"], report["findings"]


def test_a_visualization_run_accounts_for_both_classes():
    document = complete(_plan([("pca", PCA)]), purpose="representation")

    report = _validate(document, purpose="visualization")

    unaccounted = [f for f in report["findings"] if f["code"] == "method_unaccounted"]
    assert unaccounted and "tsne" in unaccounted[0]["message"]


def test_a_visualization_run_registers_a_visualization_method():
    report = _validate(complete(_plan([("pca", PCA), ("tsne", TSNE)])),
                       purpose="visualization")

    assert report["valid"], report["findings"]
    assert report["purpose"] == "visualization"


# ---------------------------------------------------------------- the checkpoint


def _profiled(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    cli("recon", "--run-dir", runs / "r1", "--decision",
        json.dumps({"values": "not_counts", "features": "one_type",
                    "decided_by": "user"}))
    return runs / "r1"


def test_the_checkpoint_is_recorded_and_logged(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)

    result = cli("checkpoint", "--run-dir", run, "--answers",
                 json.dumps(checkpoint(purpose="visualization", purpose_decided_by="user")))

    assert result.code == 0
    stored = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    assert stored["purpose"] == "visualization"
    logged = [json.loads(line) for line in
              (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged[-1]["stage"] == "checkpoint"


def test_an_answer_the_agent_gave_needs_evidence(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)

    result = cli("checkpoint", "--run-dir", run, "--answers",
                 json.dumps(checkpoint(focus="local", focus_decided_by="agent")))

    assert result.code == 2 and "evidence" in result.stderr


def test_the_checkpoint_refuses_a_key_that_does_not_resolve(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)

    result = cli("checkpoint", "--run-dir", run, "--answers", json.dumps(checkpoint(
        focus="local", focus_decided_by="agent", evidence=["profile.no.such.key"])))

    assert result.code == 2


def test_the_checkpoint_is_frozen_by_registration(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("checkpoint", "--run-dir", run, "--answers", json.dumps(CHECKPOINT))
    (run / "plan.json").write_text(json.dumps(complete(_plan([("pca", PCA)]))),
                                   encoding="utf-8")
    assert cli("validate-plan", "--run-dir", run).code == 0

    moved = cli("checkpoint", "--run-dir", run, "--answers",
                json.dumps(checkpoint(purpose="visualization")))

    assert moved.code == 2 and "already registered" in moved.stderr


def test_a_checkpoint_edited_by_hand_after_registration_is_refused(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("checkpoint", "--run-dir", run, "--answers", json.dumps(CHECKPOINT))
    (run / "plan.json").write_text(json.dumps(complete(_plan([("pca", PCA)]))),
                                   encoding="utf-8")
    cli("validate-plan", "--run-dir", run)
    (run / "checkpoint.json").write_text(json.dumps(checkpoint(purpose="visualization")),
                                         encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2 and "changed by hand" in result.stderr
    assert cli("status", "--run-dir", run).payload["purpose"] == "representation"


def test_status_asks_for_the_checkpoint_before_the_plan(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)

    assert cli("status", "--run-dir", run).payload["next"] == "checkpoint"
    cli("checkpoint", "--run-dir", run, "--answers", json.dumps(CHECKPOINT))
    assert cli("status", "--run-dir", run).payload["next"] == "plan"


# ------------------------------------------------------------ a visualization run


@pytest.fixture
def pictured(cli, csv_dataset, tmp_path):
    """A visualization run with three candidates embedded and evaluated."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    reconnoitre(cli, run, purpose="visualization")
    document = complete(_plan([("pca", PCA), ("isomap", ISOMAP), ("tsne", TSNE)]))
    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", run).code == 0
    cli("prepare-reference", "--run-dir", run)
    for candidate in ("pca", "isomap", "tsne"):
        assert cli("embed", "--run-dir", run, "--id", candidate, "--in-process").code == 0
        assert cli("evaluate", "--run-dir", run, "--id", candidate).code == 0
    return run


def test_every_candidate_in_a_visualization_run_is_drawn_at_d_2(pictured):
    for candidate in ("pca", "isomap", "tsne"):
        assert np.load(pictured / "embeddings" / f"{candidate}.npy").shape[1] == 2


def test_a_visualization_run_draws_no_plots_a_or_b(pictured):
    record = json.loads((pictured / "metrics" / "isomap.json").read_text(encoding="utf-8"))

    assert "plots" not in record and not (pictured / "plots").exists()


def test_rank_refuses_a_visualization_run(cli, pictured):
    result = cli("rank", "--run-dir", pictured)

    assert result.code == 2 and "compare" in result.stderr
    assert not (pictured / "ranking.json").exists()


def test_compare_sets_each_metric_beside_the_others_and_sums_none(cli, pictured):
    comparison = cli("compare", "--run-dir", pictured).payload

    assert comparison["candidates"][0] == "pca"
    trust = comparison["metrics"]["trustworthiness"]
    assert trust["best"] == trust["order"][0] and trust["best"] in trust["within_margin"]
    assert "score" not in json.dumps(comparison) and "winner" not in comparison


def _recommend(cli, run, **fields):
    document = {"recommended": ["tsne"], "rationale": "keeps neighbourhoods best",
                "evidence": ["comparison.metrics.trustworthiness.best"], **fields}
    return cli("recommend", "--run-dir", run, "--json", json.dumps(document))


def test_a_recommendation_needs_the_comparison_first(cli, pictured):
    assert _recommend(cli, pictured).code == 2


def test_a_recommendation_cites_a_measurement(cli, pictured):
    cli("compare", "--run-dir", pictured)

    result = _recommend(cli, pictured, evidence=["profile.shape.n_samples"])

    assert result.code == 2 and "comparison" in result.stderr


def test_a_recommendation_names_compared_candidates(cli, pictured):
    cli("compare", "--run-dir", pictured)

    assert _recommend(cli, pictured, recommended=["umap"]).code == 2


def test_a_recommendation_is_made_once(cli, pictured):
    cli("compare", "--run-dir", pictured)
    assert _recommend(cli, pictured).code == 0

    again = _recommend(cli, pictured, recommended=["pca"])

    assert again.code == 2 and "made once" in again.stderr


def test_status_finishes_evaluating_at_the_recommendation(cli, pictured):
    cli("compare", "--run-dir", pictured)
    assert cli("status", "--run-dir", pictured).payload["next"] == "evaluate"

    _recommend(cli, pictured)

    assert cli("status", "--run-dir", pictured).payload["next"] == "report"


def test_only_the_user_adopts_and_only_after_the_recommendation(cli, pictured):
    cli("compare", "--run-dir", pictured)
    choice = json.dumps({"candidate": "isomap", "rationale": "I prefer the layout"})
    assert cli("adopt", "--run-dir", pictured, "--json", choice).code == 2

    _recommend(cli, pictured)
    result = cli("adopt", "--run-dir", pictured, "--json", choice)

    assert result.code == 0 and result.payload["decided_by"] == "user"


def test_the_report_carries_the_comparison_and_who_chose(cli, pictured):
    cli("compare", "--run-dir", pictured)
    _recommend(cli, pictured)
    cli("figures", "--run-dir", pictured)

    document = (pictured / "report.md")
    assert cli("report", "--run-dir", pictured).code == 0
    text = document.read_text(encoding="utf-8")

    assert "## 7. Comparison and recommendation" in text
    assert "Recommended by the agent after seeing the results" in text
    assert "The user did not choose a picture." in text
    assert "No candidate was ranked." in text
    assert "How to read a tsne picture." in text


def test_every_candidate_gets_its_diagnostics_in_a_visualization_run(cli, pictured):
    cli("compare", "--run-dir", pictured)

    drawn = cli("figures", "--run-dir", pictured).payload

    for candidate in ("pca", "isomap", "tsne"):
        assert f"shepard_{candidate}" in drawn


# ---------------------------------------------------------------- plots A and B


def test_plot_a_is_a_rotation_so_a_2_d_embedding_keeps_every_distance():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(50, 2)) @ np.array([[3.0, 1.0], [0.0, 0.5]])

    rotated, share = principal_axes(xy)

    before = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    after = np.linalg.norm(rotated[:, None] - rotated[None], axis=-1)
    assert np.allclose(before, after) and share == pytest.approx(1.0)


def test_plot_a_states_the_share_its_axes_carry():
    rng = np.random.default_rng(1)
    embedding = rng.normal(size=(200, 5)) * np.array([5.0, 3.0, 1.0, 1.0, 1.0])

    _, share = principal_axes(embedding)

    assert 0.7 < share < 1.0


def test_plot_b_is_omitted_at_d_2():
    coordinates, record = draw_plots(np.random.default_rng(0).normal(size=(40, 2)),
                                     n_neighbors=10, seed=0)

    assert set(coordinates) == {"A"} and record["B"] is None


def test_a_representation_run_records_plots_a_and_b(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    reconnoitre(cli, run)
    (run / "plan.json").write_text(json.dumps(complete(_plan([("pca", PCA)]))),
                                   encoding="utf-8")
    cli("validate-plan", "--run-dir", run)
    cli("prepare-reference", "--run-dir", run)
    cli("embed", "--run-dir", run, "--id", "pca", "--in-process")

    record = cli("evaluate", "--run-dir", run, "--id", "pca").payload

    plots = record["plots"]
    assert (run / "plots" / "pca_A.npy").exists()
    if plots["d"] > 2:
        assert plots["B"]["method"] == "umap" and (run / "plots" / "pca_B.npy").exists()
    else:
        assert plots["B"] is None


# ------------------------------------------------------------------ the readings


def test_every_method_that_can_be_a_picture_says_how_to_read_it():
    registry = load_registry()
    for name, spec in registry.ops.items():
        if spec.is_reduction or spec.is_visualization:
            assert all(spec.reading[field] for field in READING_FIELDS), name


def test_a_judgment_of_an_earlier_portfolio_is_not_shown_as_current(cli, pictured):
    """Found by review: after a re-plan round the old recommendation and adoption
    stayed on disk and the report presented them as made of the new comparison."""
    cli("compare", "--run-dir", pictured)
    _recommend(cli, pictured)
    cli("adopt", "--run-dir", pictured, "--json", json.dumps({"candidate": "isomap"}))

    plan = json.loads((pictured / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"].append({"id": "umap", "stages": [{"op": "umap", "params": {}}],
                               "evidence": ["profile.shape.n_samples"]})
    (pictured / "plan.json").write_text(json.dumps(complete(plan)), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", pictured).code == 0
    cli("embed", "--run-dir", pictured, "--id", "umap", "--in-process")
    cli("evaluate", "--run-dir", pictured, "--id", "umap")
    cli("compare", "--run-dir", pictured)

    refused = cli("adopt", "--run-dir", pictured, "--json", json.dumps({"candidate": "umap"}))
    assert refused.code == 2 and "current comparison" in refused.stderr
    assert cli("status", "--run-dir", pictured).payload["next"] == "evaluate"
    assert cli("report", "--run-dir", pictured).code == 2

    assert _recommend(cli, pictured, recommended=["umap"]).code == 0
    cli("report", "--run-dir", pictured)
    text = (pictured / "report.md").read_text(encoding="utf-8")
    assert "**umap**" in text
    assert "The user adopted" not in text and "The user did not choose" in text
