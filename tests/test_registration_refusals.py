"""What registration refuses (day 12).

Each refusal is tested beside a plan that differs from it in the one respect the rule
reads and is accepted, so every rule is known to be able to pass as well as to fail.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from drtools.pipeline import PipelineError, validate_stages
from drtools.plan import validate_plan
from drtools.registry import load_registry
from plans import complete


def profile(n_samples: int = 300, n_features: int = 10, *, labels: bool = False) -> dict:
    return {
        "shape": {"n_samples": n_samples, "n_features": n_features, "storage": "dense"},
        "values": {"suspected_kind": "continuous"},
        "labels": {"present": labels},
    }


def stage(op: str, **params) -> dict:
    return {"op": op, "params": params}


def plan(candidates, **overrides) -> dict:
    document = {
        "dataset": "blobs",
        "candidates": [
            {"id": f"c{i}", "stages": stages} for i, stages in enumerate(candidates)
        ],
        "evaluation": {
            "weights": {"trustworthiness": 0.25, "continuity": 0.25,
                        "shepard_correlation": 0.5},
        },
    }
    document.update(overrides)
    return document


def codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["findings"]}


BASELINE = [stage("pca", n_components=2)]


# ------------------------------------------------------------------ chain length


def test_a_second_pca_after_pca_is_refused() -> None:
    with pytest.raises(PipelineError, match="stage 1 .*pca.*repeat"):
        validate_stages([stage("pca", n_components=10), stage("pca", n_components=2)])


def test_pca_twice_before_a_method_is_refused() -> None:
    with pytest.raises(PipelineError, match="stage 1 .*pca.*repeat"):
        validate_stages([
            stage("pca", n_components=20),
            stage("pca", n_components=10),
            stage("umap", n_components=2),
        ])


def test_a_third_method_is_refused_even_when_all_three_differ() -> None:
    # Unreachable while only PCA may stand before another method, so the registry is
    # widened here: the rule must not depend on the roles happening to forbid it.
    registry = load_registry()
    kernel = registry.ops["kernel_pca"]
    widened = replace(kernel, raw={**kernel.raw, "roles": ["intermediate", "terminal"]})
    registry = replace(registry, ops={**registry.ops, "kernel_pca": widened})
    with pytest.raises(PipelineError, match="stage 2 .*umap.*at most two"):
        validate_stages(
            [
                stage("pca", n_components=20),
                stage("kernel_pca", n_components=10),
                stage("umap", n_components=2),
            ],
            registry,
        )


def test_a_pre_step_and_a_different_method_are_accepted() -> None:
    validate_stages([
        stage("standardise"),
        stage("pca", n_components=10),
        stage("umap", n_components=2),
    ])


# ------------------------------------------------------------ the base preprocessing


def test_a_reduction_in_the_base_preprocessing_is_refused() -> None:
    report = validate_plan(
        plan([BASELINE], base_preprocessing=[stage("pca", n_components=5)]), profile()
    )
    assert "reduction_in_base" in codes(report)
    assert not report["valid"]


def test_a_base_of_preprocessing_only_draws_no_such_finding() -> None:
    report = validate_plan(plan([BASELINE], base_preprocessing=[stage("standardise")]),
                           profile())
    assert "reduction_in_base" not in codes(report)


# -------------------------------------------------------------- the linear baseline


def test_a_subsampled_pca_is_not_the_linear_baseline() -> None:
    report = validate_plan(
        plan([[stage("subsample", n_samples=50), stage("pca", n_components=2)]]),
        profile(),
    )
    assert "no_linear_baseline" in codes(report)


def test_a_pca_behind_a_candidate_specific_preprocessing_is_not_the_baseline() -> None:
    report = validate_plan(
        plan([[stage("standardise"), stage("pca", n_components=2)]]), profile()
    )
    assert "no_linear_baseline" in codes(report)


def test_a_candidate_of_one_pca_stage_is_the_baseline() -> None:
    report = validate_plan(plan([BASELINE]), profile())
    assert "no_linear_baseline" not in codes(report)


# --------------------------------------------------------------- limits on d


def test_hessian_lle_below_its_neighbour_minimum_is_refused_at_registration() -> None:
    lle = stage("lle", method="hessian", n_components=4, n_neighbors=10)
    report = validate_plan(plan([BASELINE, [lle]]), profile())
    found = [f for f in report["findings"] if f["code"] == "d_limit_violated"]
    assert found and found[0]["op"] == "lle" and "15" in found[0]["message"]


def test_hessian_lle_at_its_neighbour_minimum_is_accepted() -> None:
    lle = stage("lle", method="hessian", n_components=4, n_neighbors=15)
    report = validate_plan(plan([BASELINE, [lle]]), profile())
    assert "d_limit_violated" not in codes(report)


# --------------------------------------------------------------- subsampling


def test_a_subsample_the_method_does_not_need_is_refused() -> None:
    candidate = [stage("subsample", n_samples=100), stage("isomap", n_components=2)]
    report = validate_plan(plan([BASELINE, candidate]), profile(n_samples=300))
    found = [f for f in report["findings"] if f["code"] == "subsample_not_needed"]
    assert found and "5,000" in found[0]["message"]


def test_a_subsample_above_the_method_limit_is_accepted() -> None:
    candidate = [stage("subsample", n_samples=3000), stage("isomap", n_components=2)]
    report = validate_plan(plan([BASELINE, candidate]), profile(n_samples=20000))
    assert "subsample_not_needed" not in codes(report)
    assert "exceeds_scale_limit" not in codes(report)


def test_a_method_that_cannot_place_new_rows_is_refused_above_its_limit() -> None:
    candidate = [stage("subsample", n_samples=3000), stage("mds", n_components=2)]
    report = validate_plan(plan([BASELINE, candidate]), profile(n_samples=20000))
    found = [f for f in report["findings"] if f["code"] == "cannot_place_new_rows"]
    assert found and "profile.shape.n_samples" in found[0]["fix"]


def test_the_same_method_within_its_limit_draws_no_such_finding() -> None:
    report = validate_plan(plan([BASELINE, [stage("mds", n_components=2)]]),
                           profile(n_samples=300))
    assert "cannot_place_new_rows" not in codes(report)


def test_the_scale_limit_fix_does_not_offer_a_subsample_to_mds() -> None:
    report = validate_plan(plan([BASELINE, [stage("mds", n_components=2)]]),
                           profile(n_samples=20000))
    for finding in report["findings"]:
        if finding["op"] == "mds":
            assert "insert a subsample" not in finding["fix"]


# ------------------------------------------------ every method accounted for


def reject(method: str, evidence=("profile.shape.n_samples",)) -> dict:
    return {"method": method, "reason": "a reason", "evidence": list(evidence)}


def test_a_method_neither_nominated_nor_rejected_is_refused() -> None:
    report = validate_plan(plan([BASELINE]), profile())
    found = [f for f in report["findings"] if f["code"] == "method_unaccounted"]
    assert found and "isomap" in found[0]["message"] and "umap" in found[0]["message"]
    assert not report["valid"]


def test_a_plan_accounting_for_every_method_draws_no_such_finding() -> None:
    report = validate_plan(complete(plan([BASELINE])), profile())
    assert "method_unaccounted" not in codes(report)
    assert report["valid"], report["findings"]


def test_a_method_rejected_and_run_behind_a_pre_step_is_accepted() -> None:
    pre_step = [stage("pca", n_components=5), stage("umap", n_components=2)]
    document = complete(plan([BASELINE, pre_step], rejected=[reject("umap")]))
    report = validate_plan(document, profile())
    assert report["valid"], report["findings"]


def test_a_method_rejected_and_run_alone_is_refused() -> None:
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]],
                             rejected=[reject("umap")]))
    found = [f for f in validate_plan(document, profile())["findings"]
             if f["code"] == "rejected_method_run_alone"]
    assert found and found[0]["candidate"] == "c1" and found[0]["op"] == "umap"


def test_pca_cannot_be_rejected_since_the_baseline_runs_it_alone() -> None:
    document = complete(plan([BASELINE], rejected=[reject("pca")]))
    assert "rejected_method_run_alone" in codes(validate_plan(document, profile()))


@pytest.mark.parametrize("method", ["not_a_method", "standardise"])
def test_a_rejection_must_name_a_method(method: str) -> None:
    document = complete(plan([BASELINE], rejected=[reject(method)]))
    found = [f for f in validate_plan(document, profile())["findings"]
             if f["code"] == "unknown_rejected_method"]
    assert found and found[0]["op"] == method


# ------------------------------------------------------------------ evidence


def test_a_candidate_citing_nothing_is_refused() -> None:
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]]))
    document["candidates"][1]["evidence"] = []
    found = [f for f in validate_plan(document, profile())["findings"]
             if f["code"] == "unevidenced_candidate"]
    assert found and found[0]["severity"] == "error" and found[0]["candidate"] == "c1"


def test_the_linear_baseline_needs_no_evidence() -> None:
    document = complete(plan([BASELINE]))
    document["candidates"][0]["evidence"] = []
    assert "unevidenced_candidate" not in codes(validate_plan(document, profile()))


def test_a_rejection_citing_nothing_is_now_refused() -> None:
    document = complete(plan([BASELINE], rejected=[reject("mds", evidence=())]))
    found = [f for f in validate_plan(document, profile())["findings"]
             if f["code"] == "unevidenced_rejection"]
    assert found and found[0]["severity"] == "error"


@pytest.mark.parametrize("where", ["candidate", "rejection", "evaluation"])
def test_an_evidence_key_that_does_not_resolve_is_refused(where: str) -> None:
    broken = "recon.this.key.does.not.exist"
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]],
                             rejected=[reject("mds")]))
    if where == "candidate":
        document["candidates"][1]["evidence"] = [broken]
    elif where == "rejection":
        document["rejected"][0]["evidence"] = [broken]
    else:
        document["evaluation"]["evidence"] = [broken]
    found = [f for f in validate_plan(document, profile())["findings"]
             if f["code"] == "unresolved_evidence"]
    assert found and broken in found[0]["message"]
    assert "profile" in found[0]["message"]  # names what the run does hold


def test_a_key_that_resolves_is_accepted() -> None:
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]]))
    document["candidates"][1]["evidence"] = ["profile.shape.n_features"]
    assert "unresolved_evidence" not in codes(validate_plan(document, profile()))


def test_a_plan_cannot_cite_itself() -> None:
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]]))
    document["candidates"][1]["evidence"] = ["plan.candidates"]
    artifacts = {"profile": profile(), "plan": document}
    report = validate_plan(document, profile(), artifacts=artifacts)
    found = [f for f in report["findings"] if f["code"] == "unresolved_evidence"]
    assert found and "cannot cite itself" in found[0]["message"]


# ---------------------------------------------------------------- the weighting


def weights(document: dict, values: dict, evidence=None) -> dict:
    document["evaluation"] = {"weights": values, "justification": "a reason"}
    if evidence is not None:
        document["evaluation"]["evidence"] = evidence
    return document


UNLABELLED = {"trustworthiness": 0.25, "continuity": 0.25, "shepard_correlation": 0.5}
LABELLED = {"trustworthiness": 0.175, "continuity": 0.175, "shepard_correlation": 0.35,
            "knn_label_preservation": 0.2, "silhouette": 0.1}


def test_runtime_can_no_longer_be_weighted() -> None:
    document = weights(complete(plan([BASELINE])),
                       {"trustworthiness": 0.9, "runtime_s": 0.1},
                       ["profile.shape.n_samples"])
    found = [f for f in validate_plan(document, profile())["findings"]
             if f["code"] == "invalid_weighting"]
    assert found and "runtime_s" in found[0]["message"]


def test_a_label_metric_without_labels_is_refused() -> None:
    document = weights(complete(plan([BASELINE])), LABELLED, ["profile.shape.n_samples"])
    found = [f for f in validate_plan(document, profile(labels=False))["findings"]
             if f["code"] == "label_metric_without_labels"]
    assert found and "silhouette" in found[0]["message"]


def test_a_label_metric_with_labels_is_accepted() -> None:
    document = weights(complete(plan([BASELINE])), LABELLED, ["profile.labels.present"])
    report = validate_plan(document, profile(labels=True))
    assert "label_metric_without_labels" not in codes(report)
    assert report["weighting"] == "default_trusted_labels"


def test_the_default_needs_no_evidence_without_labels() -> None:
    document = weights(complete(plan([BASELINE])), dict(UNLABELLED), [])
    report = validate_plan(document, profile(labels=False))
    assert report["valid"], report["findings"]
    assert report["weighting"] == "default"


def test_a_departure_citing_nothing_is_refused() -> None:
    document = weights(complete(plan([BASELINE])),
                       {"trustworthiness": 0.5, "continuity": 0.5}, [])
    report = validate_plan(document, profile(labels=False))
    assert "uncited_weighting" in codes(report)
    assert report["weighting"] == "departure"


def test_a_departure_citing_evidence_is_accepted() -> None:
    document = weights(complete(plan([BASELINE])),
                       {"trustworthiness": 0.5, "continuity": 0.5},
                       ["profile.shape.n_samples"])
    report = validate_plan(document, profile(labels=False))
    assert report["valid"], report["findings"]
    assert report["weighting"] == "departure"


@pytest.mark.parametrize("values", [UNLABELLED, LABELLED])
def test_with_labels_even_a_default_must_cite_evidence(values: dict) -> None:
    # Whether the labels are trusted is itself a decision, and either default makes it.
    document = weights(complete(plan([BASELINE])), dict(values), [])
    found = [f for f in validate_plan(document, profile(labels=True))["findings"]
             if f["code"] == "uncited_weighting"]
    assert found and "trusted" in found[0]["message"]


def test_a_zero_weight_is_the_same_as_an_absent_one() -> None:
    document = weights(complete(plan([BASELINE])), {**UNLABELLED, "silhouette": 0.0}, [])
    assert validate_plan(document, profile())["weighting"] == "default"


def test_matching_tolerates_rounding_in_the_written_weights() -> None:
    near = {"trustworthiness": 0.2500000001, "continuity": 0.25,
            "shepard_correlation": 0.4999999999}
    document = weights(complete(plan([BASELINE])), near, [])
    assert validate_plan(document, profile())["weighting"] == "default"


# ------------------------------------------------------------ through the CLI


def _registered(cli, csv_dataset, tmp_path, document, *, recon=False):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    if recon:
        cli("recon", "--run-dir", runs / "r1")
    (runs / "r1" / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    return runs / "r1", cli("validate-plan", "--run-dir", runs / "r1")


def _records(run, stage_name):
    lines = (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["stage"] == stage_name]


def test_the_registration_record_names_the_weighting(cli, csv_dataset, tmp_path):
    run, result = _registered(cli, csv_dataset, tmp_path, complete(plan([BASELINE])))
    assert result.code == 0, result.payload
    assert _records(run, "register_plan")[0]["weighting"] == "default"


def test_a_key_into_the_runs_recon_resolves(cli, csv_dataset, tmp_path):
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]]))
    document["candidates"][1]["evidence"] = ["recon.neighbourhood.n_connected_components"]
    _, result = _registered(cli, csv_dataset, tmp_path, document, recon=True)
    assert result.code == 0, result.payload


def test_the_same_key_without_recon_is_refused(cli, csv_dataset, tmp_path):
    document = complete(plan([BASELINE, [stage("umap", n_components=2)]]))
    document["candidates"][1]["evidence"] = ["recon.neighbourhood.n_connected_components"]
    run, result = _registered(cli, csv_dataset, tmp_path, document)
    assert "unresolved_evidence" in codes(result.payload)
    assert not (run / "plan.registered.json").exists()


def test_the_weighting_evidence_is_frozen_with_the_weights(cli, csv_dataset, tmp_path):
    run, result = _registered(cli, csv_dataset, tmp_path, complete(plan([BASELINE])))
    assert result.code == 0
    document = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    document["evaluation"]["evidence"] = ["profile.shape.n_features"]
    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")

    again = cli("validate-plan", "--run-dir", run)
    assert again.code == 2
    assert "evaluation" in again.stderr and "evidence" in again.stderr


# ------------------------------------------------- suggestions and provenance


TSNE_OP = "tsne"


def _profiled(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    return runs / "r1"


def _register(cli, run, document):
    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    return cli("validate-plan", "--run-dir", run)


def _tsne_plan(**tsne_stage):
    tsne = {"op": "tsne", "params": {"n_components": 2, **tsne_stage.get("params", {})}}
    if "overrides" in tsne_stage:
        tsne["overrides"] = tsne_stage["overrides"]
    return complete(plan([BASELINE, [tsne]]))


def test_suggest_params_persists_its_answer(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("suggest-params", "--op", "tsne", "--run-dir", run)
    assert result.code == 0
    stored = json.loads((run / "suggestions" / "tsne.json").read_text(encoding="utf-8"))
    assert stored["suggested"]["perplexity"]["value"] == 5.0
    assert stored["suggested"] == result.payload["suggested"]


def test_suggest_params_takes_the_stage_settings_it_depends_on(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    settings = json.dumps({"method": "hessian", "n_components": 4})
    result = cli("suggest-params", "--op", "lle", "--params", settings, "--run-dir", run)
    assert result.payload["suggested"]["n_neighbors"]["value"] == 15
    stored = json.loads((run / "suggestions" / "lle.json").read_text(encoding="utf-8"))
    assert stored["for_params"] == {"method": "hessian", "n_components": 4}


def test_suggest_base_persists_its_answer(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("suggest-base", "--run-dir", run)
    stored = json.loads((run / "suggestions" / "base.json").read_text(encoding="utf-8"))
    assert stored["stages"] == result.payload["stages"]


def _provenance(run, candidate="c1", op="tsne"):
    record = _records(run, "register_plan")[-1]
    stages = record["provenance"][candidate]
    return next(s for s in stages if s["op"] == op)["params"]


def test_a_value_equal_to_the_suggestion_is_recorded_as_suggested(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("suggest-params", "--op", "tsne", "--run-dir", run)
    result = _register(cli, run, _tsne_plan(params={"perplexity": 5}))
    assert result.code == 0, result.payload
    params = _provenance(run)
    assert params["perplexity"]["state"] == "suggested"
    assert params["n_components"]["state"] == "specified"  # nothing was suggested for it
    assert params["n_iter"]["state"] == "registry_default"


def test_an_override_without_a_reason_is_refused(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("suggest-params", "--op", "tsne", "--run-dir", run)
    result = _register(cli, run, _tsne_plan(params={"perplexity": 8}))
    found = [f for f in result.payload["findings"] if f["code"] == "unexplained_override"]
    assert found and "perplexity" in found[0]["message"] and "5" in found[0]["message"]
    assert not (run / "plan.registered.json").exists()


def test_an_override_with_a_reason_is_recorded_with_it(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("suggest-params", "--op", "tsne", "--run-dir", run)
    reason = {"perplexity": {"reason": "a wider neighbourhood for the clusters",
                             "evidence": ["suggestions.tsne.suggested.perplexity.value"]}}
    result = _register(cli, run, _tsne_plan(params={"perplexity": 8}, overrides=reason))
    assert result.code == 0, result.payload
    perplexity = _provenance(run)["perplexity"]
    assert perplexity == {"state": "overridden", "suggested": 5.0,
                          "reason": "a wider neighbourhood for the clusters"}


def _with_suggestion(value, *, given=None, overrides=None):
    suggestions = {"tsne": {"suggested": {"perplexity": {
        "value": value, "rationale": "", "evidence": []}}}}
    tsne = {"op": "tsne", "params": {"n_components": 2, **({"perplexity": given}
                                                          if given is not None else {})}}
    if overrides:
        tsne["overrides"] = overrides
    document = complete(plan([BASELINE, [tsne]]))
    return validate_plan(document, profile(),
                         artifacts={"profile": profile(), "suggestions": suggestions})


def test_leaving_out_a_parameter_whose_suggestion_differs_is_an_override() -> None:
    """Omitting the parameter must not be a way around giving the reason."""
    report = _with_suggestion(5.0)  # the registry default is 30
    found = [f for f in report["findings"] if f["code"] == "unexplained_override"]
    assert found and "left unset" in found[0]["message"]
    perplexity = report["provenance"]["c1"][0]["params"]["perplexity"]
    assert perplexity == {"state": "overridden", "suggested": 5.0}


def test_leaving_out_a_parameter_with_a_reason_is_accepted() -> None:
    report = _with_suggestion(5.0, overrides={"perplexity": {"reason": "the default suits"}})
    assert "unexplained_override" not in codes(report)
    assert report["provenance"]["c1"][0]["params"]["perplexity"]["reason"] == "the default suits"


def test_leaving_out_a_parameter_whose_suggestion_is_the_default_follows_it() -> None:
    report = _with_suggestion(30.0)
    assert "unexplained_override" not in codes(report)
    assert report["provenance"]["c1"][0]["params"]["perplexity"]["state"] == "registry_default"


def test_a_pre_step_suggestion_is_not_held_against_the_baseline() -> None:
    """The PCA component count is for a pre-step; a terminal PCA is not overriding it."""
    suggestions = {"pca": {"suggested": {"n_components": {
        "value": 10, "rationale": "", "evidence": [], "applies_to": "intermediate"}}}}
    pre_step = [stage("pca", n_components=10), stage("umap", n_components=2)]
    document = complete(plan([BASELINE, pre_step]))
    report = validate_plan(document, profile(),
                           artifacts={"profile": profile(), "suggestions": suggestions})
    assert "unexplained_override" not in codes(report)
    baseline, chained = report["provenance"]["c0"], report["provenance"]["c1"]
    assert baseline[0]["params"]["n_components"]["state"] == "specified"
    assert chained[0]["params"]["n_components"]["state"] == "suggested"


def test_the_pca_suggestion_says_it_is_for_a_pre_step() -> None:
    from drtools.heuristics import suggest

    recon = {"spectrum": {"probe": {"elbow": 8, "n_components_for_90pct": 20}}}
    entry = suggest("pca", {"shape": {"n_samples": 500}}, recon)["n_components"]
    assert entry["applies_to"] == "intermediate"


def test_an_op_with_suggestions_never_requested_draws_a_warning(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = _register(cli, run, _tsne_plan(params={"perplexity": 5}))
    found = [f for f in result.payload["findings"] if f["code"] == "no_suggestion_requested"]
    assert found and found[0]["severity"] == "warning" and found[0]["op"] == "tsne"
    assert _provenance(run)["perplexity"]["state"] == "specified"


def test_re_running_suggest_params_after_registration_does_not_relabel(
    cli, csv_dataset, tmp_path
):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("suggest-params", "--op", "tsne", "--run-dir", run)
    assert _register(cli, run, _tsne_plan(params={"perplexity": 5})).code == 0
    (run / "suggestions" / "tsne.json").write_text(json.dumps(
        {"op": "tsne", "for_params": {}, "suggested": {
            "perplexity": {"value": 9.0, "rationale": "", "evidence": []}}}),
        encoding="utf-8")
    assert cli("embed", "--run-dir", run, "--id", "c1", "--in-process").code == 0
    record = json.loads((run / "embeddings" / "c1.json").read_text(encoding="utf-8"))
    tsne = next(s for s in record["stages"] if s["op"] == "tsne")
    assert tsne["param_provenance"]["perplexity"] == "suggested"


def test_the_registration_records_whether_the_base_followed_its_suggestion(
    cli, csv_dataset, tmp_path
):
    run = _profiled(cli, csv_dataset, tmp_path)
    suggested = cli("suggest-base", "--run-dir", run).payload["stages"]
    assert _register(cli, run, complete(plan([BASELINE], base_preprocessing=suggested))).code == 0
    assert _records(run, "register_plan")[-1]["base_matches_suggestion"] is True


def test_the_report_prints_the_override_and_its_reason(cli, csv_dataset, tmp_path):
    from drtools.report import _block_hyperparameters
    from drtools.runs import RunDir

    run = _profiled(cli, csv_dataset, tmp_path)
    cli("suggest-params", "--op", "tsne", "--run-dir", run)
    reason = {"perplexity": {"reason": "a wider neighbourhood for the clusters"}}
    assert _register(cli, run, _tsne_plan(params={"perplexity": 8}, overrides=reason)).code == 0
    cli("embed", "--run-dir", run, "--id", "c1", "--in-process")
    block = _block_hyperparameters(RunDir(run))
    assert "overridden" in block and "a wider neighbourhood for the clusters" in block
