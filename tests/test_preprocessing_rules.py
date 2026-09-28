"""Preprocessing by rule (day 13, section 3.10).

Each rule is tested beside a case that differs from it in the one respect the rule reads,
so every rule is known to be able to pass as well as to fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from drtools.executors import Context
from drtools.executors.preprocessing import constant_features, drop_constant


# ------------------------------------------------------------- the constant test


def _columns(*columns: np.ndarray) -> np.ndarray:
    return np.column_stack(columns)


RNG = np.random.default_rng(0)
N = 1000
VARYING = RNG.normal(size=N)


def test_a_column_of_one_stored_value_is_constant() -> None:
    constant, by_tolerance = constant_features(_columns(np.full(N, 7.7), VARYING))
    assert constant.tolist() == [True, False]
    assert by_tolerance.tolist() == [False, False]  # its range is exactly zero


def test_a_column_constant_only_up_to_rounding_is_dropped_and_named() -> None:
    """Defect 19: variance > 0 kept such a column, and a z-score blew it up."""
    rounded = 0.1 + RNG.uniform(-1e-16, 1e-16, size=N)
    constant, by_tolerance = constant_features(_columns(rounded, VARYING))
    assert constant.tolist() == [True, False]
    assert by_tolerance.tolist() == [True, False]


def test_the_tolerance_is_relative_for_large_values() -> None:
    large = 1e6 + RNG.uniform(0, 1e-7, size=N)
    assert constant_features(_columns(large, VARYING))[0].tolist() == [True, False]


def test_the_floor_makes_the_tolerance_absolute_for_small_values() -> None:
    tiny = 1e-13 * (1 + 0.5 * RNG.uniform(size=N))
    small = 1e-9 * (1 + 0.5 * RNG.uniform(size=N))
    constant, _ = constant_features(_columns(tiny, small))
    assert constant.tolist() == [True, False]


def test_a_sparse_column_counts_its_implicit_zeros() -> None:
    rare = np.zeros(N)
    rare[:: 10] = 5.0
    X = sp.csr_matrix(_columns(rare, np.zeros(N), np.full(N, 3.0)))
    constant, by_tolerance = constant_features(X)
    assert constant.tolist() == [False, True, True]
    assert not by_tolerance.any()


def test_drop_constant_records_the_features_the_tolerance_dropped() -> None:
    rounded = 0.1 + RNG.uniform(-1e-16, 1e-16, size=N)
    X = _columns(VARYING, np.full(N, 2.0), rounded, VARYING * 2)
    out, notes = drop_constant(X, Context())
    assert out.shape == (N, 2)
    assert notes["n_dropped"] == 2
    assert notes["dropped_by_tolerance"] == [2]  # original column index


def test_the_profile_counts_constants_by_the_same_test() -> None:
    from drtools.profile import profile_dataset

    rounded = 0.1 + RNG.uniform(-1e-16, 1e-16, size=N)
    X = _columns(VARYING, rounded, np.full(N, 2.0))
    features = profile_dataset(X, None, {"name": "t"})["features"]
    assert features["n_constant"] == 2
    assert "n_near_constant" not in features


# ----------------------------------------------------- evidence for feature type


def _mixed_table():
    sex = RNG.integers(0, 2, size=N).astype(float)
    age = RNG.integers(18, 90, size=N).astype(float)
    height = RNG.normal(170, 10, size=N)
    weight = RNG.normal(70, 12, size=N)
    X = _columns(sex, age, height, weight, np.full(N, 1.0))
    meta = {"name": "people", "source": "people.csv",
            "feature_names": ["sex", "age", "height", "weight", "constant"]}
    return X, meta


def test_the_profile_counts_columns_by_kind_with_examples() -> None:
    from drtools.profile import profile_dataset

    X, meta = _mixed_table()
    kinds = profile_dataset(X, None, meta)["features"]["column_kinds"]
    assert kinds["n_binary"] == 1 and kinds["n_integer"] == 1 and kinds["n_continuous"] == 2
    assert kinds["examples"]["binary"] == ["sex"]
    assert kinds["examples"]["continuous"] == ["height", "weight"]
    assert "constant" not in sum(kinds["examples"].values(), [])  # constants excluded


def test_whole_numbered_counts_show_no_continuous_columns() -> None:
    from drtools.profile import profile_dataset

    counts = RNG.poisson(3.0, size=(N, 6)).astype(float)
    kinds = profile_dataset(counts, None, {"name": "c", "source": "synthetic"})[
        "features"]["column_kinds"]
    assert kinds["n_continuous"] == 0 and kinds["n_integer"] + kinds["n_binary"] == 6


def test_the_profile_records_the_source_format_and_the_first_names() -> None:
    from drtools.profile import profile_dataset

    X, meta = _mixed_table()
    features = profile_dataset(X, None, meta)["features"]
    assert features["source_format"] == ".csv"
    assert features["names_head"] == ["sex", "age", "height", "weight", "constant"]


def test_an_observation_reads_the_mix_of_kinds_aloud() -> None:
    from drtools.profile import profile_dataset

    X, meta = _mixed_table()
    observations = profile_dataset(X, None, meta)["observations"]
    found = [o for o in observations if "features.column_kinds" in " ".join(o["evidence"])]
    assert found and "mixed" in found[0]["observation"]


def test_column_kinds_agree_on_sparse_and_dense_input() -> None:
    from drtools.profile import profile_dataset

    X, meta = _mixed_table()
    X[: N // 2, 1] = 0.0  # sparse-looking age column, still integer
    dense = profile_dataset(X, None, meta)["features"]["column_kinds"]
    sparse = profile_dataset(sp.csr_matrix(X), None, meta)["features"]["column_kinds"]
    assert dense == sparse


# ------------------------------------------------------------- the data decision


from drtools.decision import DataDecision, base_rule, default_decision  # noqa: E402


def _ops(stages):
    return [stage["op"] for stage in stages]


@pytest.mark.parametrize(
    ("values", "features", "expected"),
    [
        ("raw_counts", "one_type", ["drop_constant", "normalise_total", "log1p"]),
        ("raw_counts", "mixed", ["drop_constant", "normalise_total", "log1p", "standardise"]),
        ("not_counts", "one_type", ["drop_constant"]),
        ("not_counts", "mixed", ["drop_constant", "standardise"]),
    ],
)
def test_the_base_rule_follows_the_two_facts(values, features, expected) -> None:
    decision = DataDecision(values=values, features=features, decided_by="user")
    assert _ops(base_rule(decision)) == expected


def test_the_default_decision_suspects_counts_and_assumes_mixed() -> None:
    counts = default_decision({"values": {"suspected_kind": "counts"}})
    other = default_decision({"values": {"suspected_kind": "continuous"}})
    assert (counts.values, counts.features, counts.decided_by) == ("raw_counts", "mixed", "default")
    assert (other.values, other.features) == ("not_counts", "mixed")


def _profiled(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8), "--runs-root", runs, "--run-id", "r1")
    return runs / "r1"


def _decision(**fields):
    import json

    document = {"values": "not_counts", "features": "one_type", "decided_by": "user",
                "rationale": "declared when the analysis started"}
    document.update(fields)
    return json.dumps(document)


def test_recon_refuses_to_run_without_a_decision(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("recon", "--run-dir", run)
    assert result.code == 2
    assert "--decision" in result.stderr and "default" in result.stderr
    assert not (run / "recon.json").exists()


def test_an_agents_decision_must_cite_evidence(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("recon", "--run-dir", run, "--decision", _decision(decided_by="agent"))
    assert result.code == 2 and "evidence" in result.stderr


def test_an_agents_evidence_must_resolve(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("recon", "--run-dir", run, "--decision",
                 _decision(decided_by="agent", evidence=["profile.no.such.key"]))
    assert result.code == 2 and "profile.no.such.key" in result.stderr


def test_recon_records_the_decision_and_probes_under_it(cli, csv_dataset, tmp_path):
    import json

    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("recon", "--run-dir", run, "--decision", _decision(
        features="mixed", decided_by="agent", evidence=["profile.features.column_kinds"]))
    assert result.code == 0, result.stderr
    recon = json.loads((run / "recon.json").read_text(encoding="utf-8"))
    assert recon["data_decision"]["features"] == "mixed"
    assert recon["probe_representation"]["transform"] == ["drop_constant", "standardise"]
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    logged = [r for r in records if r["stage"] == "recon"][-1]
    assert logged["data_decision"]["decided_by"] == "agent"
    assert logged["evidence"] == ["profile.features.column_kinds"]


def test_suggest_base_offers_the_rule_for_the_recorded_decision(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    cli("recon", "--run-dir", run, "--decision", _decision(features="mixed"))
    suggested = cli("suggest-base", "--run-dir", run).payload
    assert _ops(suggested["stages"]) == ["drop_constant", "standardise"]
    assert "recon.data_decision" in suggested["evidence"]


def test_suggest_base_refuses_before_a_decision_exists(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    result = cli("suggest-base", "--run-dir", run)
    assert result.code == 2 and "recon" in result.stderr


def test_registration_refuses_without_a_recorded_decision() -> None:
    from drtools.plan import validate_plan
    from plans import checkpoint_for
    from plans import complete

    profile = {"shape": {"n_samples": 300, "n_features": 10, "storage": "dense"},
               "values": {"suspected_kind": "continuous"}, "labels": {"present": False}}
    document = complete({"dataset": "d", "candidates": [
        {"id": "a", "stages": [{"op": "pca", "params": {}}]}],
        "evaluation": {"weights": {"trustworthiness": 0.25, "continuity": 0.25,
                                   "shepard_correlation": 0.5}}})
    report = validate_plan(document, profile, recon=None,
                           checkpoint=checkpoint_for(document))
    assert "no_data_decision" in {f["code"] for f in report["findings"]}


# ------------------------------------------------------- registration: the base rule


from drtools.plan import estimate_peak_bytes, validate_plan  # noqa: E402
import json  # noqa: E402

from plans import CHECKPOINT, checkpoint_for  # noqa: E402
from plans import complete  # noqa: E402

GB = 1e9


def _profile(n=300, d=10, *, storage="dense", dtype="float32", sparsity=0.0):
    itemsize = 4 if dtype == "float32" else 8
    return {
        "shape": {"n_samples": n, "n_features": d, "storage": storage, "dtype": dtype,
                  "memory_mb": n * d * itemsize * (1 - sparsity) / 1e6},
        "values": {"suspected_kind": "continuous", "sparsity": sparsity},
        "labels": {"present": False},
    }


def _recon(values="not_counts", features="one_type"):
    return {"data_decision": {"values": values, "features": features,
                              "decided_by": "user", "rationale": "", "evidence": []}}


def _stage(op, **params):
    return {"op": op, "params": params}


PCA2 = [_stage("pca")]


def _plan(candidates, **overrides):
    document = {
        "dataset": "d",
        "base_preprocessing": [_stage("drop_constant")],
        "candidates": [{"id": f"c{i}", "stages": s} for i, s in enumerate(candidates)],
        "evaluation": {"weights": {"trustworthiness": 0.25, "continuity": 0.25,
                                   "shepard_correlation": 0.5}},
    }
    document.update(overrides)
    return complete(document, base=False)


def _codes(report):
    return {finding["code"] for finding in report["findings"]}


def _validate(document, profile=None, recon=None, **kwargs):
    kwargs.setdefault("checkpoint", checkpoint_for(document))
    return validate_plan(document, profile or _profile(), recon or _recon(), **kwargs)


def test_a_base_that_follows_the_rule_is_recorded_as_the_rule() -> None:
    report = _validate(_plan([PCA2]))
    assert report["valid"], report["findings"]
    assert report["base"] == "rule"


def test_a_base_departing_from_the_rule_without_a_reason_is_refused() -> None:
    report = _validate(_plan([PCA2]), recon=_recon(features="mixed"))  # standardise missing
    found = [f for f in report["findings"] if f["code"] == "unexplained_base_departure"]
    assert found and "standardise" in found[0]["message"]


def test_a_base_departure_with_a_reason_and_evidence_is_accepted() -> None:
    departure = {"reason": "the features share one unit", "evidence": ["profile.shape.n_features"]}
    report = _validate(_plan([PCA2], base_departure=departure), recon=_recon(features="mixed"))
    assert report["valid"], report["findings"]
    assert report["base"] == "departure"


def test_a_base_departure_needs_evidence() -> None:
    departure = {"reason": "the features share one unit", "evidence": []}
    report = _validate(_plan([PCA2], base_departure=departure), recon=_recon(features="mixed"))
    assert "unexplained_base_departure" in _codes(report)


def test_a_base_departure_citing_a_broken_key_is_refused() -> None:
    departure = {"reason": "r", "evidence": ["profile.no.such.key"]}
    report = _validate(_plan([PCA2], base_departure=departure), recon=_recon(features="mixed"))
    assert "unresolved_evidence" in _codes(report)


def test_a_parameter_set_by_hand_in_the_base_is_a_departure() -> None:
    base = [_stage("drop_constant"), _stage("normalise_total", target=1e4), _stage("log1p")]
    report = _validate(_plan([PCA2], base_preprocessing=base), recon=_recon(values="raw_counts"))
    assert "unexplained_base_departure" in _codes(report)


def test_drop_constant_must_come_first_whatever_the_reason() -> None:
    departure = {"reason": "r", "evidence": ["profile.shape.n_features"]}
    base = [_stage("standardise")]
    report = _validate(_plan([PCA2], base_preprocessing=base, base_departure=departure),
                       recon=_recon(features="mixed"))
    assert "drop_constant_not_first" in _codes(report)


def test_raw_counts_come_from_the_decision_not_the_profiles_guess() -> None:
    # The profile does not suspect counts, and the decision says they are.
    base = [_stage("drop_constant"), _stage("normalise_total"), _stage("log1p")]
    report = _validate(_plan([PCA2], base_preprocessing=base), recon=_recon(values="raw_counts"))
    assert report["valid"], report["findings"]
    report = _validate(_plan([PCA2]), recon=_recon(values="raw_counts"))
    assert "raw_counts_not_normalised" in _codes(report)


# ---------------------------------------------- registration: candidate-specific stages


WIDE = _profile(n=300, d=3000)
SELECT = [_stage("select_variable_features", n_features=2000), _stage("standardise")]
UMAP = _stage("umap", n_components=2)


def test_a_euclidean_first_method_on_many_features_of_one_type_must_select() -> None:
    report = _validate(_plan([PCA2, [_stage("pca"), UMAP]]), WIDE)
    found = [f for f in report["findings"] if f["code"] == "selection_required"]
    assert found and found[0]["candidate"] == "c1"


def test_the_required_selection_and_z_score_are_accepted() -> None:
    report = _validate(_plan([PCA2, [*SELECT, _stage("pca"), UMAP]]), WIDE)
    assert report["valid"], report["findings"]


def test_the_linear_baseline_never_selects() -> None:
    report = _validate(_plan([PCA2]), WIDE)
    assert "selection_required" not in _codes(report)
    report = _validate(_plan([PCA2, [*SELECT, _stage("pca")]]), WIDE)
    assert report["valid"], report["findings"]  # that one is not the baseline


def test_selection_alone_without_the_z_score_is_refused() -> None:
    selected = [_stage("select_variable_features", n_features=2000), _stage("umap", n_components=2)]
    assert "selection_required" in _codes(_validate(_plan([PCA2, selected]), WIDE))


def test_any_other_number_of_selected_features_is_refused() -> None:
    few = [_stage("select_variable_features", n_features=500), _stage("standardise"), UMAP]
    assert "selection_count" in _codes(_validate(_plan([PCA2, few]), WIDE))


@pytest.mark.parametrize(
    ("profile", "recon", "first"),
    [
        (WIDE, _recon(features="mixed"), UMAP),  # mixed: z-scored in the base instead
        (_profile(d=1500), _recon(), UMAP),  # 2,000 features or fewer
        (WIDE, _recon(), _stage("umap", n_components=2, metric="cosine")),  # not Euclidean
    ],
)
def test_selection_is_refused_where_the_rule_does_not_call_for_it(profile, recon, first) -> None:
    base = [_stage("drop_constant"), _stage("standardise")] if recon["data_decision"]["features"] == "mixed" else [_stage("drop_constant")]
    report = _validate(_plan([PCA2, [*SELECT, first]], base_preprocessing=base), profile, recon)
    assert "selection_forbidden" in _codes(report)
    report = _validate(_plan([PCA2, [first]], base_preprocessing=base), profile, recon)
    assert report["valid"], report["findings"]


def test_a_z_score_alone_in_a_candidate_is_refused() -> None:
    report = _validate(_plan([PCA2, [_stage("standardise"), UMAP]]))
    assert "selection_forbidden" in _codes(report)


# ------------------------------------------------ registration: the memory estimate


PATHMNIST = _profile(n=107_180, d=2352, dtype="float32")


def test_the_estimate_is_the_loaded_data_plus_the_heaviest_stage() -> None:
    from drtools.plan import _stage_peaks
    from drtools.registry import load_registry

    candidate = [_stage("drop_constant"), *SELECT, _stage("pca"), UMAP]
    loaded_bytes = 107_180 * 2352 * 4
    selected = 107_180 * 2000 * 4
    loaded, peaks = _stage_peaks(candidate, PATHMNIST, load_registry())
    by_op = dict(peaks)
    assert loaded == loaded_bytes
    # selection holds its full input beside its output; PCA its input and a centred copy,
    # and, since tuning chooses its k, the largest output it could choose: 100 columns
    assert by_op["select_variable_features"] == pytest.approx(loaded_bytes + selected)
    assert by_op["pca"] == pytest.approx(2 * selected + 107_180 * 100 * 4)
    assert estimate_peak_bytes(candidate, PATHMNIST) == pytest.approx(
        loaded_bytes + loaded_bytes + selected)


def test_a_z_score_of_sparse_input_is_counted_dense() -> None:
    sparse = _profile(n=100_000, d=2000, storage="sparse_csr", sparsity=0.9)
    dense_out = 100_000 * 2000 * 4
    assert estimate_peak_bytes([_stage("standardise"), _stage("pca")],
                               sparse) > 2 * dense_out


#: Sparse and wide: z-scoring 2,000 selected features turns a 0.34 GB sparse matrix into a
#: 0.86 GB dense one, and PCA's centred copy doubles it -- section 3.10's case.
SPARSE = _profile(n=107_000, d=20_000, storage="sparse_csr", sparsity=0.98)


def test_on_dense_data_the_baseline_is_the_heaviest_candidate() -> None:
    selecting = [_stage("drop_constant"), *SELECT, _stage("pca")]
    baseline = [_stage("drop_constant"), _stage("pca")]
    assert estimate_peak_bytes(baseline, PATHMNIST) > estimate_peak_bytes(selecting, PATHMNIST)


def test_a_candidate_over_the_limit_is_refused_naming_the_stage() -> None:
    candidate = [*SELECT, _stage("pca"), UMAP]
    report = _validate(_plan([PCA2, candidate]), SPARSE, memory_limit_bytes=int(1.5 * GB))
    found = [f for f in report["findings"] if f["code"] == "exceeds_memory_limit"]
    assert [f["candidate"] for f in found] == ["c1"]
    assert "pca" in found[0]["message"] and "subsample" in found[0]["fix"]


def test_memory_is_a_reason_to_subsample() -> None:
    candidate = [_stage("subsample", n_samples=20_000), *SELECT,
                 _stage("pca"), UMAP]
    report = _validate(_plan([PCA2, candidate]), SPARSE, memory_limit_bytes=int(1.5 * GB))
    assert "subsample_not_needed" not in _codes(report)
    assert "exceeds_memory_limit" not in _codes(report)


def test_a_subsample_within_both_limits_is_still_refused() -> None:
    candidate = [_stage("subsample", n_samples=20_000), *SELECT,
                 _stage("pca"), UMAP]
    report = _validate(_plan([PCA2, candidate]), SPARSE, memory_limit_bytes=int(3 * GB))
    assert "subsample_not_needed" in _codes(report)


def test_a_method_that_cannot_place_new_rows_is_refused_over_the_memory_limit() -> None:
    wide = _profile(n=4000, d=200_000, dtype="float64")  # within MDS's 5,000 rows
    candidate = [_stage("subsample", n_samples=1000), _stage("mds")]
    report = _validate(_plan([[_stage("pca")], candidate]), wide,
                       memory_limit_bytes=int(2 * GB))
    assert "cannot_place_new_rows" in _codes(report)


def test_the_linear_baseline_over_the_limit_says_the_plan_cannot_register() -> None:
    report = _validate(_plan([PCA2]), SPARSE, memory_limit_bytes=int(0.6 * GB))
    found = [f for f in report["findings"] if f["code"] == "exceeds_memory_limit"]
    assert found and "baseline" in found[0]["fix"]


# ------------------------------------------------------------------- executors


def test_standardise_densifies_sparse_input_instead_of_refusing() -> None:
    from drtools.executors.preprocessing import standardise

    X = sp.csr_matrix(RNG.poisson(1.0, size=(200, 5)).astype(np.float32))
    out, _ = standardise(X, Context())
    assert not sp.issparse(out)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(out.std(axis=0), 1, atol=1e-4)


def test_standardise_keeps_the_dtype_it_receives() -> None:
    from drtools.executors.preprocessing import standardise

    X = RNG.normal(size=(100, 4)).astype(np.float32)
    assert standardise(X, Context())[0].dtype == np.float32
    assert standardise(X.astype(np.float64), Context())[0].dtype == np.float64


def test_selection_ranks_by_variance_and_has_no_other_criterion() -> None:
    from drtools.registry import load_registry

    assert "criterion" not in load_registry()["select_variable_features"].params


# ---------------------------------------------- the memory limit, through the CLI


def _reconnoitred(cli, csv_dataset, tmp_path):
    run = _profiled(cli, csv_dataset, tmp_path)
    assert cli("recon", "--run-dir", run, "--decision", _decision()).code == 0
    assert cli("checkpoint", "--run-dir", run, "--answers",
               json.dumps(CHECKPOINT)).code == 0
    return run


def _register(cli, run, document):
    import json

    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    return cli("validate-plan", "--run-dir", run)


def _registrations(run):
    import json

    lines = (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["stage"] == "register_plan"]


def test_the_physical_memory_is_detected() -> None:
    from drtools.memory import physical_memory_bytes

    assert physical_memory_bytes() > 1e9


def test_the_limit_is_half_the_memory_and_recorded_at_first_registration(
    cli, csv_dataset, tmp_path, monkeypatch
):
    import drtools.memory

    monkeypatch.setattr(drtools.memory, "physical_memory_bytes", lambda: 16 * 2**30)
    run = _reconnoitred(cli, csv_dataset, tmp_path)
    assert _register(cli, run, _plan([PCA2])).code == 0
    record = _registrations(run)[0]
    assert record["memory"]["physical_bytes"] == 16 * 2**30
    assert record["memory"]["limit_bytes"] == 8 * 2**30
    assert record["base"] == "rule"


def test_a_re_registration_keeps_the_limit_the_run_first_recorded(
    cli, csv_dataset, tmp_path, monkeypatch
):
    import drtools.memory

    monkeypatch.setattr(drtools.memory, "physical_memory_bytes", lambda: 16 * 2**30)
    run = _reconnoitred(cli, csv_dataset, tmp_path)
    assert _register(cli, run, _plan([PCA2])).code == 0
    monkeypatch.setattr(drtools.memory, "physical_memory_bytes", lambda: 2 * 2**30)
    assert _register(cli, run, _plan([PCA2, [_stage("pca")]])).code == 0
    assert [r["memory"]["limit_bytes"] for r in _registrations(run)] == [8 * 2**30] * 2


def test_registration_holds_candidates_to_the_detected_limit(
    cli, csv_dataset, tmp_path, monkeypatch
):
    import drtools.memory

    monkeypatch.setattr(drtools.memory, "physical_memory_bytes", lambda: 2_000)
    run = _reconnoitred(cli, csv_dataset, tmp_path)
    result = _register(cli, run, _plan([PCA2]))
    assert "exceeds_memory_limit" in _codes(result.payload)


# ---------------------------------------------------------- the report, section 2


def _reported_run(cli, tmp_path, data, decision_fields, base, **plan_fields):
    import json

    from plans import complete, decision

    runs = tmp_path / "runs"
    assert cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1").code == 0
    run = runs / "r1"
    assert cli("recon", "--run-dir", run, "--decision", decision(**decision_fields)).code == 0
    assert cli("checkpoint", "--run-dir", run, "--answers",
               json.dumps(CHECKPOINT)).code == 0
    document = complete({
        "dataset": "d", "base_preprocessing": base,
        "candidates": [{"id": "pca2", "stages": [_stage("pca")]}],
        "evaluation": {"weights": {"trustworthiness": 0.25, "continuity": 0.25,
                                   "shepard_correlation": 0.5}},
        **plan_fields,
    }, base=False)
    (run / "plan.json").write_text(json.dumps(document), encoding="utf-8")
    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 0, result.payload
    assert cli("prepare-reference", "--run-dir", run).code == 0
    return run


def test_section_2_states_the_decision_and_the_counts_transform(cli, tmp_path):
    from drtools.report import _block_preprocessing
    from drtools.runs import RunDir

    base = [_stage("drop_constant"), _stage("normalise_total"), _stage("log1p")]
    run = _reported_run(cli, tmp_path, "sparse_counts",
                        {"values": "raw_counts", "decided_by": "user"}, base)
    block = _block_preprocessing(RunDir(run))
    assert "raw_counts" in block and "one_type" in block and "user" in block
    assert "The base follows the rule" in block
    assert "median total" in block and "log(1 + x)" in block
    assert "supplied transformed" in block


def test_section_2_names_a_base_departure_and_its_reason(cli, csv_dataset, tmp_path):
    from drtools.report import _block_preprocessing
    from drtools.runs import RunDir

    base = [_stage("drop_constant"), _stage("standardise")]
    departure = {"reason": "the columns come from different instruments",
                 "evidence": ["profile.features.std_ratio_p95_p05"]}
    run = _reported_run(cli, tmp_path, csv_dataset(rows=60, cols=8), {}, base,
                        base_departure=departure)
    block = _block_preprocessing(RunDir(run))
    assert "departs from the rule" in block
    assert "the columns come from different instruments" in block
    assert "median total" not in block


def test_section_2_names_the_features_the_tolerance_dropped(cli, tmp_path):
    import pandas as pd

    from drtools.report import _block_preprocessing
    from drtools.runs import RunDir

    frame = pd.DataFrame(RNG.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    frame["almost"] = 0.1 + np.arange(80) * 1e-17 * 7  # distinct floats, range ~5e-15
    path = tmp_path / "near.csv"
    frame.to_csv(path, index=False)
    run = _reported_run(cli, tmp_path, str(path), {}, [_stage("drop_constant")])
    block = _block_preprocessing(RunDir(run))
    assert "almost" in block and "tolerance" in block
