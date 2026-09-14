import json


def _plan(base, candidates):
    return {
        "dataset": "d",
        "base_preprocessing": base,
        "candidates": candidates,
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "declared"},
    }


PCA = [{"op": "pca", "params": {"n_components": 2}}]
STANDARDISE = [{"op": "standardise", "params": {}}]


def test_prepare_reference_takes_base_stages_from_the_plan(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = _plan(STANDARDISE, [{"id": "a", "stages": PCA}])
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")

    result = cli("prepare-reference", "--run-dir", runs / "r1")

    assert result.code == 0
    assert result.payload["stages"] == [{"op": "standardise", "params": {}}]


def test_reference_records_the_settings_it_fixes(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1", "--seed", 4)
    plan = _plan(STANDARDISE, [{"id": "a", "stages": PCA}])
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("prepare-reference", "--run-dir", runs / "r1")

    recorded = json.loads((runs / "r1" / "data" / "reference.json").read_text(encoding="utf-8"))
    assert recorded["settings"] == {"k": 15, "max_samples": 2000, "seed": 4}
    assert recorded["n_rows"] == 60


def test_evaluate_refuses_when_the_plan_declares_a_base_and_none_was_prepared(
    cli, csv_dataset, tmp_path
):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = _plan(STANDARDISE, [{"id": "a", "stages": PCA}])
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    # The candidate's registered stages are the base plus its own, since prepare-reference
    # (below) has not run yet to give `embed` anything else to key on.
    cli("embed", "--run-dir", runs / "r1", "--id", "a",
        "--stages", json.dumps(STANDARDISE + PCA), "--in-process")

    result = cli("evaluate", "--run-dir", runs / "r1", "--id", "a")

    assert result.code == 2
    assert "prepare-reference" in result.stderr


def test_evaluate_needs_no_reference_when_the_plan_declares_no_base(
    cli, csv_dataset, tmp_path
):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    (runs / "r1" / "plan.json").write_text(
        json.dumps(_plan([], [{"id": "a", "stages": PCA}])), encoding="utf-8"
    )
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("embed", "--run-dir", runs / "r1", "--id", "a",
        "--stages", json.dumps(PCA), "--in-process")

    result = cli("evaluate", "--run-dir", runs / "r1", "--id", "a")
    assert result.code == 0


def test_evaluate_rejects_a_neighbourhood_argument(cli, tmp_path):
    result = cli("evaluate", "--run-dir", tmp_path, "--id", "a", "--k", "5")
    assert result.code != 0
