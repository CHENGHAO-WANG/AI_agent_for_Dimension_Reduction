"""What `status` reports, and where it reads it from.

Existence comes from artefacts; anything the freeze governs comes from the decision
log, because files can be deleted and the log cannot.
"""

from __future__ import annotations


def test_a_fresh_run_reports_recon_as_next(cli, csv_dataset, tmp_path):
    cli(
        "profile",
        "--data",
        csv_dataset(rows=60, cols=8),
        "--runs-root",
        tmp_path / "runs",
        "--run-id",
        "r1",
    )
    result = cli("status", "--run-dir", tmp_path / "runs" / "r1")
    assert result.code == 0, result.stderr
    assert result.payload["profiled"] is True
    assert result.payload["reconnoitred"] is False
    assert result.payload["registered"] is False
    assert result.payload["candidates"] == {}
    assert result.payload["next"] == "recon"


def test_status_refuses_a_run_that_does_not_exist(cli, tmp_path):
    result = cli("status", "--run-dir", tmp_path / "nope")
    assert result.code == 2
    assert "no run at" in result.stderr


def test_status_has_no_side_effects(cli, csv_dataset, tmp_path):
    run = tmp_path / "runs" / "r1"
    cli(
        "profile",
        "--data",
        csv_dataset(rows=60, cols=8),
        "--runs-root",
        tmp_path / "runs",
        "--run-id",
        "r1",
    )
    before = sorted(str(p.relative_to(run)) for p in run.rglob("*"))
    decisions = run / "decisions.jsonl"
    log_before = decisions.read_text() if decisions.exists() else ""

    cli("status", "--run-dir", run)

    after = sorted(str(p.relative_to(run)) for p in run.rglob("*"))
    log_after = decisions.read_text() if decisions.exists() else ""
    assert before == after
    assert log_before == log_after


def test_status_creates_nothing_in_a_directory_that_is_not_a_run(cli, tmp_path):
    """The previous test could not catch this: a real run already has the directories.

    Opening a RunDir makes `embeddings/`, `metrics/` and `figures/`, which is right for
    a command about to write one and wrong for a command that only reports.
    """
    not_a_run = tmp_path / "somewhere"
    not_a_run.mkdir()

    cli("status", "--run-dir", not_a_run)

    assert list(not_a_run.iterdir()) == []


def test_a_ranking_from_before_the_latest_registration_is_stale(
    cli, csv_dataset, tmp_path
):
    """The re-plan round adds a candidate; nothing clears the previous ranking.

    Reading `ranking.json`'s existence alone would report the run finished as soon as
    the added candidate had metrics, skipping the re-rank that was the whole reason
    for adding it.
    """
    import json

    runs = tmp_path / "runs"
    cli(
        "profile",
        "--data",
        csv_dataset(rows=60, cols=8),
        "--runs-root",
        runs,
        "--run-id",
        "r1",
    )
    run = runs / "r1"
    cli("recon", "--run-dir", run)

    plan = {
        "dataset": "csv",
        "candidates": [
            {"id": "pca2", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}},
    }
    (run / "plan.json").write_text(json.dumps(plan))
    assert cli("validate-plan", "--run-dir", run).code == 0
    assert cli("prepare-reference", "--run-dir", run).code == 0
    assert cli("embed", "--run-dir", run, "--id", "pca2", "--in-process").code == 0
    assert cli("evaluate", "--run-dir", run, "--id", "pca2").code == 0
    assert cli("rank", "--run-dir", run).code == 0

    ranked = cli("status", "--run-dir", run).payload
    assert ranked["ranked"] is True
    assert ranked["next"] == "report"

    # The re-plan round: one added candidate, the weighting untouched.
    plan["candidates"].append(
        {"id": "pca3", "stages": [{"op": "pca", "params": {"n_components": 3}}]}
    )
    (run / "plan.json").write_text(json.dumps(plan))
    assert cli("validate-plan", "--run-dir", run).code == 0

    after_replan = cli("status", "--run-dir", run).payload
    assert after_replan["ranked"] is False, "the ranking predates this registration"
    assert after_replan["next"] == "execute"

    assert cli("embed", "--run-dir", run, "--id", "pca3", "--in-process").code == 0
    assert cli("evaluate", "--run-dir", run, "--id", "pca3").code == 0

    scored = cli("status", "--run-dir", run).payload
    assert scored["ranked"] is False
    assert scored["next"] == "evaluate", "the run must re-rank, not finish"

    assert cli("rank", "--run-dir", run).code == 0
    assert cli("status", "--run-dir", run).payload["next"] == "report"
