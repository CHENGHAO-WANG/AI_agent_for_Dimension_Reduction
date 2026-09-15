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
