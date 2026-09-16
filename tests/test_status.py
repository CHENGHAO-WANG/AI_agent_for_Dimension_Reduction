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


def _register(cli, run, candidates, budget="standard"):
    import json

    plan = {
        "dataset": "d",
        "budget": budget,
        "candidates": candidates,
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return cli("validate-plan", "--run-dir", run)


def _mark(run, candidate_id, outcome, abandoned=False):
    """Append one embed record, optionally marking the candidate given up on."""
    import json

    record = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "stage": "embed",
        "question": f"What did candidate {candidate_id} produce?",
        "options_considered": [],
        "chosen": candidate_id,
        "rationale": "recorded by the test",
        "evidence": [],
        "actor": "agent",
        "candidate": candidate_id,
        "outcome": outcome,
    }
    if abandoned:
        record["abandoned"] = True
    with (run / "decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_a_candidate_given_up_on_stops_offering_a_retry(cli, csv_dataset, tmp_path):
    """A retry need not reuse the failed id, and often should not.

    Counting attempts alone, a candidate replaced by a new id keeps offering a retry
    nobody intends to take, and a run resuming from status is told to execute forever.
    """
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    assert _register(cli, run, [{"id": "a", "stages": [{"op": "pca", "params": {"n_components": 2}}]}]).code == 0
    cli("prepare-reference", "--run-dir", run)

    _mark(run, "a", "failed")
    assert cli("status", "--run-dir", run).payload["candidates"]["a"]["retry_available"] is True
    assert cli("status", "--run-dir", run).payload["next"] == "execute"

    _mark(run, "a", "failed", abandoned=True)

    after = cli("status", "--run-dir", run).payload
    assert after["candidates"]["a"]["abandoned"] is True
    assert after["candidates"]["a"]["retry_available"] is False
    assert after["next"] != "execute"


def test_a_run_where_everything_failed_is_not_sent_to_rank(cli, csv_dataset, tmp_path):
    """`rank` raises with no successful metrics, so routing there parks the run on a
    stage that cannot complete."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    assert _register(cli, run, [{"id": "a", "stages": [{"op": "pca", "params": {"n_components": 2}}]}]).code == 0
    cli("prepare-reference", "--run-dir", run)

    _mark(run, "a", "failed")
    _mark(run, "a", "failed")

    status = cli("status", "--run-dir", run).payload
    assert status["candidates"]["a"]["retry_available"] is False
    assert status["next"] == "plan", "the re-plan round is the available move"

    # And once the round is spent, the run reports rather than looping.
    assert cli("rank", "--run-dir", run).code != 0, "rank genuinely cannot run here"


def test_a_refused_re_registration_writes_nothing(cli, csv_dataset, tmp_path):
    """A refusal that has already overwritten plan_validation.json has left the run
    describing a plan it never registered."""
    import json

    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    candidates = [{"id": "a", "stages": [{"op": "pca", "params": {"n_components": 2}}]}]
    assert _register(cli, run, candidates, budget="fast").code == 0

    # A sentinel rather than a copy of the real report: the refused plan differs only
    # in its budget, which does not change what the validator finds, so comparing
    # content would pass even though the file had been rewritten.
    validation = run / "plan_validation.json"
    validation.write_text('{"sentinel": true}', encoding="utf-8")
    log_before = (run / "decisions.jsonl").read_text(encoding="utf-8")

    assert _register(cli, run, candidates, budget="thorough").code == 2

    assert validation.read_text(encoding="utf-8") == '{"sentinel": true}'
    assert (run / "decisions.jsonl").read_text(encoding="utf-8") == log_before
