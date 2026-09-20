"""What `status` reports, and where it reads it from.

Existence comes from artefacts; anything the freeze governs comes from the decision
log, because files can be deleted and the log cannot.
"""

from __future__ import annotations

import json


def _registered_run(cli, csv_dataset, tmp_path):
    """A run with one registered PCA candidate."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    (run / "plan.json").write_text(json.dumps({
        "dataset": "d",
        "candidates": [
            {"id": "pca2", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }), encoding="utf-8")
    cli("validate-plan", "--run-dir", run)
    return run


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


def test_an_all_failed_run_does_not_offer_the_replan_round_forever(
    cli, csv_dataset, tmp_path
):
    """Ranking needs a candidate that succeeded, so an all-failed run never records
    one. Anchoring the round on a ranking left such a run offered `plan` for ever."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    pca2 = [{"op": "pca", "params": {"n_components": 2}}]
    pca3 = [{"op": "pca", "params": {"n_components": 3}}]

    assert _register(cli, run, [{"id": "a", "stages": pca2}]).code == 0
    cli("prepare-reference", "--run-dir", run)
    _mark(run, "a", "failed")
    _mark(run, "a", "failed")
    assert cli("status", "--run-dir", run).payload["next"] == "plan"

    # The round: one added candidate. It fails too.
    assert _register(cli, run, [{"id": "a", "stages": pca2}, {"id": "b", "stages": pca3}]).code == 0
    _mark(run, "b", "failed")
    _mark(run, "b", "failed")

    status = cli("status", "--run-dir", run).payload
    assert status["replan_round_spent"] is True
    assert status["next"] == "report", "the round is spent; every candidate failing is a finding"


def test_revising_a_failed_candidate_does_not_spend_the_replan_round(
    cli, csv_dataset, tmp_path
):
    """A retry changes the plan's digest without extending the portfolio.

    A digest comparison would spend the round on the ordinary diagnose-and-retry,
    which is the opposite of what the round is for.
    """
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    assert _register(cli, run, [{"id": "a", "stages": [{"op": "pca", "params": {"n_components": 2}}]}]).code == 0
    _mark(run, "a", "failed")
    assert _register(cli, run, [{"id": "a", "stages": [{"op": "pca", "params": {"n_components": 3}}]}]).code == 0

    assert cli("status", "--run-dir", run).payload["replan_round_spent"] is False


def test_adding_a_candidate_before_anything_ran_is_not_the_round(
    cli, csv_dataset, tmp_path
):
    """Planning iteration before execution is not the post-results round."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    pca2 = [{"op": "pca", "params": {"n_components": 2}}]
    pca3 = [{"op": "pca", "params": {"n_components": 3}}]

    assert _register(cli, run, [{"id": "a", "stages": pca2}]).code == 0
    assert _register(cli, run, [{"id": "a", "stages": pca2}, {"id": "b", "stages": pca3}]).code == 0

    assert cli("status", "--run-dir", run).payload["replan_round_spent"] is False


def _give_up(cli, run, tmp_path, candidate_id):
    """Abandon a candidate through the toolbox, as execute-plan does."""
    import json

    document = tmp_path / f"give-up-{candidate_id}.json"
    document.write_text(json.dumps({
        "stage": "execute",
        "question": f"Retry candidate {candidate_id}, or replace it?",
        "chosen": "replace it",
        "rationale": "the precondition fails, not the parameters, so the same stages "
        "would fail the same way",
        "evidence": [],
        "candidate": candidate_id,
        "abandoned": True,
    }), encoding="utf-8")
    return cli("log-decision", "--run-dir", run, "--json", f"@{document}")


def test_replacing_a_failed_candidate_does_not_spend_the_replan_round(
    cli, csv_dataset, tmp_path
):
    """A replacement id is the retry, not the round.

    The two allowances are separate: one diagnose-and-retry per candidate, and one
    post-evaluation round of extending the portfolio. Counting id growth alone spent
    the round on the retry, so later evaluation would wrongly report no round left.
    """
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    pca2 = [{"op": "pca", "params": {"n_components": 2}}]
    pca3 = [{"op": "pca", "params": {"n_components": 3}}]

    assert _register(cli, run, [{"id": "a", "stages": pca2}]).code == 0
    _mark(run, "a", "failed")
    assert _give_up(cli, run, tmp_path, "a").code == 0

    # The replacement, registered under a new id.
    assert _register(cli, run, [{"id": "a", "stages": pca2}, {"id": "b", "stages": pca3}]).code == 0

    status = cli("status", "--run-dir", run).payload
    assert status["replan_round_spent"] is False, "replacing is retrying, not extending"


def test_extending_beyond_a_replacement_does_spend_the_round(cli, csv_dataset, tmp_path):
    """Growth net of replacements is the round. Two added for one abandoned is
    extension."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    pca2 = [{"op": "pca", "params": {"n_components": 2}}]
    pca3 = [{"op": "pca", "params": {"n_components": 3}}]
    pca4 = [{"op": "pca", "params": {"n_components": 4}}]

    assert _register(cli, run, [{"id": "a", "stages": pca2}]).code == 0
    _mark(run, "a", "failed")
    assert _give_up(cli, run, tmp_path, "a").code == 0

    assert _register(cli, run, [
        {"id": "a", "stages": pca2},
        {"id": "b", "stages": pca3},
        {"id": "c", "stages": pca4},
    ]).code == 0

    assert cli("status", "--run-dir", run).payload["replan_round_spent"] is True


def test_the_candidate_set_comes_from_the_log_not_the_registered_file(
    cli, csv_dataset, tmp_path
):
    """A rewritten plan file must not redirect the agent.

    Day 7 made the log the authority for the freeze because a file can be rewritten,
    and `rank` cross-checks the two. `status` decides what the agent does next, so
    enumerating candidates from the file left the one command that steers the run
    trusting the one artefact the freeze does not.
    """
    run = _registered_run(cli, csv_dataset, tmp_path)

    registered = json.loads((run / "plan.registered.json").read_text(encoding="utf-8"))
    registered["candidates"].append(
        {"id": "ghost", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
    )
    (run / "plan.registered.json").write_text(json.dumps(registered), encoding="utf-8")

    status = cli("status", "--run-dir", run).payload

    assert "ghost" not in status["candidates"]
    assert "pca2" in status["candidates"]


def test_a_ranking_with_no_registration_recorded_is_not_ranked(
    cli, csv_dataset, tmp_path
):
    """The toolbox cannot produce that state, so reading it as ranked fails open.

    `rank` refuses unless the log records a registration, so a ranking.json with no
    `register_plan` record behind it was not written by this toolbox. Taking it at its
    word would let a hand-written file declare the run finished.
    """
    run = _registered_run(cli, csv_dataset, tmp_path)
    (run / "decisions.jsonl").write_text("", encoding="utf-8")
    (run / "ranking.json").write_text(
        json.dumps({"winner": "pca2", "plan_digest": "whatever"}), encoding="utf-8"
    )

    assert cli("status", "--run-dir", run).payload["ranked"] is False
