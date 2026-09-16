import json

import pytest

PCA = [{"op": "pca", "params": {"n_components": 2}}]
# perplexity must stay well below n/3 for the 60-row fixture dataset, or validate-plan
# refuses the candidate outright (see validate_plan's perplexity_too_large finding).
TSNE = [{"op": "tsne", "params": {"n_components": 2, "perplexity": 5}}]


def _plan(candidates, weights={"trustworthiness": 1.0}):
    return {
        "dataset": "d",
        "candidates": candidates,
        "evaluation": {"weights": weights, "justification": "up front"},
    }


def _append_embed_decision(run, candidate_id, outcome):
    """Simulate a later embed attempt ending unsuccessfully.

    The re-registration freeze reads `run.decisions()`, not the embeddings artefact's
    `status` field (an artefact can be edited or deleted; the decision log is
    append-only), so a test exercising the revision rule must add a record here, not
    just rewrite the artefact file.
    """
    with (run / "decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00",
            "stage": "embed",
            "question": f"What did candidate {candidate_id} produce?",
            "options_considered": [],
            "chosen": candidate_id,
            "rationale": "simulated retry for the test",
            "evidence": [],
            "actor": "agent",
            "candidate": candidate_id,
            "outcome": outcome,
        }) + "\n")


def _rewrite_last_embed_outcome(run, candidate_id, outcome):
    """Restate how the last attempt at this candidate ended, without adding one.

    Appending a second record would leave the run reading as two attempts, which is
    the whole allowance, so a test wanting the state after *one* unsuccessful attempt
    has to edit rather than append. That state is real: a candidate can write its
    artefacts and then time out.
    """
    path = run / "decisions.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in reversed(records):
        if record.get("stage") == "embed" and record.get("candidate") == candidate_id:
            record["outcome"] = outcome
            break
    else:  # pragma: no cover - a test that reaches this has mis-set up
        raise AssertionError(f"no embed record for {candidate_id}")
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _prepared(cli, csv_dataset, tmp_path, candidates):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    (runs / "r1" / "plan.json").write_text(json.dumps(_plan(candidates)), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    return runs / "r1"


def test_re_embedding_a_successful_candidate_is_refused(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    result = cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    assert result.code == 2
    assert "already" in result.stderr


def test_rank_ignores_metrics_for_a_candidate_that_did_not_succeed(
    cli, csv_dataset, tmp_path
):
    """The reproduction, in the only shape it can still take.

    A candidate that succeeded can no longer be re-embedded at all, so the original
    route to stale scores is closed upstream. This asserts the belt-and-braces half:
    even with a metrics file sitting there, a candidate whose outcome is not ok is not
    ranked. That is what stopped `rank` reporting a timed-out candidate as a winner.
    """
    run = _prepared(cli, csv_dataset, tmp_path,
                    [{"id": "a", "stages": PCA}, {"id": "b", "stages": TSNE}])
    for candidate in ("a", "b"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
        cli("evaluate", "--run-dir", run, "--id", candidate)

    # Exactly what a timed-out retry used to leave behind: a stale metrics file beside
    # an outcome that is no longer ok.
    record = json.loads((run / "embeddings" / "b.json").read_text(encoding="utf-8"))
    record["status"] = "timeout"
    (run / "embeddings" / "b.json").write_text(json.dumps(record), encoding="utf-8")
    assert (run / "metrics" / "b.json").exists()

    ranked = cli("rank", "--run-dir", run)

    assert [r["id"] for r in ranked.payload["ranking"]] == ["a"]
    assert "b" in ranked.payload["failed_candidates"]


def test_a_retry_clears_what_the_previous_attempt_left(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path,
                    [{"id": "a", "stages": [{"op": "subsample", "params": {"n_samples": 30}},
                                            *PCA]}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    assert (run / "embeddings" / "a.index.npy").exists()

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = PCA
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    # a succeeded, so revising it is refused; a failed attempt is the retry path. The
    # freeze reads the decision log, so the retry needs the recorded outcome changed,
    # not just the artefact file rewritten — and changed rather than added to, or the
    # run reads as two attempts and the retry allowance is already spent.
    record = json.loads((run / "embeddings" / "a.json").read_text(encoding="utf-8"))
    record["status"] = "failed"
    (run / "embeddings" / "a.json").write_text(json.dumps(record), encoding="utf-8")
    _rewrite_last_embed_outcome(run, "a", "failed")

    assert cli("validate-plan", "--run-dir", run).code == 0
    assert cli("embed", "--run-dir", run, "--id", "a", "--in-process").code == 0

    # The subsample index from the first attempt must not survive to subset a reference
    # the second attempt never subsampled.
    assert not (run / "embeddings" / "a.index.npy").exists()


@pytest.mark.parametrize("outcome", ["failed", "timeout", "crashed"])
def test_any_unsuccessful_outcome_may_be_revised(cli, csv_dataset, tmp_path, outcome):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    record = json.loads((run / "embeddings" / "a.json").read_text(encoding="utf-8"))
    record["status"] = outcome
    (run / "embeddings" / "a.json").write_text(json.dumps(record), encoding="utf-8")
    _rewrite_last_embed_outcome(run, "a", outcome)

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = [{"op": "pca", "params": {"n_components": 3}}]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0


def test_a_failed_candidate_can_be_revised_and_retried(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--timeout", "0.01")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = [{"op": "pca", "params": {"n_components": 3}}]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0
    assert cli("embed", "--run-dir", run, "--id", "a", "--in-process").code == 0


def test_a_successful_candidates_stages_cannot_be_revised(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = [{"op": "pca", "params": {"n_components": 3}}]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 2
    assert "succeeded" in result.stderr


def test_the_weighting_can_never_be_revised(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["evaluation"]["weights"] = {"runtime_s": 1.0}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 2
    assert "amendment" in result.stderr.lower()


def test_adding_a_candidate_is_allowed_after_embedding(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"].append({"id": "b", "stages": TSNE})
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0


def _decision_lines(run):
    path = run / "decisions.jsonl"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def test_a_refused_embed_appends_no_decision_lines(cli, csv_dataset, tmp_path):
    """A refused attempt must not look like an attempt.

    `_recorded_outcome` treats any logged embed decision as evidence the candidate ran.
    If a refusal appended a line anyway, an unregistered or unknown candidate would
    read as `crashed` on the next try instead of never having been attempted.
    """
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    before = _decision_lines(run)

    # Unknown candidate id against a registered plan.
    result = cli("embed", "--run-dir", run, "--id", "not-a-candidate", "--in-process")
    assert result.code == 2
    assert _decision_lines(run) == before

    # No registered plan at all.
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8, seed=1),
        "--runs-root", runs, "--run-id", "r2")
    unregistered = runs / "r2"
    before_unregistered = _decision_lines(unregistered)
    result = cli("embed", "--run-dir", unregistered, "--id", "a", "--in-process")
    assert result.code == 2
    assert _decision_lines(unregistered) == before_unregistered


def test_a_third_attempt_on_one_candidate_is_refused(cli, csv_dataset, tmp_path):
    """One run plus one diagnose-and-retry is the whole allowance.

    Nothing counted attempts before this. `_check_reregistration` permits revising a
    candidate that failed any number of times, so a revise-and-retry loop could spend
    unbounded compute while the design claimed a single retry.
    """
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    _append_embed_decision(run, "a", "failed")
    _append_embed_decision(run, "a", "failed")

    result = cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    assert result.code == 2
    assert "two attempts" in result.stderr
    assert "new candidate id" in result.stderr


def test_a_second_attempt_is_still_allowed(cli, csv_dataset, tmp_path):
    """The refusal must bite on the third try, not the second.

    Day 7's lesson was that five of nine fixes reintroduced the class of defect they
    closed. An off-by-one here would remove the retry entirely rather than bound it.
    """
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    _append_embed_decision(run, "a", "failed")

    result = cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    assert result.code == 0, result.stderr


def test_an_interrupted_run_still_offers_the_retry(cli, csv_dataset, tmp_path):
    """A failed candidate has an Outcome, so "lacks an Outcome" would skip it.

    The failure must become permanent because the diagnosis was exhausted, never
    because the process died between the attempt and the retry.
    """
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    _append_embed_decision(run, "a", "failed")

    status = cli("status", "--run-dir", run).payload

    assert status["candidates"]["a"]["outcome"] == "failed"
    assert status["candidates"]["a"]["attempts"] == 1
    assert status["candidates"]["a"]["retry_available"] is True
    assert status["next"] == "execute"


def test_a_candidate_out_of_attempts_is_not_offered_for_execution(
    cli, csv_dataset, tmp_path
):
    """Once the allowance is spent the run moves on rather than stalling on it."""
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("prepare-reference", "--run-dir", run)
    _append_embed_decision(run, "a", "failed")
    _append_embed_decision(run, "a", "failed")

    status = cli("status", "--run-dir", run).payload

    assert status["candidates"]["a"]["attempts"] == 2
    assert status["candidates"]["a"]["retry_available"] is False
    # Nothing left to execute: the one candidate is out of attempts and the reference
    # is prepared, so the run proceeds to evaluation rather than retrying forever.
    assert status["next"] == "evaluate"


def test_an_in_process_failure_is_counted_as_an_attempt(cli, csv_dataset, tmp_path):
    """A route that runs a candidate without recording one is a route that evades.

    The in-process path let `run_pipeline` raise past the decision record, so the
    attempt was never counted: repeating the same failing command bypassed the
    two-attempt allowance while `status` reported no attempts at all.
    """
    run = _prepared(
        cli,
        csv_dataset,
        tmp_path,
        # 60 rows, so asking PCA for 80 components cannot be satisfied.
        [{"id": "a", "stages": [{"op": "pca", "params": {"n_components": 80}}]}],
    )
    first = cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    assert first.payload["status"] == "failed", first.stderr

    status = cli("status", "--run-dir", run).payload
    assert status["candidates"]["a"]["attempts"] == 1
    assert status["candidates"]["a"]["outcome"] == "failed"
    assert status["candidates"]["a"]["retry_available"] is True

    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    third = cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    assert third.code == 2
    assert "two attempts" in third.stderr


def test_revalidating_an_unchanged_plan_does_not_spend_the_replan_round(
    cli, csv_dataset, tmp_path
):
    """Re-registering an identical plan is legal and appends a record either way.

    Testing position alone would let merely revalidating after a ranking consume the
    run's one opportunity to extend the portfolio, and mark a perfectly current
    ranking stale.
    """
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("prepare-reference", "--run-dir", run)
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    cli("evaluate", "--run-dir", run, "--id", "a")
    assert cli("rank", "--run-dir", run).code == 0

    before = cli("status", "--run-dir", run).payload
    assert before["ranked"] is True
    assert before["replan_round_spent"] is False

    # The same plan, registered again. Nothing about the portfolio changed.
    assert cli("validate-plan", "--run-dir", run).code == 0

    after = cli("status", "--run-dir", run).payload
    assert after["replan_round_spent"] is False, "an identical re-registration is not a round"
    assert after["ranked"] is True, "the ranking still describes this plan"
    assert after["next"] == "report"


def test_a_changed_registration_after_ranking_does_spend_the_round(
    cli, csv_dataset, tmp_path
):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("prepare-reference", "--run-dir", run)
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    cli("evaluate", "--run-dir", run, "--id", "a")
    cli("rank", "--run-dir", run)

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"].append(
        {"id": "b", "stages": [{"op": "pca", "params": {"n_components": 3}}]}
    )
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", run).code == 0

    after = cli("status", "--run-dir", run).payload
    assert after["replan_round_spent"] is True
    assert after["ranked"] is False
