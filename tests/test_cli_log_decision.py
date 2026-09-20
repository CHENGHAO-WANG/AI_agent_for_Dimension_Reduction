"""The one route into the decision log, and what it refuses.

Checking a key at write time and reading its value at report time are different
guarantees unless the log carries the value it saw. Artefacts are rewritten in place,
so a key can resolve to one value when the decision is logged and to another by the
time the report reads it.
"""

from __future__ import annotations

import json


def _decision(**overrides):
    record = {
        "stage": "plan",
        "question": "What base preprocessing should every candidate share?",
        "chosen": "none",
        "rationale": "values are already on a comparable scale",
        "evidence": ["profile.shape.n_samples"],
        "options_considered": ["none", "standardise"],
    }
    record.update(overrides)
    return record


def _write(tmp_path, document):
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return f"@{path}"


def _lines(run):
    path = run / "decisions.jsonl"
    if not path.exists():
        # `profile` logs nothing, so a run that has only been profiled has no log at
        # all — which is itself the strongest form of "the refusal wrote nothing".
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _only(run, stage):
    """The single record at this stage. Other commands append records of their own."""
    matching = [record for record in _lines(run) if record["stage"] == stage]
    assert len(matching) == 1, f"expected one {stage} record, found {len(matching)}"
    return matching[0]


def test_a_decision_is_appended_with_what_its_keys_resolved_to(
    cli, csv_dataset, tmp_path
):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli("log-decision", "--run-dir", run, "--json", _write(tmp_path, _decision()))

    assert result.code == 0, result.stderr
    written = _lines(run)[-1]
    assert written["chosen"] == "none"
    assert written["stage"] == "plan"
    assert written["evidence_resolved"]["profile.shape.n_samples"] == 60


def test_a_key_that_does_not_resolve_is_refused(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    before = _lines(run)

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(evidence=["profile.shape.n_cells"])),
    )

    assert result.code == 2
    assert "profile.shape.n_cells" in result.stderr
    # The refusal names the keys that do exist, because an agent told only that
    # something failed will guess again.
    assert "n_samples" in result.stderr
    assert _lines(run) == before, "a refusal writes nothing"


def test_an_empty_evidence_list_is_allowed(cli, csv_dataset, tmp_path):
    """Claiming no support is not the same as claiming false support."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(evidence=[])),
    )

    assert result.code == 0, result.stderr
    assert _lines(run)[-1]["evidence"] == []


def test_a_decision_missing_its_reasoning_is_refused(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(rationale="")),
    )

    assert result.code == 2
    assert "rationale" in result.stderr


def test_a_regenerated_artefact_leaves_the_logged_reading_intact(
    cli, csv_dataset, tmp_path
):
    """The regression that lets the report generator detect a rebound rationale.

    `recon` still carries its own --k and --max-samples, which day 9 owns. Until then
    a cited reconnaissance value can move under a decision that cited it.
    """
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    cli("recon", "--run-dir", run, "--max-samples", 50)
    at_write_artefact = json.loads((run / "recon.json").read_text())["subsample"]["n_used"]

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(evidence=["recon.subsample.n_used"])),
    )
    assert result.code == 0, result.stderr
    logged = _only(run, "plan")["evidence_resolved"]["recon.subsample.n_used"]
    assert logged == at_write_artefact

    cli("recon", "--run-dir", run, "--max-samples", 30)
    now = json.loads((run / "recon.json").read_text())["subsample"]["n_used"]

    assert logged != now, "the artefact moved under the citation"
    assert _only(run, "plan")["evidence_resolved"]["recon.subsample.n_used"] == logged


def test_extra_fields_reach_the_record(cli, csv_dataset, tmp_path):
    """`abandoned` is the one status reads, so the route that writes it is the
    production path for giving up on a candidate."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(
            stage="execute", candidate="c1", abandoned=True, evidence=[],
        )),
    )

    assert result.code == 0, result.stderr
    written = _lines(run)[-1]
    assert written["candidate"] == "c1"
    assert written["abandoned"] is True


def test_giving_up_on_a_candidate_goes_through_the_toolbox(cli, csv_dataset, tmp_path):
    """End to end: the field status depends on has a route the agent can reach.

    Without this, `abandoned` was a field only the test suite wrote, and a candidate
    replaced rather than retried would leave the run routed to `execute` for ever.
    """
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"
    plan = {
        "dataset": "d",
        "candidates": [
            {"id": "a", "stages": [{"op": "pca", "params": {"n_components": 2}}]},
            {"id": "b", "stages": [{"op": "pca", "params": {"n_components": 3}}]},
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", run).code == 0
    cli("prepare-reference", "--run-dir", run)
    cli("embed", "--run-dir", run, "--id", "b", "--in-process")

    # `a` fails once, and the agent decides to replace rather than retry it.
    with (run / "decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00", "stage": "embed",
            "question": "What did candidate a produce?", "options_considered": [],
            "chosen": "a", "rationale": "recorded by the test", "evidence": [],
            "actor": "agent", "candidate": "a", "outcome": "failed",
        }) + "\n")
    assert cli("status", "--run-dir", run).payload["next"] == "execute"

    assert cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, {
            "stage": "execute",
            "question": "Retry candidate a, or replace it?",
            "chosen": "replace it; a is abandoned",
            "rationale": "the failure is in the method's precondition, not its "
            "parameters, so the same stages would fail the same way",
            "evidence": ["plan.registered.candidates"],
            "candidate": "a",
            "abandoned": True,
        }),
    ).code == 0

    status = cli("status", "--run-dir", run).payload
    assert status["candidates"]["a"]["abandoned"] is True
    assert status["candidates"]["a"]["retry_available"] is False
    assert status["next"] == "evaluate", "the run proceeds with what it has"


import pytest


@pytest.mark.parametrize(
    "stage", ["register_plan", "embed", "rank", "validate_plan", "recon"]
)
def test_lifecycle_stages_cannot_be_written_by_hand(cli, csv_dataset, tmp_path, stage):
    """The freeze anchors on these records, so a generic route must not emit them.

    `rank` cross-checks registration against the log, attempts are counted from it,
    and status derives the whole state machine from it. A forged register_plan record
    would let a plan be "registered" without ever being validated -- the
    pre-registration guarantee defeated by the mechanism built to record it.
    """
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(stage=stage, evidence=[])),
    )

    assert result.code == 2
    assert stage in result.stderr
    assert _lines(run) == []


@pytest.mark.parametrize(
    "field,value",
    [("plan_digest", "deadbeef"), ("outcome", "ok"), ("weights", {"t": 1.0}),
     ("candidates", ["a", "b"]), ("max_candidates", 7)],
)
def test_lifecycle_fields_cannot_be_written_by_hand(
    cli, csv_dataset, tmp_path, field, value
):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, _decision(evidence=[], **{field: value})),
    )

    assert result.code == 2
    assert field in result.stderr
    assert _lines(run) == []


def test_a_forged_registration_cannot_satisfy_rank(cli, csv_dataset, tmp_path):
    """The reproduction, end to end: the route must not be able to fake the freeze."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    run = runs / "r1"

    result = cli(
        "log-decision", "--run-dir", run,
        "--json", _write(tmp_path, {
            "stage": "register_plan",
            "question": "What will this run compare?",
            "chosen": "registered 2 candidates",
            "rationale": "asserting a registration that never happened",
            "evidence": [],
        }),
    )

    assert result.code == 2
    assert not (run / "plan.registered.json").exists()
    assert cli("status", "--run-dir", run).payload["registered"] is False
