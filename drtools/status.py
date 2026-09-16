"""Where a run stands, derived rather than remembered.

`/analyze` resumes by asking this rather than by stat-ing files described in prose, and
every skill's first step reads its position from here. Prose that tells an agent which
files to check decays silently and cannot be tested; a command can be.

Existence comes from artefacts. Anything the freeze governs — a candidate's outcome, how
many attempts it has had, whether the re-plan round is spent — comes from the decision
log, which is append-only and predates every embedding. Files can be deleted; the log is
the authority, as day 7 established for the registration check.
"""

from __future__ import annotations

from typing import Any

from drtools import jsonio
from drtools.runs import RunDir

MAX_ATTEMPTS = 2
"""One run plus the one diagnose-and-retry the design promises."""


def attempts(run: RunDir, candidate_id: str) -> list[dict[str, Any]]:
    """Every recorded attempt at one candidate, oldest first.

    `_recorded_outcome` in the CLI reads the same records but returns only the last
    outcome, so the count has been derivable all along and derived nowhere. Without it
    the one-retry rule is a claim about conversation history rather than about the run.

    An attempt is counted when it ends, not when it begins, and that is deliberate. A
    candidate killed with the parent process — Ctrl-C, a reboot — records nothing and
    gets its allowance back, which is the same reading `_recorded_outcome` already
    takes of an attempt that died mid-flight: recoverable rather than wedged. The
    unbounded-compute worry this leaves is a loop that kills the toolbox mid-candidate
    and retries, and the two ways that happens by itself are both closed elsewhere —
    a child that dies is turned into a `crashed` record by `run_candidate`, and the
    in-process path that could take the interpreter down with it is not reachable from
    a skill. Counting from a start record instead would trade that recoverability for
    a case nothing in the agent loop produces.
    """
    return [
        record
        for record in run.decisions()
        if record.get("stage") == "embed" and record.get("candidate") == candidate_id
    ]


def run_status(run: RunDir) -> dict[str, Any]:
    """Stage-by-stage state of one run, with the stage that should happen next."""
    decisions = run.decisions()
    registered_path = run.path / "plan.registered.json"
    registered = registered_path.exists()
    plan = jsonio.read(registered_path) if registered else None
    manifest = jsonio.read(run.manifest_path) if run.manifest_path.exists() else {}

    abandoned = _abandoned(decisions)
    candidates: dict[str, Any] = {}
    for candidate in (plan or {}).get("candidates", []):
        candidate_id = candidate["id"]
        tries = attempts(run, candidate_id)
        # An attempt that recorded no outcome died before it could write one, which is
        # revisable — the same reading `_recorded_outcome` takes.
        outcome = (tries[-1].get("outcome") or "crashed") if tries else None
        candidates[candidate_id] = {
            "outcome": outcome,
            "attempts": len(tries),
            "retry_available": outcome not in (None, "ok")
            and len(tries) < MAX_ATTEMPTS
            and candidate_id not in abandoned,
            "abandoned": candidate_id in abandoned,
            "metrics": (run.path / "metrics" / f"{candidate_id}.json").exists(),
        }

    state = {
        "run_id": run.id,
        "seed": manifest.get("seed"),
        "budget": (plan or {}).get("budget"),
        "profiled": run.profile_path.exists(),
        "reconnoitred": run.recon_path.exists(),
        "registered": registered,
        "reference_prepared": (run.path / "data" / "reference.json").exists(),
        "ranked": _ranked(run, decisions),
        "replan_round_spent": _replan_spent(decisions),
        "candidates": candidates,
    }
    state["next"] = _next_stage(state)
    return state


def _abandoned(decisions: list[dict[str, Any]]) -> set[str]:
    """Candidates the agent has given up on rather than retried.

    A retry need not reuse the failed candidate's id: the sanctioned repair for a
    method that cannot run as configured is often a differently-shaped attempt under a
    new id, and the old one then stays failed for the record. Counting attempts alone,
    that candidate would keep offering a retry nobody intends to take, and a run
    resuming from `status` would be told to execute forever.

    So giving up is recorded rather than inferred. `execute-plan` writes it when the
    allowance is spent or when it registers a replacement instead of a retry, which is
    the permanent-failure record the design already asks for.
    """
    return {
        record["candidate"]
        for record in decisions
        if record.get("abandoned") and record.get("candidate")
    }


def _ranked(run: RunDir, decisions: list[dict[str, Any]]) -> bool:
    """Whether a ranking exists *for the plan currently registered*.

    The file alone is not enough. `_invalidate_candidate` clears a candidate's own
    embedding and metrics before every attempt, but nothing clears `ranking.json`, so
    a re-plan round that adds a candidate leaves the previous ranking sitting there.
    Reading the file's existence would report the run finished once the added
    candidate had metrics, skipping the re-rank that was the whole reason for adding
    it.

    Position in the log is the wrong test, because re-registering an identical plan is
    legal and appends a record. `rank` already stamps the ranking with the digest of
    the plan it ranked, so ask the question directly: is this ranking the one this
    plan would produce?
    """
    ranking_path = run.path / "ranking.json"
    if not ranking_path.exists():
        return False
    registered = _registered_digests(decisions)
    if not registered:
        # A ranking with no registration recorded predates the freeze; nothing to
        # compare it against, so take the artefact at its word.
        return True
    return jsonio.read(ranking_path).get("plan_digest") == registered[-1]


def _registered_digests(decisions: list[dict[str, Any]]) -> list[str | None]:
    return [
        record.get("plan_digest")
        for record in decisions
        if record.get("stage") == "register_plan"
    ]


def _replan_spent(decisions: list[dict[str, Any]]) -> bool:
    """Whether the one bounded round of extending the portfolio has been used.

    The round is a registration that *adds* candidate ids once something has been
    attempted. Three things have to be told apart, and each of the simpler tests gets
    one of them wrong:

    Re-registering an identical plan is not a round. `validate-plan` permits it and
    appends a record either way, so counting registrations would let merely
    revalidating consume the run's single opportunity.

    Revising a candidate that failed is not a round either — it is the diagnose-and-
    retry, and it changes the plan's digest. So a digest comparison would spend the
    round on an ordinary retry. Only a grown set of ids is an extended portfolio.

    And anchoring on a ranking, which an earlier version did, cannot work: ranking
    needs metrics from a candidate that succeeded, so a run where everything failed
    would never record one and would be offered the round forever.
    """
    attempted = False
    previous_ids: set[str] | None = None
    for record in decisions:
        stage = record.get("stage")
        if stage == "embed":
            attempted = True
        elif stage == "register_plan":
            ids = set(record.get("candidates") or [])
            if attempted and previous_ids is not None and ids > previous_ids:
                return True
            previous_ids = ids
    return False


def _next_stage(state: dict[str, Any]) -> str:
    """The first stage that has work outstanding."""
    if not state["profiled"]:
        return "profile"
    if not state["registered"]:
        # Reconnaissance is evidence for planning, and `validate-plan` accepts a plan
        # without it. So it is only outstanding while a plan is still to be written:
        # naming it as the next stage of a run that is already executing would send
        # the agent back to re-probe evidence its registered plan was argued from.
        return "recon" if not state["reconnoitred"] else "plan"

    outstanding = [
        candidate_id
        for candidate_id, record in state["candidates"].items()
        # Never run, or failed with the retry still owed. Reading "has an outcome"
        # alone would make a failure permanent because a process died rather than
        # because the diagnosis was exhausted.
        if record["outcome"] is None
        or (record["outcome"] != "ok" and record["retry_available"])
    ]
    if outstanding or not state["reference_prepared"]:
        return "execute"

    succeeded = [
        candidate_id
        for candidate_id, record in state["candidates"].items()
        if record["outcome"] == "ok"
    ]
    if not succeeded:
        # Nothing to rank, and `rank` raises rather than producing an empty ranking.
        # Sending the run there would park it on a stage that cannot complete. While
        # the re-plan round is unspent there is a real move available — register
        # candidates that can run under this budget — and once it is spent, a run
        # where every candidate failed is a finding the report should carry rather
        # than an error to loop on.
        return "plan" if not state["replan_round_spent"] else "report"

    scorable = [
        candidate_id
        for candidate_id in succeeded
        if not state["candidates"][candidate_id]["metrics"]
    ]
    if scorable or not state["ranked"]:
        return "evaluate"
    return "report"
