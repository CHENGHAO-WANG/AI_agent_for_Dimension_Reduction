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
            and len(tries) < MAX_ATTEMPTS,
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
        "ranked": (run.path / "ranking.json").exists(),
        "replan_round_spent": _replan_spent(decisions),
        "candidates": candidates,
    }
    state["next"] = _next_stage(state)
    return state


def _replan_spent(decisions: list[dict[str, Any]]) -> bool:
    """A registration recorded after a ranking is the one re-plan round, spent.

    The log is append-only and ordered, so position is the whole test. Before any
    ranking exists a re-registration is an ordinary diagnose-and-retry, not a round:
    the round is defined by happening after results, which is what makes it bounded.
    """
    stages = [record.get("stage") for record in decisions]
    if "rank" not in stages:
        return False
    first_rank = stages.index("rank")
    return "register_plan" in stages[first_rank + 1 :]


def _next_stage(state: dict[str, Any]) -> str:
    """The first stage that has work outstanding."""
    if not state["profiled"]:
        return "profile"
    if not state["reconnoitred"]:
        return "recon"
    if not state["registered"]:
        return "plan"

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

    scorable = [
        candidate_id
        for candidate_id, record in state["candidates"].items()
        if record["outcome"] == "ok" and not record["metrics"]
    ]
    if scorable or not state["ranked"]:
        return "evaluate"
    return "report"
