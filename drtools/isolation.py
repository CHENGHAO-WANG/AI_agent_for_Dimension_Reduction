"""Running a candidate in a process that can be killed.

Three things go wrong when dimension reduction meets real data, and only one of them is
an exception. A method can raise, which is catchable. It can allocate more than the OS
will give and take the interpreter down, which is not. Or it can simply not finish —
SMACOF on fifty thousand points does not fail, it runs until someone stops it.

A wall-clock cap on a subprocess handles all three the same way, and turns each into a
record the agent can read and act on. Which matters more than tidiness: "Isomap timed
out after 300s at n=40,000" is a fact the planner can use to revise, where a hung
process is just an analysis that never finished.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from drtools import jsonio
from drtools.runs import RunDir

DEFAULT_TIMEOUT_S = 600.0
STDERR_TAIL_LINES = 8

BUDGET_TIMEOUTS_S = {"fast": 120.0, "standard": 600.0, "thorough": 3600.0}
"""Per-candidate wall-clock caps. `standard` keeps what the default has always been."""


def budget_timeout(budget: str) -> float:
    """The wall-clock one candidate may spend under this Budget.

    The mapping lives here rather than in skill prose so that the Budget reaches the
    command that spends it. A table in a skill would be a number the agent retypes,
    and a number the agent retypes is a number the agent can get wrong.
    """
    return BUDGET_TIMEOUTS_S[budget]


BUDGET_MAX_CANDIDATES = {"fast": 8, "standard": 7, "thorough": 6}
"""How many Candidates one Run may ever register, by Budget.

Attempts are capped per candidate id and the id set only grows, so this is what
bounds a Run: at most `MAX_ATTEMPTS x ceiling` executions. Without it the id set is
the one spendable quantity nothing declares, and it is precisely what an agent mints
to reset the per-candidate allowance — exhaust two attempts, abandon, register a
replacement, repeat.

The ceiling shrinks as `BUDGET_TIMEOUTS_S` grows, which states the trade rather than
hiding it: a larger time allowance per candidate buys fewer of them. It sits strictly
above section 3.4's 3-5 portfolio guidance, and that gap is the repair headroom — a
ceiling of 5 would bound the loop by forbidding the sanctioned retry instead.
"""


def max_candidates(budget: str) -> int:
    """The size of the Plan one Run may register under this Budget."""
    return BUDGET_MAX_CANDIDATES[budget]

# Signals worth naming, because the remedy differs. A process killed by the OS for
# memory needs a smaller problem; one that aborted needs a different method.
KILLED_BY_MEMORY = {137, -9}


def run_candidate(
    run: RunDir,
    candidate_id: str,
    stages: list[dict[str, Any]],
    *,
    seed: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    python: str | None = None,
) -> dict[str, Any]:
    """Execute one candidate in a subprocess, returning its outcome record either way."""
    jobs = run.path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_path = jobs / f"{candidate_id}.json"
    jsonio.write(
        job_path,
        {
            "id": candidate_id,
            "run_dir": str(run.path),
            "stages": stages,
            "seed": seed,
        },
    )

    record_path = run.path / "embeddings" / f"{candidate_id}.json"
    record_path.unlink(missing_ok=True)

    command = [python or sys.executable, "-m", "drtools._worker", str(job_path)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _write_outcome(
            record_path,
            {
                "id": candidate_id,
                "status": "timeout",
                "duration_s": round(time.perf_counter() - started, 3),
                "stages_requested": stages,
                "failure": {
                    "op": None,
                    "error_type": "Timeout",
                    "message": f"exceeded the {timeout_s:g}s wall-clock budget and was "
                    "stopped. The method did not fail; it did not finish. Reduce the "
                    "sample count, choose a method that scales, or raise the budget if "
                    "this candidate is worth the time.",
                    "timeout_s": timeout_s,
                },
            },
        )

    duration = round(time.perf_counter() - started, 3)

    if record_path.exists():
        # The worker recorded its own outcome, success or handled failure.
        record = jsonio.read(record_path)
        record.setdefault("wall_clock_s", duration)
        return record

    # No record means the process died before it could write one.
    return _write_outcome(
        record_path,
        {
            "id": candidate_id,
            "status": "crashed",
            "duration_s": duration,
            "stages_requested": stages,
            "failure": {
                "op": None,
                "error_type": "ProcessDied",
                "message": _describe_death(completed.returncode),
                "exit_code": completed.returncode,
                "stderr_tail": _tail(completed.stderr),
            },
        },
    )


def _describe_death(returncode: int) -> str:
    if returncode in KILLED_BY_MEMORY:
        return (
            "the process was killed by the operating system, which on this exit status "
            "almost always means it asked for more memory than was available. Subsample "
            "before this method, or choose one whose memory is not quadratic in n."
        )
    return (
        f"the process exited with status {returncode} without recording a result, so it "
        "died rather than failing cleanly. This is usually an allocation the allocator "
        "refused or a crash inside a native library."
    )


def _tail(text: str | None) -> list[str]:
    if not text:
        return []
    return text.strip().splitlines()[-STDERR_TAIL_LINES:]


def _write_outcome(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    jsonio.write(path, record)
    return record
