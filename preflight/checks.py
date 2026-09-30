"""What the pre-flight asserts, over a transcript and over a Run directory.

Each check returns one line per violation, so an empty list means it holds, and a
failing scenario says everything that went wrong at once rather than the first thing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from preflight.transcript import Call, Transcript

#: The commands that can take minutes. Two in one call can outlast the tool's 600 s
#: timeout, and the harness then moves the call to the background (day 21).
SLOW = re.compile(r"\bdrtools\s+(embed|evaluate|prepare-reference|recon|render)\b")
#: A shell loop's body opens with `do` after a `;` or a newline; a Python comprehension
#: inside `python -c` has `for x in` but no `do`.
LOOP = re.compile(r"[;\n]\s*do\b")
#: The fields of each reading tool that name what it reads. A Grep's pattern or a Bash
#: call's description is text, not a destination.
READ_FIELDS = {"Read": ("file_path",), "Glob": ("path", "pattern"),
               "Grep": ("path", "glob"), "Bash": ("command",)}
#: `runs/<name>`, where a wildcard name counts as another Run once the path descends
#: into it: `ls runs/` finds a Run to resume, `runs/*/plan.json` reads them all.
RUN_PATH = re.compile(r"runs/([\w.*-]+)(/?)")


def _show(call: Call) -> str:
    return f"{call.name}: {json.dumps(call.input)[:200]}"


def _norm(text: str) -> str:
    return text.replace("\\\\", "/").replace("\\", "/").lower()


def _repo_forms(repo: Path) -> list[str]:
    posix = repo.as_posix().lower()  # d:/pythonproject/...
    drive, _, rest = posix.partition(":")
    return [posix, f"/{drive}{rest}"]  # and git bash's /d/pythonproject/...


def background(t: Transcript) -> list[str]:
    return [_show(c) for c in t.calls
            if c.input.get("run_in_background") or "moved to the background" in c.result]


def chained_slow(t: Transcript) -> list[str]:
    found = []
    for c in t.calls:
        if c.name != "Bash":
            continue
        command = c.input.get("command", "")
        slow = len(SLOW.findall(command))
        if slow > 1 or (slow == 1 and LOOP.search(command)):
            found.append(_show(c))
    return found


def foreign_reads(t: Transcript, run_id: str, repo: Path) -> list[str]:
    forms = _repo_forms(repo)
    found = []
    for c in t.calls:
        if c.name not in READ_FIELDS:
            continue
        text = _norm(" ".join(str(c.input.get(f, "")) for f in READ_FIELDS[c.name]))
        others = {name for name, slash in RUN_PATH.findall(text)
                  if name != run_id.lower() and ("*" not in name or slash)}
        if any(form in text for form in forms) or others:
            found.append(_show(c))
    return found


def stray_writes(t: Transcript, run_id: str) -> list[str]:
    own = f"/runs/{run_id.lower()}/"
    found = []
    for c in t.calls:
        if c.name not in ("Write", "Edit"):
            continue
        path = _norm(c.input.get("file_path", ""))
        allowed = (f"{own}inputs/" in path or path.endswith(f"{own}plan.json")
                   or path.endswith(f"{own}results/report.md"))
        if not allowed:
            found.append(_show(c))
    return found


STAGE = re.compile(r"\bdrtools\s+(profile|recon|checkpoint|suggest-base|validate-plan|"
                   r"prepare-reference|embed|evaluate|rank|compare|recommend|figures|"
                   r"report|render)\b")


def _decisions(run_dir: Path) -> list[dict]:
    path = run_dir / "decisions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finished(run_dir: Path) -> list[str]:
    from drtools.runs import RunDir
    from drtools.status import run_status

    found = []
    if not run_dir.exists():
        return [f"no Run at {run_dir}"]
    stage = run_status(RunDir(run_dir, create=False))["next"]
    if stage != "done":
        found.append(f"status says the next stage is {stage}, not done")
    if not (run_dir / "results" / "report.pdf").exists():
        found.append("no results/report.pdf")
    return found


def representation(run_dir: Path, summary: dict, transcripts: list[Transcript]) -> list[str]:
    from drtools.cli import LIFECYCLE_STAGES
    from drtools.decision import recorded_decision
    from drtools.plan import CandidateSpec, is_linear_baseline, is_selected_baseline

    found = []
    recon_path = run_dir / "recon.json"
    recon = json.loads(recon_path.read_text(encoding="utf-8")) if recon_path.exists() else None
    decision = recorded_decision(recon)
    if decision is None or decision.values != "raw_counts":
        found.append("the data decision does not say raw_counts")
    stated = [r for r in _decisions(run_dir) if r.get("stage") not in LIFECYCLE_STAGES
              and re.search(r"count", json.dumps(r), re.I) and re.search(r"\blog", json.dumps(r), re.I)]
    if not stated:
        found.append("no agent-written decision states the counts transform")
    plan_path = run_dir / "plan.registered.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    candidates = [CandidateSpec.model_validate(c) for c in plan.get("candidates", [])]
    if not any(is_linear_baseline(c) for c in candidates):
        found.append("the registered Plan has no Linear baseline")
    if not any(is_selected_baseline(c) for c in candidates):
        found.append("the registered Plan has no Selected baseline")
    if not (run_dir / "ranking.json").exists():
        found.append("no ranking.json")
    return found


def resume(run_dir: Path, summary: dict, transcripts: list[Transcript]) -> list[str]:
    from drtools.runs import RunDir
    from drtools.status import attempts

    launches = summary.get("launches", [])
    if len(launches) != 2 or len(transcripts) != 2:
        return ["resume needs exactly two launches"]
    if launches[0].get("next") == "done":
        return ["the first launch finished inside the turn limit, so resumption was tested nothing"]
    found = []
    decisions = _decisions(run_dir)
    registrations = [r for r in decisions if r.get("stage") == "register_plan"]
    if len(registrations) != 1:
        found.append(f"{len(registrations)} registrations, not 1")
    run = RunDir(run_dir, create=False)
    for candidate in (registrations[-1].get("candidates") or []) if registrations else []:
        tries = attempts(run, candidate)
        if tries and tries[0].get("outcome") == "ok" and len(tries) > 1:
            found.append(f"{candidate} succeeded first time and ran again")
    # Positions as (call, offset), so two commands in one call are ordered too.
    bash = [c.input.get("command", "") for c in transcripts[1].calls if c.name == "Bash"]
    status_at = next(((i, m.start()) for i, cmd in enumerate(bash)
                      for m in re.finditer(r"\bdrtools\s+status\b", cmd)), None)
    stage_at = next(((i, m.start()) for i, cmd in enumerate(bash)
                     for m in STAGE.finditer(cmd)), None)
    if status_at is None or (stage_at is not None and stage_at < status_at):
        found.append("the second launch ran a stage before reading status")
    return found


def visualization(run_dir: Path, summary: dict, transcripts: list[Transcript]) -> list[str]:
    found = []
    if (run_dir / "ranking.json").exists():
        found.append("a Visualization run has a ranking.json")
    if not (run_dir / "recommendation.json").exists():
        found.append("no recommendation.json")
    if any(r.get("stage") == "adopt" for r in _decisions(run_dir)):
        found.append("a picture was adopted under --auto")
    return found


SCENARIO_CHECKS = {"representation": representation, "resume": resume,
                   "visualization": visualization}
