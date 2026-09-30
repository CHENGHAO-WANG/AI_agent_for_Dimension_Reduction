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
LOOP = re.compile(r"\b(for|while)\b")
READING_TOOLS = {"Read", "Glob", "Grep", "Bash"}


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
        if c.name not in READING_TOOLS:
            continue
        text = _norm(json.dumps(c.input))
        others = {m for m in re.findall(r"runs/([\w.-]+)", text) if m != run_id.lower()}
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
