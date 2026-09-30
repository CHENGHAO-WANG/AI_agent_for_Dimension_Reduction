# Agent-behaviour pre-flight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pre-flight that launches the dr-agent plugin headless exactly as day 24 will,
on small synthetic data, and checks the agent's conduct from its transcript and Run.

**Architecture:** A `preflight/` package at the repository root: `transcript.py` reads a
stream-json transcript into tool calls; `checks.py` holds the conduct checks (over a
transcript) and the Run checks (over a Run directory); `launch.py` runs a scenario and
writes its folder. `tests/test_preflight_checks.py` validates the checks against trimmed
day 20 and 21 transcripts and always runs; `tests/test_preflight.py` applies them to a
pre-flight's output when `DRAGENT_PREFLIGHT` names one.

**Tech Stack:** Python 3.12, pytest, the `claude` CLI (`-p`, `--output-format stream-json`).

**Spec:** `design/specs/2026-09-30-agent-behaviour-preflight.md`

One implementer, in order; later tasks refer back to earlier ones.

## Global Constraints

- The allowlist is exactly `Bash(drtools:*) Bash(ls:*) Bash(cat:*) Read Write Edit Glob Grep Skill`, with `PowerShell` disallowed, hooks disabled (`{"disableAllHooks": true}`) and `--strict-mcp-config`.
- The prompt is `/dr-agent:analyze <args>`, passed as one list element; the plugin directory is the repository root in `D:/...` form (`Path.as_posix()`).
- Each scenario runs in its own new directory outside the repository.
- The default `pytest` run makes no agent call and spends nothing.
- Slow commands: `drtools embed`, `evaluate`, `prepare-reference`, `recon`, `render`.
- Allowed agent writes: `runs/<id>/inputs/…`, `runs/<id>/plan.json`, `runs/<id>/results/report.md`. (The spec lists only the first and last; `validate-plan` reads `plan.json`, which the agent writes. Task 1 corrects the spec.)

## Review Focus

- A transcript line that is not JSON, or an event with no `message` -- a truncated transcript after a crash. Expected: the reader skips what it cannot parse and still returns the calls before it. Pinned in Task 1.
- Windows paths in either slash form and either case (`D:\PythonProject\...`, `d:/pythonproject/...`, `/d/PythonProject/...`). Expected: check 3 and check 4 treat them as the same path. Pinned in Task 1.
- A Run id that is a prefix of another (`day21-pathmnist` inside `day21-pathmnist-b`). Expected: check 3 does not mistake the own Run for another. Pinned in Task 1.
- `ls runs/` with no Run named -- `/analyze` does this to find a Run to resume. Expected: allowed. Pinned in Task 1.
- A first `resume` launch that reaches `done` inside the turn limit. Expected: the scenario fails saying it tested nothing, rather than passing. Pinned in Task 3.

---

### Task 1: The transcript reader and the conduct checks, validated on recorded runs

**Files:**
- Create: `preflight/__init__.py`, `preflight/transcript.py`, `preflight/checks.py`
- Create: `tests/fixtures/transcripts/{day20,day20b,day21,day21b,day21b2}.jsonl` (trimmed)
- Create: `tests/test_preflight_checks.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`: add `pythonpath = ["."]`)
- Modify: `design/specs/2026-09-30-agent-behaviour-preflight.md` (check 4's allowed writes)

**Interfaces:**
- Produces: `preflight.transcript.Call(name: str, input: dict, result: str, is_error: bool)`, `Transcript(init: dict, calls: list[Call], result: dict | None)`, `read(path) -> Transcript`, `trim(source, destination) -> None`.
- Produces: `preflight.checks.background(t) -> list[str]`, `chained_slow(t) -> list[str]`, `foreign_reads(t, run_id: str, repo: Path) -> list[str]`, `stray_writes(t, run_id: str) -> list[str]`. Each returns one line per violation; empty means it holds.

- [ ] **Step 1: Write `preflight/transcript.py`**

```python
"""A `claude -p --output-format stream-json` transcript, read as the tool calls it made.

The pre-flight judges the agent by what it did, and what it did is its tool calls: each
with its input and the result the harness returned. Lines that are not JSON are skipped,
so a transcript cut short by a crash still yields the calls before the cut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How much of a tool result a trimmed fixture keeps: enough for "moved to the
#: background" and for a refusal's first sentence.
TRIMMED_RESULT = 400


@dataclass
class Call:
    name: str
    input: dict[str, Any]
    result: str = ""
    is_error: bool = False


@dataclass
class Transcript:
    init: dict[str, Any] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    result: dict[str, Any] | None = None


def _events(path: Path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def read(path: Path) -> Transcript:
    transcript = Transcript()
    by_id: dict[str, Call] = {}
    for event in _events(path):
        if event.get("type") == "system" and event.get("subtype") == "init":
            transcript.init = event
        elif event.get("type") == "result":
            transcript.result = event
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                call = Call(block.get("name", ""), block.get("input") or {})
                transcript.calls.append(call)
                by_id[block.get("id", "")] = call
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in by_id:
                call = by_id[block["tool_use_id"]]
                call.result = _text(block.get("content"))
                call.is_error = bool(block.get("is_error"))
    return transcript


def trim(source: Path, destination: Path) -> None:
    """Keep what the checks read: the init event's commands, every tool call and the
    start of its result, and the result event's totals."""
    lines = []
    for event in _events(source):
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            lines.append({"type": "system", "subtype": "init",
                          "slash_commands": event.get("slash_commands", [])})
        elif kind == "result":
            lines.append({"type": "result", **{k: event.get(k) for k in
                          ("num_turns", "duration_ms", "total_cost_usd")}})
        elif kind in ("assistant", "user") and isinstance(event.get("message"), dict):
            blocks = []
            for block in event["message"].get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    blocks.append({k: block.get(k) for k in ("type", "id", "name", "input")})
                elif block.get("type") == "tool_result":
                    blocks.append({"type": "tool_result", "tool_use_id": block.get("tool_use_id"),
                                   "is_error": bool(block.get("is_error")),
                                   "content": _text(block.get("content"))[:TRIMMED_RESULT]})
            if blocks:
                lines.append({"type": kind, "message": {"content": blocks}})
    Path(destination).write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
```

`preflight/__init__.py` holds one line: `"""The agent-behaviour pre-flight (design/specs/2026-09-30-agent-behaviour-preflight.md)."""`

- [ ] **Step 2: Make the trimmed fixtures**

```bash
cd /d/PythonProject/AI_agent_for_Dimension_Reduction && mkdir -p tests/fixtures/transcripts && .venv/Scripts/python -c "
from pathlib import Path
from preflight.transcript import trim
src = Path('D:/PythonProject/dr-agent-e2e')
for name in ['day20', 'day20b', 'day21', 'day21b', 'day21b2']:
    trim(src / f'{name}.jsonl', Path(f'tests/fixtures/transcripts/{name}.jsonl'))
" && ls -la tests/fixtures/transcripts
```

Expected: five files, each well under 200 KB. Open one and confirm it holds `tool_use` and `tool_result` blocks and no assistant prose.

- [ ] **Step 3: Write the failing validation tests** in `tests/test_preflight_checks.py`

```python
"""The conduct checks, validated against runs that really broke each rule.

A check that stays green on the failure it exists for checks nothing. So each has a
recorded transcript from days 20 and 21 it must fail on, and one it must pass on. These
run in the default suite: they read fixtures and call no agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preflight.checks import background, chained_slow, foreign_reads, stray_writes
from preflight.transcript import Transcript, Call, read

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
REPO = Path(__file__).resolve().parents[1]
RUN_IDS = {"day20": "day20-pbmc3k", "day20b": "day20-pbmc3k-b", "day21": "day21-pathmnist",
           "day21b": "day21-pathmnist-b", "day21b2": "day21-pathmnist-b"}


def _t(name: str) -> Transcript:
    return read(FIXTURES / f"{name}.jsonl")


def test_background_fails_on_the_run_that_backgrounded_an_embed():
    assert background(_t("day21"))
    assert not background(_t("day21b2"))


def test_chained_slow_fails_on_the_run_that_chained_three_evaluates():
    assert chained_slow(_t("day21b"))
    assert not chained_slow(_t("day21b2"))


def test_foreign_reads_fails_on_the_run_that_read_the_earlier_plan():
    assert foreign_reads(_t("day21b"), RUN_IDS["day21b"], REPO)
    assert not foreign_reads(_t("day21b2"), RUN_IDS["day21b2"], REPO)


def test_stray_writes_fails_on_the_run_that_wrote_into_the_working_directory():
    assert stray_writes(_t("day20"), RUN_IDS["day20"])
    assert not stray_writes(_t("day20b"), RUN_IDS["day20b"])


def test_a_truncated_transcript_still_yields_its_calls(tmp_path):
    good = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "drtools status"}}]}}
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(good) + "\n" + '{"type": "assist', encoding="utf-8")
    assert [c.name for c in read(path).calls] == ["Bash"]


def _with(*calls: Call) -> Transcript:
    return Transcript(calls=list(calls))


@pytest.mark.parametrize("path", [
    "D:\\PythonProject\\AI_agent_for_Dimension_Reduction\\tests\\plans.py",
    "d:/pythonproject/ai_agent_for_dimension_reduction/drtools/cli.py",
    "/d/PythonProject/AI_agent_for_Dimension_Reduction/design/notes.md",
])
def test_a_read_of_the_repository_is_caught_in_every_path_form(path):
    assert foreign_reads(_with(Call("Read", {"file_path": path})), "r1", REPO)


def test_the_own_run_is_not_mistaken_for_another_it_extends():
    own = _with(Call("Read", {"file_path": "D:\\e2e\\runs\\day21-pathmnist-b\\plan.json"}))
    assert not foreign_reads(own, "day21-pathmnist-b", REPO)
    other = _with(Call("Read", {"file_path": "D:\\e2e\\runs\\day21-pathmnist\\plan.json"}))
    assert foreign_reads(other, "day21-pathmnist-b", REPO)


def test_listing_runs_to_find_one_to_resume_is_allowed():
    assert not foreign_reads(_with(Call("Bash", {"command": "ls runs/ 2>/dev/null"})), "r1", REPO)


def test_a_loop_over_a_slow_command_counts_as_chaining():
    loop = Call("Bash", {"command": "for c in a b; do drtools evaluate --id $c; done"})
    assert chained_slow(_with(loop))
    assert not chained_slow(_with(Call("Bash", {"command": "drtools checkpoint && drtools status"})))
```

- [ ] **Step 4: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_preflight_checks.py -q`
Expected: collection error, `No module named 'preflight.checks'` (after `pythonpath = ["."]` is added to `pyproject.toml`; without it, `No module named 'preflight'`).

- [ ] **Step 5: Write `preflight/checks.py`** (conduct half)

```python
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
```

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_preflight_checks.py -q`
Expected: all pass. If a "must pass on" fixture fails, print the violation lines, and decide whether the recorded run broke the rule (then pick another clean fixture and say so in the log) or the check is wrong (then fix the check). Do not loosen a check to make a recorded failure pass.

- [ ] **Step 7: Correct the spec's check 4**, so it reads: "Every Write and Edit lands in `runs/<id>/inputs/`, or is `runs/<id>/plan.json` or `runs/<id>/results/report.md`."

- [ ] **Step 8: Commit**

```bash
git add preflight tests/fixtures/transcripts tests/test_preflight_checks.py pyproject.toml design/specs/2026-09-30-agent-behaviour-preflight.md
git commit -m "Day 23: the pre-flight's conduct checks, validated on recorded runs"
```

---

### Task 2: The `wide_counts` fixture

**Files:**
- Modify: `drtools/synthetic.py` (`GENERATORS`)
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Produces: `drtools datasets` lists `wide_counts`; `load("wide_counts")` returns 800 × 3,000 non-negative integer counts.

- [ ] **Step 1: Write the failing test** (append to `tests/test_synthetic.py`)

```python
def test_wide_counts_is_wide_enough_for_selection_and_is_counts():
    """Day 23's pre-flight fixture: more than 2,000 features of one type, so the
    Selected baseline is required, and integer counts, so the counts path runs."""
    from drtools.loaders import load

    X, _, _ = load("wide_counts")
    dense = X.toarray() if hasattr(X, "toarray") else X
    assert dense.shape == (800, 3000)
    assert (dense >= 0).all() and (dense == dense.round()).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_synthetic.py -q -k wide_counts`
Expected: FAIL, `unknown synthetic dataset 'wide_counts'`.

- [ ] **Step 3: Implement** -- in `drtools/synthetic.py`, above `GENERATORS`:

```python
def wide_counts(seed: int = 0) -> Dataset:
    """`sparse_counts` at 3,000 genes: wide enough that selection applies, so the
    Selected baseline is required. The agent-behaviour pre-flight runs on it (day 23)."""
    return sparse_counts(n_cells=800, n_genes=3000, seed=seed)
```

and add `"wide_counts": wide_counts,` to `GENERATORS`. If `sparse_counts` returns metadata with a `name`, set it to `"wide_counts"` on the returned dict, so the Run records the fixture it was given.

- [ ] **Step 4: Run to verify it passes**, then `.venv/Scripts/python -m pytest tests/test_synthetic.py tests/test_loaders.py -q`. Expected: pass.

- [ ] **Step 5: Commit** -- `git commit -m "Day 23: the wide_counts fixture"`

---

### Task 3: The Run checks

**Files:**
- Modify: `preflight/checks.py` (append)
- Test: `tests/test_preflight_checks.py` (append)

**Interfaces:**
- Consumes: `read`, `Transcript` (Task 1).
- Produces: `finished(run_dir: Path) -> list[str]`; `representation(run_dir, summary, transcripts) -> list[str]`, `resume(...)`, `visualization(...)` with the same signature; `SCENARIO_CHECKS: dict[str, Callable]`. `summary` is the dict of Task 4's `summary.json`; `transcripts` is `list[Transcript]` in launch order.

- [ ] **Step 1: Write the failing tests** (append)

```python
from preflight.checks import finished, representation, resume, visualization


def test_finished_names_the_stage_a_bare_run_is_at(tmp_path):
    from drtools.cli import main
    assert main(["profile", "--data", "blobs", "--runs-root", str(tmp_path), "--run-id", "r"]) == 0
    problems = finished(tmp_path / "r")
    assert any("not done" in p for p in problems)
    assert any("report.pdf" in p for p in problems)


def test_representation_reads_the_decision_and_the_plan(finished_run):
    # blobs is not counts and selects nothing, so both checks must speak.
    problems = representation(finished_run.path, {}, [])
    assert any("raw_counts" in p for p in problems)
    assert any("Selected baseline" in p for p in problems)


def test_visualization_refuses_a_ranked_run(finished_run):
    assert any("ranking" in p for p in visualization(finished_run.path, {}, []))


def test_resume_says_it_tested_nothing_when_the_first_launch_finished(finished_run):
    summary = {"launches": [{"next": "done"}, {"next": "done"}]}
    assert any("tested nothing" in p for p in resume(finished_run.path, summary, [Transcript(), Transcript()]))
```

- [ ] **Step 2: Run to verify it fails** -- `ImportError: cannot import name 'finished'`.

- [ ] **Step 3: Implement** (append to `preflight/checks.py`)

```python
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
    bash = [c.input.get("command", "") for c in transcripts[1].calls if c.name == "Bash"]
    status_at = next((i for i, cmd in enumerate(bash) if "drtools status" in cmd), None)
    stage_at = next((i for i, cmd in enumerate(bash) if STAGE.search(cmd)), None)
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
```

- [ ] **Step 4: Run to verify it passes** -- `.venv/Scripts/python -m pytest tests/test_preflight_checks.py -q`.

- [ ] **Step 5: Commit** -- `git commit -m "Day 23: the pre-flight's Run checks"`

---

### Task 4: The launch script and the scenario tests

**Files:**
- Create: `preflight/launch.py`
- Create: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `read` (Task 1), `SCENARIO_CHECKS`, `finished` and the conduct checks (Tasks 1, 3).
- Produces: `python -m preflight.launch <scenario> [--dataset D] [--run-id ID] [--out DIR]` writing `<out>/<scenario>-<YYYYmmdd-HHMMSS>/` with `summary.json` = `{"scenario": str, "run_id": str, "launches": [{"transcript": str, "turns": int, "minutes": float, "cost_usd": float, "next": str | None}]}`. `build_command(claude: str, prompt: str, max_turns: int | None) -> list[str]`; `plugin_loaded(t: Transcript) -> bool`.

- [ ] **Step 1: Write the failing tests** in `tests/test_preflight.py`

```python
"""The pre-flight: its launch command, and the checks applied to a real pre-flight.

The scenario test reads the folder `DRAGENT_PREFLIGHT` names, written by
`python -m preflight.launch`, and is skipped when it is unset, so the default suite
calls no agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from preflight.checks import (SCENARIO_CHECKS, background, chained_slow, finished,
                              foreign_reads, stray_writes)
from preflight.launch import ALLOWED, REPO, build_command, plugin_loaded
from preflight.transcript import Transcript, read

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"


def test_the_command_carries_day_24s_configuration_unchanged():
    cmd = build_command("claude", "/dr-agent:analyze wide_counts --auto --run-id r", None)
    assert cmd[1:3] == ["-p", "/dr-agent:analyze wide_counts --auto --run-id r"]
    assert cmd[cmd.index("--plugin-dir") + 1] == REPO.as_posix()
    assert all(tool in cmd for tool in ALLOWED)
    assert cmd[cmd.index("--disallowedTools") + 1] == "PowerShell"
    assert "--max-turns" not in cmd
    assert build_command("claude", "x", 25)[-2:] == ["--max-turns", "25"]


def test_a_transcript_without_the_plugin_is_recognised():
    assert plugin_loaded(read(FIXTURES / "day21b2.jsonl"))
    assert not plugin_loaded(Transcript(init={"slash_commands": ["analyze"]}))


PREFLIGHT = os.environ.get("DRAGENT_PREFLIGHT")


@pytest.mark.parametrize("scenario", sorted(SCENARIO_CHECKS))
def test_scenario(scenario):
    if not PREFLIGHT:
        pytest.skip("DRAGENT_PREFLIGHT is not set")
    folders = sorted(Path(PREFLIGHT).glob(f"{scenario}-*"))
    if not folders:
        pytest.skip(f"no {scenario} folder under {PREFLIGHT}")
    folder = folders[-1]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    run_dir = folder / "runs" / summary["run_id"]
    transcripts = [read(folder / launch["transcript"]) for launch in summary["launches"]]
    problems = []
    for t in transcripts:
        problems += background(t) + chained_slow(t)
        problems += foreign_reads(t, summary["run_id"], REPO) + stray_writes(t, summary["run_id"])
    problems += finished(run_dir) + SCENARIO_CHECKS[scenario](run_dir, summary, transcripts)
    assert not problems, "\n".join(problems)
```

- [ ] **Step 2: Run to verify it fails** -- `No module named 'preflight.launch'`.

- [ ] **Step 3: Implement `preflight/launch.py`**

```python
"""Launch the dr-agent plugin headless, exactly as the graded Runs are launched.

    python -m preflight.launch representation | resume | visualization
    python -m preflight.launch graded --dataset pbmc3k --run-id graded-pbmc3k

Arguments go to `claude` as a list, so no shell rewrites the prompt or the plugin path
(day 21's Git Bash traps). Each scenario runs in a new folder outside the repository, so
neither the build instructions in `.claude/` nor an earlier Run can reach the agent.
Set DRTOOLS_DATA_DIR to reuse a download cache for the graded datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from preflight.transcript import Transcript, read

REPO = Path(__file__).resolve().parents[1]
ALLOWED = ["Bash(drtools:*)", "Bash(ls:*)", "Bash(cat:*)",
           "Read", "Write", "Edit", "Glob", "Grep", "Skill"]
SCENARIOS = {
    "representation": ("wide_counts --auto", "pf-rep"),
    "resume": ("wide_counts --auto", "pf-resume"),
    "visualization": ("wide_counts --auto --purpose visualization", "pf-vis"),
}
#: Registration finished inside 25 turns on days 20 and 21, so this stops the first
#: `resume` launch partway through execution.
RESUME_FIRST_TURNS = 25


def build_command(claude: str, prompt: str, max_turns: int | None) -> list[str]:
    command = [claude, "-p", prompt,
               "--plugin-dir", REPO.as_posix(),
               "--settings", json.dumps({"disableAllHooks": True}),
               "--strict-mcp-config",
               "--disallowedTools", "PowerShell",
               "--allowedTools", *ALLOWED,
               "--output-format", "stream-json", "--verbose"]
    if max_turns:
        command += ["--max-turns", str(max_turns)]
    return command


def plugin_loaded(transcript: Transcript) -> bool:
    return "dr-agent:analyze" in transcript.init.get("slash_commands", [])


def _next_stage(run_dir: Path) -> str | None:
    from drtools.runs import RunDir
    from drtools.status import run_status

    return run_status(RunDir(run_dir, create=False))["next"] if run_dir.exists() else None


def _launch(claude: str, prompt: str, workdir: Path, name: str, run_id: str,
            env: dict, max_turns: int | None) -> dict:
    with open(workdir / name, "w", encoding="utf-8") as out, \
            open(workdir / f"{name}.err", "w", encoding="utf-8") as err:
        subprocess.run(build_command(claude, prompt, max_turns), cwd=workdir,
                       stdout=out, stderr=err, env=env, check=False)
    transcript = read(workdir / name)
    if not plugin_loaded(transcript):
        raise SystemExit(f"the dr-agent plugin did not load from {REPO.as_posix()}; "
                         f"see {workdir / name}")
    result = transcript.result or {}
    return {"transcript": name, "turns": result.get("num_turns"),
            "minutes": round((result.get("duration_ms") or 0) / 60000, 1),
            "cost_usd": result.get("total_cost_usd"),
            "next": _next_stage(workdir / "runs" / run_id)}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=[*SCENARIOS, "graded"])
    parser.add_argument("--dataset", help="the dataset spec, for a graded Run")
    parser.add_argument("--run-id", help="the Run id, for a graded Run")
    parser.add_argument("--out", type=Path, default=REPO.parent / "dr-agent-preflight")
    args = parser.parse_args(argv)

    if args.scenario == "graded":
        if not (args.dataset and args.run_id):
            parser.error("a graded Run needs --dataset and --run-id")
        arguments, run_id = f"{args.dataset} --auto", args.run_id
    else:
        arguments, run_id = SCENARIOS[args.scenario]
    prompt = f"/dr-agent:analyze {arguments} --run-id {run_id}"

    claude = shutil.which("claude")
    if claude is None:
        raise SystemExit("claude is not on PATH")
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if shutil.which("drtools", path=env["PATH"]) is None:
        raise SystemExit("drtools is not on PATH; install the toolbox into this Python")

    workdir = args.out / f"{args.scenario}-{datetime.now():%Y%m%d-%H%M%S}"
    workdir.mkdir(parents=True)
    launches = []
    if args.scenario == "resume":
        launches.append(_launch(claude, prompt, workdir, "transcript-1.jsonl", run_id, env,
                                RESUME_FIRST_TURNS))
        launches.append(_launch(claude, prompt, workdir, "transcript-2.jsonl", run_id, env, None))
    else:
        launches.append(_launch(claude, prompt, workdir, "transcript.jsonl", run_id, env, None))
    summary = {"scenario": args.scenario, "run_id": run_id, "launches": launches}
    (workdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\ncheck it with: DRAGENT_PREFLIGHT={args.out.as_posix()} pytest tests/test_preflight.py")
    return workdir


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes** -- `.venv/Scripts/python -m pytest tests/test_preflight.py tests/test_preflight_checks.py -q`. Expected: the two launch tests pass, the scenario tests skip.

- [ ] **Step 5: Full suite** -- `.venv/Scripts/python -m pytest -q > suite.txt 2>&1; echo "pytest exit $?"`. Expected: exit 0, and no new agent calls.

- [ ] **Step 6: Commit** -- `git commit -m "Day 23: the pre-flight launch script and scenario test"`

---

### Task 5: Run the pre-flight, and fix what it breaks

- [ ] **Step 1:** From the repository, run the scenarios one at a time; each takes about 5 to 10 minutes and one agent session:

```bash
.venv/Scripts/python -m preflight.launch representation
.venv/Scripts/python -m preflight.launch resume
.venv/Scripts/python -m preflight.launch visualization
```

- [ ] **Step 2:** `DRAGENT_PREFLIGHT=D:/PythonProject/dr-agent-preflight .venv/Scripts/python -m pytest tests/test_preflight.py -v`
- [ ] **Step 3:** For each failure: read the transcript to find the cause, reproduce it, and fix it where it lives -- a skill, `/analyze`, the toolbox, or a check that proved wrong. Each fix gets a test that fails without it, as on days 20 and 21. A check is corrected only when the transcript shows the agent's conduct was right.
- [ ] **Step 4:** Rerun only the scenarios a fix touches, until all three pass. Record turns, minutes and dollars from each `summary.json` for the log.
- [ ] **Step 5:** Commit each fix separately, naming what the pre-flight found.

---

### Task 6: README and the glossary review

**Files:** `README.md`; `CONTEXT.md`; `skills/*/SKILL.md` and `commands/analyze.md` only where the review finds a term to correct.

- [ ] **Step 1: README.** Replace the Quick start's manual launch with `python -m preflight.launch`: the pre-flight's three scenarios and how to check them, and `graded --dataset … --run-id …` for the graded Runs, with `DRTOOLS_DATA_DIR` for the download cache. Say what days 20 to 22 changed: a Plan whose Candidates select features holds the Selected baseline; `drtools status` says `done` once the report is rendered. Add `preflight/` to the repository layout table.
- [ ] **Step 2: Glossary review.** Read each skill and `/analyze` against every term in `CONTEXT.md` and its `_Avoid_` list. Replace any avoided synonym with the term. List each concept that has no term; add a term only through the `domain-modeling` skill, and say which were added.
- [ ] **Step 3:** `.venv/Scripts/python -m pytest tests/test_skills.py -q` -- the prose still names only commands and flags that exist.
- [ ] **Step 4: Commit** -- `git commit -m "Day 23: README for the pre-flight and graded launch; glossary review"`

---

### Task 7: The decision log

- [ ] **Step 1:** Append a **Day 23** entry to `design/notes.md` (CRLF line endings, as the file has): what the pre-flight is and why it is split from its checks; the checks validated on recorded runs, and the one check the recordings changed before it was written ("one command per call" → "no two slow commands"); check 4's allowed writes widened to `plan.json`; each scenario's turns, minutes and dollars; what the pre-flight found and how each was fixed; the glossary review's result. Then the test count and time.
- [ ] **Step 2: Commit** -- `git commit -m "Day 23: decision log"`
