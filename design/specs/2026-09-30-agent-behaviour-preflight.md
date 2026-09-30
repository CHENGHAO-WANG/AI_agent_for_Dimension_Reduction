# Day 23: the agent-behaviour pre-flight

## What it is for

The toolbox has 798 tests, and `tests/test_skills.py` checks that every command and flag
the skills name exists. Nothing checks what the agent *does* when it follows the skills.
Days 20 and 21 found exactly that kind of defect -- a command left in the background, three
slow commands chained into one call, another Run read while planning, JSON written into
the user's directory -- and found them only by running the agent by hand and reading its
transcripts.

The pre-flight is a check run by hand before day 24's graded Runs, and again after any late
change to the skills. It launches the agent headless on small synthetic data, exactly as
day 24 will launch it, and asserts over the transcript and the Run directory. It is not part
of the default `pytest` run, since each scenario costs an agent run.

Decided in the day 23 design pass: the purpose (a pre-flight, not evidence for the report
and not a replay of recorded runs only); three scenarios; and the split between a launch
script and the checks.

## The launch: `preflight/launch.py`

A Python script, so that `claude` is called with its arguments as a list and no shell
rewrites them. That removes both of day 21's Git Bash traps: the prompt
`/dr-agent:analyze` stays as written, and the plugin directory stays in `D:/...` form.

**One configuration, day 24's.** `--plugin-dir` at the repository root; hooks disabled;
`--strict-mcp-config` with no servers; PowerShell disallowed; and the allowlist, written
here and nowhere else:

```
Bash(drtools:*)  Bash(ls:*)  Bash(cat:*)  Read  Write  Edit  Glob  Grep  Skill
```

It is kept as narrow as days 20 and 21 ran it. The refusals it produced -- a heredoc, a
`for` loop, a redirect, `python -c` -- cost turns, never correctness, and the skills
already steer the agent to the Write tool. `--output-format stream-json --verbose`.

**A clean working directory per scenario**, outside the repository, so the build
instructions in `.claude/` cannot reach the agent (day 20) and no earlier Run is there to
be read (day 21). A scenario's folder holds `transcript.jsonl` (two for `resume`),
`runs/<id>/`, and `summary.json`: turns, minutes and dollars from each transcript's result
event. The folder's root is an argument; each scenario gets a timestamped subfolder.

**Scenarios.**

| Scenario | Prompt | Notes |
|---|---|---|
| `representation` | `/dr-agent:analyze wide_counts --auto --run-id pf-rep` | |
| `resume` | the same with `--run-id pf-resume` | first launch with `--max-turns 25`, then relaunched unchanged |
| `visualization` | `/dr-agent:analyze wide_counts --auto --purpose visualization --run-id pf-vis` | |
| `graded <dataset>` | `/dr-agent:analyze <dataset> --auto --run-id <id>` | day 24's Runs; no checks attached |

`--max-turns 25` stops the first `resume` launch after registration and partway through
execution, going by days 20 and 21, where registration finished inside 25 turns. If a
first launch reaches `done` anyway, the scenario reports that it tested nothing rather
than passing.

**Failing early.** The script stops, naming the cause, when `claude` is not on the PATH,
when `drtools` is not, and when the transcript's first event does not list
`dr-agent:analyze` among its commands -- the plugin did not load, which on day 21 was
silent.

## The fixture: `wide_counts`

`sparse_counts(n_cells=800, n_genes=3000)`, registered as its own name in
`drtools/synthetic.py`, since the CLI passes no generator arguments. At 3,000 genes of
one type, selection applies, so the Selected baseline is required; the values are
non-negative integers, so the counts path runs. It exercises what PBMC3k does at a
fraction of the time.

## The checks: `tests/test_preflight.py`

A reader turns a transcript into its tool calls, each with its input and its result.

**Every scenario:**
1. *Nothing ran in the background.* No call set `run_in_background`, and no result says
   the call was moved to the background.
2. *No two slow commands share a call.* A Bash call holds at most one of `drtools embed`,
   `evaluate`, `prepare-reference`, `recon` and `render`. Quick pairs such as
   `checkpoint && status` are allowed: every recorded transcript has several, and none
   did harm.
3. *The agent read only its own Run.* No Read, Glob, Grep, `ls` or `cat` touches the
   plugin's repository -- the skills arrive through the Skill tool -- or any Run other than
   its own.
4. *It wrote only its inputs and its prose.* Every Write and Edit lands in
   `runs/<id>/inputs/` or is `runs/<id>/results/report.md`.
5. *The Run is finished.* `drtools status` reports `done`, and `results/report.pdf`
   exists.

**`representation`:** the recorded data decision says raw counts; the Decision log
carries the counts statement `--auto` requires; the registered Plan holds the Linear
baseline and the Selected baseline; `ranking.json` exists.

**`resume`:** the first transcript ended before `done`; the log holds exactly one
`register_plan` record; no Candidate whose first attempt succeeded has a second; in the
second transcript, `drtools status` comes before any stage command.

**`visualization`:** no `ranking.json`; a recommendation exists; no `adopt` record.

**How they run.** The scenario checks read the folder named by `DRAGENT_PREFLIGHT` and are
skipped when it is unset, so the default suite is unchanged and costs nothing.

**Validating the checks.** Checks 1 to 4 are only worth something if they fail on the
failures they are for. Trimmed copies of the day 20 and 21 transcripts -- tool calls and
results only -- go into `tests/fixtures/transcripts/`, and each check has a test that always
runs:

| Check | Must fail on | Must pass on |
|---|---|---|
| 1 background | `day21` (3 background calls) | `day21b2` |
| 2 slow commands chained | `day21b` (three `evaluate` in one call) | `day21b2` |
| 3 other Runs | `day21b` (read `runs/day21-pathmnist`) | `day21b2` |
| 4 writes | `day20` (`decision.json` in the working directory) | `day20b` |

## The rest of day 23

- **README.** The manual launch instructions give way to the launch script, for the
  pre-flight and for the graded Runs; it describes what days 20 to 22 changed: the
  Selected baseline, `done`, and no rules planner.
- **CONTEXT.md review.** Read the five skills and `/analyze` against the glossary: each
  domain concept named by its term, and a missing term flagged rather than invented.
- **Order.** The reader and its validation tests first, against the recorded
  transcripts; then the fixture and the launch script; then the three scenarios, run
  once, and whatever they break fixed, as on days 20 and 21; then the README and the
  glossary review; then the decision log.

## Out of scope

PathMNIST-scale behaviour -- long commands, subsampling, the 600 s timeout -- which
synthetic data cannot provoke cheaply and day 21's real Run exercised. A check that every
Evidence key cites something true: the toolbox already refuses one that does not resolve.
Any assertion about which method wins: that is the analysis, not the agent's conduct.
