# dr-agent

An AI agent that performs exploratory data analysis through dimension reduction on
datasets it has not seen before. Given a dataset, it profiles the data, gathers
evidence about its structure, plans an analysis, executes it, evaluates the resulting
embeddings, and writes its own report.

> Status: under construction. See [`design/notes.md`](design/notes.md) for the full
> design and its rationale.

## How it works

**Claude Code is the agent runtime.** A deterministic Python toolbox (`drtools`) does
all of the mathematics and contains no LLM code; the agent reads its JSON output and
decides what to do next. Five chained skills, driven by one `/analyze` command:

```
profile-dataset -> plan-analysis -> execute-plan -> evaluate-embeddings -> write-report
```

Four properties shape the design:

- **Locked core, open adapter.** Method selection, execution, evaluation and ranking
  go exclusively through audited `drtools` commands, so every number in the report
  comes from code that can be inspected. The one escape hatch is data loading, where
  unfamiliar formats are handled by an agent-written adapter that is contract-checked
  before use.
- **The run directory is the only state.** Nothing lives in conversation context.
  Every decision lands in `runs/<id>/decisions.jsonl` with an `evidence` field whose
  keys must resolve in `profile.json` or `metrics.json`, and the report is generated
  *from* that log — so no rationale can rest on a number the run never computed, and
  every claim can be followed back to the artefact behind it.
- **Pre-registered evaluation.** The agent declares how it will weight the evaluation
  metrics *before* any embedding is computed, which rules out choosing a weighting
  that flatters whichever method happened to win.
- **Baselines that separate method from preprocessing.** Every Plan holds the Linear
  baseline, a single PCA on the base preprocessing. When other Candidates keep the
  2,000 most variable features and z-score them, the Plan also holds the Selected
  baseline, the same PCA behind that selection, so a lead over PCA can be split between
  the method and the preprocessing.

## Quick start

Two pieces: the plugin carries the agent's skills and the `/analyze` command, and the
`drtools` toolbox does the mathematics they drive. Both are needed.

```bash
pip install dr-agent            # the toolbox
pip install --no-deps trimap==1.2.0
```

TriMap installs on its own line: its package declares `annoy`, which has no Windows
wheel for Python 3.12, and the class the toolbox uses does not need it. Without it,
TriMap refuses to run and every other method works.

Then install the plugin from this repository in Claude Code, and run:

```
/dr-agent:analyze pbmc3k
```

Plugin commands are namespaced, so the command is `/dr-agent:analyze`. Everything the Run
produces is in `runs/<id>/results/`, and `drtools status --run-dir runs/<id>` reports
`done` once the report is rendered.

Working in a clone instead:

```bash
git clone <repo-url> && cd dr-agent
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install --no-deps trimap==1.2.0
```

## Headless runs and the pre-flight

`preflight/launch.py` runs the agent with `claude -p`, in a new folder outside the
repository, with the permissions the graded Runs use. From a clone:

```bash
python -m preflight.launch graded --dataset pbmc3k --run-id graded-pbmc3k
```

Set `DRTOOLS_DATA_DIR` to a download cache so each folder does not fetch PBMC3k or
PathMNIST again. Folders land beside the repository, in `dr-agent-preflight/`, or
wherever `--out` says.

The same script runs the pre-flight: three short scenarios on a synthetic count matrix,
checking how the agent behaves rather than what it finds -- nothing left running in the
background, no slow commands chained into one call, nothing read outside its own Run,
nothing written but its inputs and its report, and a Run that finishes.

```bash
python -m preflight.launch representation
python -m preflight.launch resume
python -m preflight.launch visualization
DRAGENT_PREFLIGHT=../dr-agent-preflight pytest tests/test_preflight.py
```

Each scenario costs one to two agent sessions, about $2 and 10 minutes. The ordinary
test suite calls no agent; it checks the pre-flight's checks against recorded runs
that broke them.

## Repository layout

| Path | Contents |
|---|---|
| `drtools/` | Deterministic toolbox: loaders, profiler, executors, metrics, viz |
| `skills/` | The five agent skills |
| `commands/` | The `/dr-agent:analyze` command |
| `.claude-plugin/` | Plugin manifest, so installing this repository delivers both |
| `preflight/` | The headless launch and the agent-behaviour checks |
| `.claude/` | Build configuration for working *on* dr-agent, not part of the product |
| `runs/` | Run directories (git-ignored) |
| `generated_report_1/`, `generated_report_2/` | The graded reports, PBMC3k and PathMNIST, with their figures and export |
| `tests/` | Pipeline tests over synthetic fixtures, and the pre-flight's checks |
| `design/notes.md` | Design decisions and their rationale |

## License

MIT. See [LICENSE](LICENSE).
