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

Three properties shape the design:

- **Locked core, open adapter.** Method selection, execution, evaluation and ranking
  go exclusively through audited `drtools` commands, so every number in the report
  comes from code that can be inspected. The one escape hatch is data loading, where
  unfamiliar formats are handled by an agent-written adapter that is contract-checked
  before use.
- **The run directory is the only state.** Nothing lives in conversation context.
  Every decision lands in `runs/<id>/decisions.jsonl` with an `evidence` field
  pointing at a key that actually exists in `profile.json` or `metrics.json`, and the
  report is generated *from* that log — so the agent cannot claim a decision it did
  not make.
- **Pre-registered evaluation.** The agent declares how it will weight the evaluation
  metrics *before* any embedding is computed, which rules out choosing a weighting
  that flatters whichever method happened to win.

## Quick start

```bash
git clone <repo-url> && cd dr-agent
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Then open Claude Code in the repository and run:

```
/analyze pbmc3k
```

The toolbox is also usable on its own, with a rule-based planner and no LLM:

```bash
drtools run --planner=rules --data pbmc3k --out runs/
```

## Repository layout

| Path | Contents |
|---|---|
| `drtools/` | Deterministic toolbox: loaders, profiler, executors, metrics, viz |
| `skills/` | The five agent skills |
| `commands/` | The `/analyze` command |
| `runs/` | Run directories (git-ignored) |
| `tests/` | Pipeline and agent-behaviour tests over synthetic fixtures |
| `design/notes.md` | Design decisions and their rationale |
| `docs/adr/` | Architecture decision records |

## License

MIT. See [LICENSE](LICENSE).
