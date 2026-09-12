# dr-agent — design notes

Running record of decisions and their rationale. Raw material for the manually
written 4-page `report.pdf`. Append as decisions are made; do not rewrite history.

Started 2026-09-12. Deadline 2026-09-28. Target completion 2026-09-25.

---

## 1. What the system is

An AI agent that performs exploratory data analysis through dimension reduction on
previously unseen datasets: it profiles the data, gathers evidence, plans an analysis,
executes it, evaluates the resulting embeddings, and writes its own report.

**Claude Code is the agent runtime.** A deterministic Python toolbox (`drtools`) does
all of the mathematics and contains no LLM code. Claude reads skill instructions,
calls toolbox commands, reads their JSON output, and decides what to do next.

Five chained skills driven by one `/analyze` command:

```
profile-dataset -> plan-analysis -> execute-plan -> evaluate-embeddings -> write-report
```

No subagents and no MCP server. Both were considered; both cost build time without
adding rubric value, and subagent failures are expensive to debug on a deadline.

---

## 2. The central design tension: autonomy vs. auditability

The grading criteria reward *degree of automation* and *reproducibility*
simultaneously, and those pull in opposite directions. An LLM given free rein is
autonomous but unreproducible; a rule engine is reproducible but is not an agent.

The resolution runs through most decisions below: **let the LLM make the judgments,
and make every judgment a checkable artifact.**

### 2.1 Locked core, open adapter

Method selection, execution, evaluation, and ranking go exclusively through `drtools`
CLI subcommands. Claude cannot hand-roll its own Isomap, so every number in the final
report comes from audited code.

The exception is *loading*. The spec's headline promise is "previously unseen
datasets", and unseen datasets bite at the loader: an unfamiliar file format, an odd
`.h5ad` layout, labels in an unexpected column. A rigidly locked agent simply fails
there, and that is the failure a grader is most likely to provoke. So there is a
documented loader contract:

```
load(path) -> X: ndarray, labels: ndarray | None, meta: dict
```

When no built-in loader recognises the input, the agent may write a small adapter,
which the toolbox contract-checks (shape, dtype, finiteness, label alignment) before
anything else runs. A narrow, verified escape hatch.

Recorded as ADR-0001.

### 2.2 The run directory is the only state

No state lives in conversation context. Every skill reads and writes files under
`runs/<id>/`: `profile.json`, `recon.json`, `plan.json`, `embeddings/`, `metrics/`,
`figures/`, `decisions.jsonl`, `run.json`. Each skill's first instruction is "read
`profile.json`", never "recall what you found".

This buys resumability when a long run dies, an auditable trail the grader can open,
a directory that *is* the "generated outputs" deliverable, and the ability for the
rules planner (section 6) to reuse the same executors.

### 2.3 Grounded report generation

`decisions.jsonl` is append-only, one structured record per choice point:

```json
{"stage": "...", "question": "...", "options_considered": ["..."],
 "chosen": "...", "rationale": "...",
 "evidence": ["profile.n_samples", "recon.intrinsic_dim"], "timestamp": "..."}
```

**The report is generated from the log**, not written freehand. The `evidence` field
pins every rationale to a key that actually exists in `profile.json` or
`metrics.json`, so the agent structurally cannot claim a decision it did not make or
cite a number it did not compute. This is the hallucination-control mechanism.

### 2.4 Pre-registered evaluation weights

The agent chooses how to weight the evaluation metrics, but declares those weights in
`plan.json` **before any embedding is computed**, justified from the user instruction
and the data profile. `drtools rank` then scores deterministically over numbers the
agent had not yet seen. Changing the weighting afterwards is possible only as a
logged amendment with a reason.

Without pre-registration the agent would pick weights that flatter whichever method
happened to win — post-hoc rationalisation that a statistician grading this would
spot immediately.

---

## 3. How the agent decides

### 3.1 Staged reconnaissance, not a lookup table

Planning does not happen directly from the profile. Before committing, the agent runs
a cheap probe pass — PCA spectrum and explained-variance curve, an intrinsic
dimensionality estimate, k-NN graph connectivity and degree distribution, a
small-sample UMAP thumbnail — and plans against *that evidence*.

"n=2700, sparse, therefore t-SNE" is a lookup table wearing a costume. Reconnaissance
costs seconds and changes every downstream decision: an intrinsic dimensionality of
about 3 with a connected k-NN graph justifies Isomap and Diffusion Maps; a
fast-decaying PCA spectrum says the structure is largely linear and manifold methods
will add little; a disconnected graph rules out spectral methods before they crash.

One optional re-plan is permitted after evaluation. Fully iterative planning was
rejected as unbounded in cost.

### 3.2 Candidates are pipelines, not methods

A candidate is an ordered stage list, not an algorithm name:

```json
{"id": "c3", "stages": [{"op": "log1p"}, {"op": "pca", "n": 50},
                        {"op": "umap", "n_neighbors": 30, "min_dist": 0.1}]}
```

Required for correctness — nobody runs t-SNE on 32k raw genes; `PCA -> UMAP` is the
field-standard scRNA-seq pipeline. It also makes an intermediate PCA's `n_components`
a hyperparameter the agent must justify, and it unlocks the best experiment in the
project: entering **both** `umap(raw)` and `pca50 -> umap` as candidates and letting
the metrics settle whether the pre-step helped. That is the agent designing an
experiment rather than executing a recipe.

The registry declares each method's role (`can_be_intermediate` for PCA/Kernel PCA,
`terminal_only` for t-SNE/UMAP). The plan declares a **shared base preprocessing**
applied to every candidate, with candidate-specific stages layered on top, so that
when two candidates differ it is clear what differed.

Ensembling / consensus embeddings were rejected: not in the course method list, hard
to evaluate honestly, and would consume the GPLVM budget. Named as future work.

### 3.3 Method capability records

The registry is metadata the agent reasons *over*, not a list of names — Claude cannot
introspect scikit-learn:

```yaml
isomap:
  family: manifold/global
  preserves: global geodesic structure
  assumes: single connected, densely sampled manifold
  complexity: O(n^2) memory
  max_n_recommended: 5000
  handles_sparse: false
  fails_when: [disconnected kNN graph, multiple clusters, n > 20000]
```

YAML carries the hard constraints a validator can enforce; the skill prose carries the
judgment that does not reduce to a table ("t-SNE cluster sizes and inter-cluster
distances are not meaningful — do not interpret them"). The YAML is the extension
point: adding a method is a registry entry plus one executor function, no skill edits.

### 3.4 Portfolio size

3–5 candidates per run, deliberately spanning families (one linear baseline, one
local-structure, one global-structure), with **PCA always included**. Spanning
families makes the comparison interpretable — when UMAP and PCA disagree, that
disagreement is itself a finding about the data's nonlinearity.

Every *rejection* is logged with a reason. "MDS rejected — O(n^2) at n=107,000; PCA
already captures the global variance structure it would recover" is evidence of
judgment in a way that running MDS anyway is not.

### 3.5 Hyperparameters

Profile-derived heuristic formulas as the default (perplexity clipped to data size,
`n_neighbors` scaled to n and to local density from the recon pass), which the agent
may override with a logged reason. A small sweep of 2–3 values only for the top-ranked
candidate, budget permitting.

Library defaults were rejected outright: t-SNE's default perplexity of 30 is plainly
wrong on a 500-sample dataset, and a grader in this field will check exactly that.

### 3.6 Plan validation gate

`drtools validate-plan` hard-rejects incoherent plans — t-SNE with perplexity >= n/3,
Isomap at 107k samples, raw counts into Euclidean MDS, `n_neighbors > n` — and the
agent must revise and resubmit. **Every rejection is logged**, which means the report
can honestly say "the planner proposed X, the validator caught it, the agent revised
to Y". That is a working agent loop demonstrated with evidence.

---

## 4. Robustness

Every method runs in an isolated subprocess with a wall-clock cap. Failures produce a
structured record — exception class, message, parameters used — not a traceback dump.
The agent gets exactly **one** diagnose-and-retry per method before recording a
permanent failure and continuing with the rest.

Not hypothetical: spectral methods die on disconnected neighbourhood graphs, MDS
exhausts memory, numba throws version errors, and anything on a large matrix can hang.

A report sentence like *"Laplacian Eigenmaps failed on the first attempt due to a
disconnected k-NN graph; the agent increased n_neighbors to 30 and succeeded"* is the
most convincing single piece of evidence of agency the system can produce.

`--budget fast|standard|thorough` caps per-method wall-clock and controls whether the
hyperparameter sweep runs. This turns "too big for Isomap" from an implicit constraint
into an explicit resource the agent reasons about and allocates.

---

## 5. Outputs

Markdown is the source of truth, rendered to PDF via pandoc. Fixed section skeleton so
both generated reports are comparable: dataset profile -> preprocessing decisions and
why -> methods selected *and rejected*, with reasons -> hyperparameter choices and why
-> figures -> quantitative comparison -> ranking with weighting justification ->
interpretation -> limitations.

The "methods rejected and why" section is the highest-value part: the most direct
evidence that the agent selected rather than sprayed.

**Visualisation** style is fixed in `drtools.viz`, not agent-authored — the agent
chooses *which* figures, never how they look. One palette used consistently, so the
same colour means the same class in every panel. Rendering adapts to n: markers with
edges below about 2k points, small markers with alpha to about 20k, rasterised
density/hexbin above that with an optional subsampled overlay. The agent may override
the rendering *mode* with a logged reason.

Standard figure set: a small-multiples panel (every candidate, same points, same
colours, shared legend) as the headline; per-candidate detail figures; a metrics
comparison chart; a Shepard diagram for the top candidate; the PCA scree plot from
reconnaissance.

**Interaction.** One checkpoint after profiling, at most 1–2 multiple-choice
questions, each stating a default and never blocking. `--auto` suppresses them; the
two graded runs use `--auto`, so the reports are provably produced with zero
intervention, and interactive mode is written up as a demonstrated capability rather
than a dependency.

---

## 6. The rules-planner hedge

`drtools run --planner=rules` executes the identical pipeline with a roughly 150-line
`profile -> plan` function of plain conditionals, no LLM anywhere.

Four things it buys: a grader without Claude Code still gets a working system; the
pipeline can be tested without burning LLM calls; it forces a clean tool interface,
which improves the plugin too; and it provides the ablation — run both planners on the
same data and show where Claude's judgment beat the rules.

Scheduled late (day 12), so slipping costs only the ablation.

---

## 7. Scope boundaries

**In:** PBMC3k and PathMNIST. Nine library-backed methods (PCA, Kernel PCA, Sparse
PCA, metric and non-metric MDS, Isomap, LLE + Modified LLE, Laplacian Eigenmaps,
Diffusion Maps, t-SNE, UMAP) plus a minimal MAP-GPLVM in torch, timeboxed to 3 hours
on day 11.

**Out, and named as future work in the report:** ensemble/consensus embeddings; a
cross-run experience store (with two datasets the prior would be n=2, worse than no
prior, and it would introduce hidden state that breaks reproducibility); CI.

**Deliberately deferred** — to be decided when reached, mostly *by the agent*:
PathMNIST subsampling policy, whether PBMC3k gets derived Leiden reference labels,
dataset caching and `.gitignore` handling.

---

## 8. Deliverables

Public GitHub repo as the front door. `.claude/` for zero-install use (clone, install
requirements, open Claude Code, type `/analyze`) plus a plugin manifest so it can be
installed elsewhere. Project-local `.venv` with a curated pinned `requirements.txt`.
Docs kept lean: `README.md`, `CONTEXT.md`, this file, and one ADR.

Submitted: source, `generated_report_1`, `generated_report_2`, and a manually written
`report.pdf` of at most four pages.

---

## 9. Schedule

| Day | Date | Work |
|---|---|---|
| 1 | Sep 12 | Env, pinned deps, repo skeleton, these notes, synthetic fixtures |
| 2 | Sep 13 | Loader contract, profiler, recon pass, CLI/JSON scaffold |
| 3 | Sep 14 | Registry YAML, pipeline engine, linear and spectral executors |
| 4 | Sep 15 | Manifold and neighbour-embedding executors; subprocess isolation, timeouts |
| 5 | Sep 16 | Metrics battery, rank, pre-registered weighting, plan validator |
| 6 | Sep 17 | Viz house style, size-adaptive rendering |
| 7 | Sep 18 | The five skills, /analyze, decisions.jsonl |
| 8 | Sep 19 | Report template + pandoc render; first end-to-end run, dataset 1 |
| 9 | Sep 20 | Fix what day 8 broke; clean run on dataset 1 |
| 10 | Sep 21 | End-to-end on dataset 2 (large) |
| 11 | Sep 22 | GPLVM, timeboxed |
| 12 | Sep 23 | Rules-planner hedge + ablation run |
| 13 | Sep 24 | Agent-behaviour tests, ADR, README, CONTEXT.md |
| 14 | Sep 25 | Final graded runs, reports, submit |

Synthetic fixtures land on day 1 and become the daily smoke test: every day ends with
the full pipeline running on toy data in under a minute.

Cut order if the schedule slips: GPLVM first, then the hedge. Never the tests, never
day 14.

---

## Decision log

- **2026-09-12** — All of the above settled in a design interview before any code was
  written: 40 questions across 9 rounds, no implementation until the frontier was
  empty.

- **2026-09-12** — `datafold` dropped; Diffusion Maps will be implemented directly.
  datafold 1.0.0 imports `sklearn.utils._message_with_time`, a private symbol removed
  from modern scikit-learn, so it fails at import against 1.9.1. The alternatives were
  pinning scikit-learn backwards — which would cascade through umap-learn and scanpy —
  or writing the method out: kernel, density normalisation by alpha, row-normalise,
  eigendecompose, scale the coordinates by the diffusion time. That is about sixty
  lines and no dependency, and it revises section 7's "nine library-backed methods" to
  eight plus two written here. This is the one place where "use libraries wherever
  possible" loses to the libraries not working.

- **2026-09-12** — torch installs as `2.14.0+cpu` from PyPI on Windows; the RTX 3060 Ti
  goes unused. Left as is. GPLVM is capped at a few thousand points by its own O(n^3)
  cost, where CPU is adequate, and a CUDA build is a 2.5 GB download to accelerate the
  one method most likely to be cut. Revisit only if day 11 finishes early.
