# dr-agent — design notes

Running record of decisions and their rationale. Raw material for the manually
written 4-page `report.pdf`. Append as decisions are made; do not rewrite history.

Fifteen work days against a 2026-09-28 deadline. Days are indexed rather than dated:
the build has run ahead of the calendar, and the index is what the schedule actually
tracks. It was fourteen until day 8, which grew a day rather than spending the hedge.

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
pins every rationale to keys that must resolve in `profile.json` or `metrics.json`;
`log-decision` refuses a citation that points at nothing, and records what each key
resolved to beside it.

What that buys is **citation integrity**: no rationale can rest on a number the run
never computed, and a reader can follow every claim back to the artefact it came
from. It is not proof that a rationale is true. A real key with a false reading
passes — a silhouette of 0.7 is 0.7, and "confirms distinct biological cell types"
is not thereby supported — and nothing binds `chosen` to an outcome the toolbox
executed. The earlier claim here, that the agent "structurally cannot claim a
decision it did not make", was more than key validation can carry, and day 5's rank
falsehood was that gap already occurring. Worth having and worth claiming; the
report should claim this and not more.

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

One optional re-plan is permitted once the portfolio has been attempted. Fully
iterative planning was rejected as unbounded in cost. The trigger is an attempt
rather than an evaluation, because a run where every candidate failed has nothing to
evaluate and would otherwise be offered the round forever — and a run with no
successes is exactly the one that needs to re-plan. The round is a portfolio that
*grew*: replacing a candidate the agent gave up on is the diagnose-and-retry of
section 4, not a new round.

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
The agent gets exactly **one** diagnose-and-retry per candidate before recording a
permanent failure and continuing with the rest. Per *candidate*, not per method:
section 3.2 settled the pipeline as the unit of comparison, and two candidates may
legitimately end in the same method — the mandatory PCA baseline alongside a PCA on
variable features, say — so a per-method allowance would let the first consume the
second's.

Not hypothetical: spectral methods die on disconnected neighbourhood graphs, MDS
exhausts memory, numba throws version errors, and anything on a large matrix can hang.

A report sentence like *"Laplacian Eigenmaps failed on the first attempt due to a
disconnected k-NN graph; the agent increased n_neighbors to 30 and succeeded"* is the
most convincing single piece of evidence of agency the system can produce.

`--budget fast|standard|thorough` declares two resources: the wall-clock one candidate
may spend, and how many candidates the run may ever register — 8, 7 and 6, shrinking as
the time allowance grows, because a longer leash per candidate buys fewer of them. This
turns "too big for Isomap" from an implicit constraint into an explicit resource the
agent reasons about and allocates. The budget also controls whether the hyperparameter
sweep runs.

The second cap is what bounds a run. Candidate ids only ever grow — a registered
candidate cannot be dropped, since one that ran and lost is part of the record — so the
size of the plan caps total compute at attempts x ceiling. Without it the id set is the
one quantity the agent can spend that nothing declares, and it is exactly what mints a
fresh per-candidate allowance.

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

**In:** PBMC3k and PathMNIST. The registry's ten reductions: nine library-backed —
PCA, Kernel PCA, Sparse PCA, MDS both metric and non-metric, Isomap, LLE with its four
variants under one op, Laplacian Eigenmaps, t-SNE, UMAP — plus Diffusion Maps, written
directly after `datafold` proved unusable against modern scikit-learn.

**Out, and named as future work in the report:** datasets with missing values, for
which see the day 8 decision below; a minimal MAP-GPLVM in torch, which
was first on the cut order and is what day 7's contract work spent; ensemble/consensus
embeddings; a cross-run experience store (with two datasets the prior would be n=2,
worse than no prior, and it would introduce hidden state that breaks reproducibility);
CI.

**Deliberately deferred** — to be decided when reached, mostly *by the agent*:
PathMNIST subsampling policy, whether PBMC3k gets derived Leiden reference labels,
dataset caching and `.gitignore` handling.

---

## 8. Deliverables

Public GitHub repo as the front door, and **installing the plugin is the route**, not a
fallback: `.claude-plugin/plugin.json` at the repo root, the five skills under `skills/`
and `/analyze` under `commands/`, so an install delivers them. Working in a clone stays
possible and is the convenience, not the path most users take.

`.claude/` holds the build configuration and nothing else. The skills were first written
there, which conflated the product with the instructions to whoever is building it — and
the symptom was immediate: the five skills appeared in the building session's own skill
list.

Installing the plugin delivers the prose, not the toolbox. `drtools` is a Python console
script and installs separately, so `/analyze` checks for it before anything else and says
what is missing rather than letting the agent meet a shell error and improvise.

Project-local `.venv` with a curated pinned `requirements.txt`.
Docs kept lean: `README.md`, `CONTEXT.md`, this file, and one ADR. No prose `docs/`
tree: an architecture page would be a third copy of what the README, the four-page
report and these notes already carry between them.

Submitted: source, `generated_report_1`, `generated_report_2`, and a manually written
`report.pdf` of at most four pages.

---

## 9. Schedule

| Day | Work |
|---|---|
| 1 | Env, pinned deps, repo skeleton, these notes, synthetic fixtures |
| 2 | Loader contract, profiler, recon pass, CLI/JSON scaffold |
| 3 | Registry YAML, pipeline engine, linear and spectral executors |
| 4 | Manifold and neighbour-embedding executors; subprocess isolation, timeouts |
| 5 | Metrics battery, rank, pre-registered weighting, plan validator |
| 6 | CONTEXT.md; viz house style, size-adaptive rendering |
| 7 | Contract repairs: ingestion identity, plan registration, evaluation protocol |
| 8 | The toolbox surface the skills need: status, budget, retry accounting, log-decision, suggest-base; the candidate ceiling; adapter provenance |
| 9 | The five skills, /analyze |
| 10 | Report template + pandoc render; first end-to-end run, dataset 1 |
| 11 | Fix what day 10 broke, plus the queued defect lists; clean run on dataset 1 |
| 12 | End-to-end on dataset 2 (large) |
| 13 | Rules-planner hedge + ablation run |
| 14 | Agent-behaviour tests, ADR-0001, README polish, CONTEXT.md review |
| 15 | Final graded runs, reports, submit |

Synthetic fixtures land on day 1 and become the daily smoke test: every day ends with
the full pipeline running on toy data in under a minute.

The cut order was GPLVM first, then the hedge. GPLVM is spent, on day 7's contract
work. Day 8 then overran too, and took a fifteenth day rather than the hedge: the
calendar had the room, and the hedge is worth more than a day — it is both the
ablation and the working system for a grader without Claude Code. The hedge is still
what a further slip costs. Never the tests, never the last day.

---

## Decision log

- **Day 1** — All of the above settled in a design interview before any code was
  written: 40 questions across 9 rounds, no implementation until the frontier was
  empty.

- **Day 1** — `datafold` dropped; Diffusion Maps will be implemented directly.
  datafold 1.0.0 imports `sklearn.utils._message_with_time`, a private symbol removed
  from modern scikit-learn, so it fails at import against 1.9.1. The alternatives were
  pinning scikit-learn backwards — which would cascade through umap-learn and scanpy —
  or writing the method out: kernel, density normalisation by alpha, row-normalise,
  eigendecompose, scale the coordinates by the diffusion time. That is about sixty
  lines and no dependency, and it revises section 7's "nine library-backed methods" to
  eight plus two written here. This is the one place where "use libraries wherever
  possible" loses to the libraries not working.

- **Day 1** — torch installs as `2.14.0+cpu` from PyPI on Windows; the RTX 3060 Ti
  goes unused. Left as is. GPLVM is capped at a few thousand points by its own O(n^3)
  cost, where CPU is adequate, and a CUDA build is a 2.5 GB download to accelerate the
  one method most likely to be cut. Revisit only if day 11 finishes early.

- **Day 2** — Settled four things the interview had left implicit.

  *The contract admits sparse matrices.* A 2,700-cell count matrix is 707 MB dense and
  8 MB sparse, and the sparsity is itself evidence the planner must reason about —
  densifying at load time would destroy it before the agent ever saw it. Densification
  therefore becomes a recorded preprocessing decision rather than a silent default.

  *Loaders never guess.* Missing values, non-numeric columns and string labels are
  rejected with a message saying what to do instead. Imputation is a decision the agent
  must make explicitly and log; a loader that quietly filled in means would put an
  unrecorded modelling choice underneath every result.

  *Evidence keys are artefact-rooted, and absence is not null.* Citations read
  `profile.shape.n_samples`, not `shape.n_samples`, so they resolve identically from
  the decision log, the report, or a test. `resolve_evidence` returns a `MISSING`
  sentinel for keys that do not exist, because a measurement that is legitimately null
  — an undefined ratio, an unestimable dimension — is a real finding, while a citation
  pointing at nothing is a broken rationale. Collapsing the two would hide the exact
  failure the mechanism exists to catch.

  *The profile carries interpretation, not just measurement.* Alongside the numbers it
  emits `observations`: statements tying a measurement to what it implies for the
  analysis, each citing the keys behind it. "sparsity: 0.87" is something the planner
  must interpret; "87% of entries are zero and sample totals vary 24-fold, so raw
  Euclidean distance mostly measures sequencing depth" is something it can act on.

- **Day 2** — Reconnaissance measures a *probe representation*, not the raw matrix,
  and says so. A k-NN graph over raw UMI counts measures library size, not biology. The
  representation is chosen by a fixed documented rule (counts → normalise-total + log1p;
  feature scales spanning >100× → standardise; otherwise as given) and reported with the
  results, so the agent knows what its evidence is conditional on. Where the transform
  changes anything, the spectrum is computed on *both* raw and probe: how much structure
  moves under normalisation is the cheapest available check on whether preprocessing
  matters for this dataset. Intrinsic dimension and the neighbourhood graph are measured
  on the leading 50 components of the probe, which is what makes them affordable and is
  what any real pipeline does — also declared rather than assumed.

  Validated against the fixtures, which is the point of having built them first:

  | fixture | two-NN (truth) | graph components | spectrum elbow (truth) |
  |---|---|---|---|
  | swiss_roll | 1.98 (2) | 1 | 2 (2) |
  | s_curve | 2.94 (2) | 1 | 2 (2) |
  | blobs | 16.3 | 5 (5 clusters) | 4 |
  | linear_subspace | 6.04 (5) | 1 | 5 (5) |
  | sparse_counts | 24.2 | 1 | 16 |

  The small-sample UMAP thumbnail named in section 3.1 is deferred to day 6, when the
  viz module exists to render it. Claude can read an image, so the thumbnail is evidence
  the agent can genuinely look at rather than a figure for the report only — worth doing
  properly rather than half now.

- **Day 3** — The registry declares thirteen ops and a test asserts that the
  set of declared ops and the set of implemented executors are *equal*. An entry with no
  executor is a method the planner can select and never run; an executor with no entry is
  a capability the planner cannot see. Adding a method stays a registry entry plus one
  function, with no skill edits, which is the reusability claim made concrete.

  Parameter resolution records provenance — `specified` against `registry_default` — for
  every value. "The agent set `n_neighbors` to 30 because density varied 40-fold" and
  "the agent left `n_neighbors` at the default" are very different claims for a report to
  make, and the difference should not rest on anyone's memory. Unknown parameter names
  are an error rather than a silent no-op, since a plan setting `perplexity` on Isomap
  has misunderstood something, and swallowing it would leave the report describing a
  setting that never took effect.

  Structural plan checks run before any compute: a terminal method cannot feed another
  stage, and a candidate must end in a reduction. Each refusal says what to do instead,
  because the agent repairs its plan from these messages.

- **Day 3** — The diffusion-maps bandwidth default was wrong, and the way it was
  wrong is worth keeping.

  The obvious heuristic is the median squared distance to the k-th neighbour. It passed
  a first check at n = 700 with the Swiss roll recovered at rho = 1.00, then failed at
  n = 600 with rho = 0.74 and at n = 400 with rho = 0.16. The cause is that the
  heuristic scales with sampling density while the gaps a manifold method must *not*
  bridge do not: on a sparsely sampled roll the seventh neighbour is far enough away
  that the kernel reaches across adjacent sheets and diffusion short-circuits between
  them. A bandwidth sweep confirmed the working range is eps in 0.5 to 4 and barely
  moves with n, collapsing to rho < 0.4 by eps = 8.

  The first replacement — Coifman and Singer's kernel-sum scaling criterion, taking eps
  at the steepest point of log S(eps) against log eps — was *worse*, choosing eps near
  33 at every n. It finds the right scaling regime, which is why its slope gives a good
  intrinsic dimension estimate (2.2 against a true 2), but the steepest point sits at
  the coarse end of that regime.

  What works is the same criterion read differently: the *lower edge* of the linear
  regime, the smallest eps whose slope has reached half the maximum, where the kernel
  has just begun to resolve the manifold rather than isolating every point. That gives
  rho >= 0.98 for both the Swiss roll and the S-curve across n from 300 to 1500, and
  reports the implied dimension as a free cross-check on reconnaissance's two-NN
  estimate.

  The lesson generalises past this one method: a default validated at a single sample
  size is not validated. The parametrised regression test now sweeps n rather than
  fixing it.

- **Day 4** — The registry reaches eighteen ops and ten reductions. LLE is
  one op with a `method` parameter over its four variants rather than four ops: they
  share every precondition and differ only in how the local problem is posed, so the
  planner's real decision is *which variant*, which is a parameter choice. The registry
  documents what each variant buys.

  Each of these methods has a precondition that fails quietly, so each executor checks
  it and reports in terms of the plan. Isomap on a disconnected graph would otherwise
  get infinite geodesics that scikit-learn patches over, leaving an embedding that looks
  fine and means nothing. Hessian LLE with too few neighbours raises from inside a
  least-squares solve, naming no parameter the plan actually set. t-SNE with the default
  perplexity of 30 at n = 60 produces a featureless disc that still plots happily. All
  three now refuse, and name the parameter to change.

- **Day 4** — Candidates run in a subprocess with a wall-clock cap.

  The reason is that only one of the three ways this goes wrong is an exception. A
  method can raise, which is catchable. It can allocate more than the OS will give and
  take the interpreter down, which is not. Or it can simply not finish — SMACOF on fifty
  thousand points does not fail, it runs until someone stops it. A killable process
  turns all three into a record: `ok`, `failed`, `timeout`, or `crashed`, where the last
  is synthesised by the parent from the exit status because the child died before it
  could write anything.

  Failure records name the stage and the parameters it ran with, and keep the underlying
  exception's type — `MemoryError` and `LinAlgError` call for different repairs. A
  traceback tells a developer where in a library something surfaced; this tells the agent
  which stage of its plan failed and how it was configured, which is what it needs to
  revise.

  The dataset is cached into the run directory once so candidates need not each reload
  it, which also makes a run self-contained: the artefacts describe an analysis of a
  matrix that is still there to inspect.

- **Day 4** — Measured LLE's sensitivity to `n_neighbors` rather than assuming it.
  On a noise-free Swiss roll at n = 1000, every variant recovered the roll parameter at
  rho = 1.00 for k between 6 and 12, fell to about 0.9 at k = 16, and collapsed to
  between 0.01 and 0.46 by k = 24. Once a neighbourhood spans two folds of the roll,
  "locally linear" is false and the reconstruction weights stop meaning anything. That
  measurement is now in the registry as guidance to the planner and in the test suite as
  a regression, since it is a property of the method rather than a bug.

  It also explains an earlier confusion: a first check at n = 700 with noise gave LLE
  about 0.55 and looked like a defect. It was the marginal zone, not a defect. Isomap
  reached rho = 1.000 on the same data and MDS 0.24, the latter being the negative
  control that makes the former mean something.

- **Day 4** — Two library-compatibility findings.

  UMAP fails on `scipy.sparse.csr_array` while working on `csr_matrix`. Its inner loops
  are numba-compiled and numba understands only scipy's legacy sparse matrix types; the
  newer sparse array — which scipy's own operations increasingly return — produces a
  type-inference error mentioning "non-precise type pyobject" and nothing about
  sparsity. The executor converts, which is free, and keeps the registry's
  `handles_sparse` claim honest. Worth remembering because this will bite again anywhere
  numba meets scipy sparse.

  scikit-learn's MDS has renamed `metric` to `metric_mds` and is changing its default
  `init` in 1.10. Both now passed explicitly, which also removes a source of run-to-run
  variation.

- **Day 5** — The planning loop closes: `validate-plan`, `evaluate`, `rank`.

  *Metrics are scored on an absolute scale, not min-max across candidates.* Min-max
  always awards 1.0 to the best candidate, so a field of uniformly poor embeddings would
  produce a winner that looks excellent. Runtime is the sole exception, because two
  seconds is fast or slow only relative to the alternatives; that difference is recorded
  rather than glossed.

  *Where a ceiling exists it is computed and reported.* If the reference representation
  only reaches 0.62 label agreement among neighbours, an embedding at 0.58 has lost
  almost nothing, and reporting 0.58 alone invites the opposite conclusion.

  *A metric available for only some candidates is dropped for all of them.* Scoring it
  where convenient would make the comparison unfair, and silently dropping it would
  change what the weighting means without saying so. When weight is redistributed, the
  ranking says so and says how much.

  *Failed candidates are excluded rather than scored as zero.* Zero would rank the
  method; what failed was one configuration of it.

- **Day 5** — The plan validator *simulates* the plan rather than pattern-matching
  on it. Walking the stage list while tracking sample count, feature count, sparsity and
  whether the values are still raw counts means it knows what each method will actually
  receive. That is what distinguishes "Isomap on 107,000 points", which is hopeless, from
  "subsample to 3,000, then Isomap", which is the correct way to do it — where a rule
  keyed on the dataset's size alone would reject both. Both cases are in the test suite
  precisely because getting that distinction wrong is the easy mistake.

  One hard requirement: every plan must include a candidate ending in plain PCA. Without
  a linear baseline there is nothing to measure the nonlinear methods against, and the
  claim that the data needs a manifold method cannot be supported — if PCA does as well,
  the extra machinery bought nothing.

- **Day 5** — Two defects the first end-to-end run exposed, both worth recording.

  `prepare-reference` failed silently. Base preprocessing is by construction a stage
  list containing only preprocessing, but `run_pipeline` required every stage list to end
  in a reduction, so the command exited non-zero and my driver script ignored the status.
  Evaluation then fell back to the raw cached matrix, and the numbers were wrong in a way
  that looked plausible: `pca_umap` scored 0.696 on trustworthiness against the raw counts
  and 0.821 against the correct base-preprocessed reference. Nothing errored. The lesson
  is about the shape of the mistake rather than the fix — a reference that silently
  defaults to the wrong thing produces a full set of believable numbers, so the artefact
  now records which reference it used, and evaluation says so in its output.

  The label-preservation note asserted that an embedding "retained 117% of the label
  structure", which is nonsense phrasing for a real phenomenon. An embedding can beat its
  own input on neighbourhood label agreement: in high dimensions distances concentrate,
  so the reference's own neighbourhoods are noisy, and a reduction that discards the
  noisy directions recovers structure the full-dimensional space obscured. The note now
  explains that rather than dividing through, and says the reference is a weak baseline
  for such a dataset rather than a ceiling.

- **Day 6** — `CONTEXT.md` pulled forward from day 13, and a term collision resolved
  before it could set into prose.

  The reason for moving it is that days 7 and 8 write the five skills, which are prose
  the agent reads *at runtime* to make decisions. A glossary written after them would
  document whatever vocabulary the skills had drifted into rather than governing it.

  The collision: `probe representation` (reconnaissance), `base preprocessing` (the
  plan) and `reference` (the metrics) all name "the transform applied before the real
  work", and nothing said how they relate. They are now distinguished on purpose rather
  than by accident. A probe representation exists so that *measurements* mean something,
  is chosen by a fixed published rule, and is discarded once the measuring is done — it
  never produces an embedding. Base preprocessing is part of the *analysis*, is chosen
  by the agent, and its output survives as the reference. So they are genuinely two
  things, and collapsing them would have been the wrong fix.

  What the distinction exposes is a connection worth building: both answer the same
  question — what transform makes distances on this data meaningful — by different
  mechanisms. Reconnaissance's rule is therefore the natural *default suggestion* for
  base preprocessing, which the planner accepts or overrides with a logged reason.
  Deferred to day 7 rather than done here, since it changes the planning skill's
  behaviour rather than the vocabulary.

  Also settled: no prose `docs/` tree. An architecture page would be a third copy of
  what the README, the four-page report and these notes already carry between them.
  `docs/adr/` is created on day 13 for ADR-0001 alone.

- **Day 6** — The figure house style, and a colour constraint that changed the design.

  The categorical palette was checked with a validator rather than by eye, and the
  result forced a decision. For scatter-like forms — where any two colours can end up
  adjacent — only **three** slots clear the separation floors. At eight slots the worst
  pair measures Delta E 7.1 to normal vision and 3.2 under simulated protanopia: red and
  orange that nobody can reliably tell apart. PBMC3k will have roughly eight clusters and
  PathMNIST has nine classes, so the conventional nine-colour UMAP was not available.

  Above three classes, identity is therefore carried by **a printed class name at each
  centroid**, and by the class facet — one panel per class, one series per panel, where
  the colour question does not arise at all. Verified at the hard case: 60,000 points and
  nine classes renders as a legible density field with nine named clusters. The
  single-cell convention is a nine-colour scatter; the convention is not readable and a
  label is. Expect a reader to ask why the figures are not colour-coded, so the reason
  belongs in the four-page report.

  Rendering adapts to sample count in three regimes — ringed markers to 2,000, small
  translucent rasterised points to 20,000, a hexbin density field above that — and the
  chosen regime is recorded in the figure's metadata so a reader knows whether they are
  looking at points or at a histogram.

- **Day 6** — Two figure bugs, the second of which I caused while fixing the first.

  *Outliers destroyed the view.* A collapsed diffusion map puts almost every point in one
  spot and a handful decades away, and on full extent that renders as an empty panel with
  two dots — indistinguishable from a broken figure. Views are now clipped to a high
  quantile, and the number of points left outside is printed on the panel so nothing is
  quietly cropped.

  *Then the labels left the building.* Class labels were positioned by a repulsion pass
  measured in full data span, while the axes showed the clipped view — so for the
  collapsed embedding they were pushed enormous distances outside the axes, and
  `bbox="tight"` expanded the canvas to reach them. The saved figure went from 1842 x 489
  to 8849 x 6722: four tiny panels marooned in white space. Labels are now positioned
  within the clipped view, clamped inside it, and drawn with clipping on. There is a
  regression test asserting the figure stays under 1200px when a point sits at 1e6.

  Both were caught by looking at the rendered file. Neither would have been caught by a
  test of the plotting code, and the second was introduced by the fix for the first —
  which is the argument for rendering and looking every time, not only when something
  seems wrong.

- **Day 6** — The dataset is cached into the run on first contact rather than at first
  embed. `prepare-reference` needs the matrix and can legitimately run before any
  candidate has, so `profile` and `recon` — whichever touches the data first — now write
  the cache. This is what the run directory being self-contained was supposed to mean:
  every later stage reads the matrix from the run rather than from the source.

- **Day 6** — Candidate panels are ordered by rank rather than by name. Alphabetical
  order put the winner wherever its id happened to fall.

- **Day 6** — An external review of everything written so far, and the decision to fix
  it on day 9 rather than now.

  The whole tree was put through Codex's reviewer as a single diff against the root
  commit — 43 files, 9,290 lines, the only excluded file being `LICENSE`. It returned
  nine defects. All nine are real: six reproduce as outright failures, and I confirmed
  the other three by reading. None of them is missing work. There are no stubs and no
  `TODO`s in `drtools/`, and the 178 tests pass; every defect sits in a finished path
  the tests do not exercise.

  The useful split is not by severity but by whether the thing *announces itself*.

  *Four crash.* Small-dataset evaluation raises rather than scoring, because the
  neighbour clamp allows `k` up to `n-1` while `trustworthiness` demands `k < n/2` —
  the default `k=15` fails for every dataset of 30 rows or fewer. A sparse reference
  saved with `--stages '[]'` writes a scalar object array that cannot be loaded back.
  Candidate indices are applied to a reference that base preprocessing already
  subsampled. The comparison figure hands the first candidate's labels to every panel.
  These four would have surfaced on day 8 anyway, which is what day 8 is for.

  *Four are silent, and they are the reason this was worth running.* When the base
  subsample happens to be large enough to contain every candidate index, the reference
  is subset by the wrong rows with no error at all. Equally sized candidates drawn from
  different samples receive each other's labels. A reused candidate id deletes only its
  outcome JSON, so `rank` — which prefers a metrics file over a failure record — can
  rank a failed retry on the previous run's numbers. A CSV with missing labels
  factorizes to `-1`, passes the integer-label contract, and is plotted as the last
  named class. Day 9 fixes what day 8 *broke*; none of these break anything visibly, so
  they would have travelled intact into the day 14 graded runs.

  *One is deployment-only.* `registry.yaml` is not in the built wheel — package
  discovery finds the modules and nothing includes the data file — so `load_registry()`
  raises outside a source checkout. Invisible here because the venv is editable.

  Deferred to day 9 rather than fixed on the spot. Day 9 already exists for exactly this
  and the fixes want the end-to-end run of day 8 to check them against; fixing blind
  today would mean touching the same paths twice. What changes is day 9's brief: it was
  "fix what day 8 broke", and it is now that plus a list that day 8 will not produce on
  its own. The four silent ones need regression tests, not just repairs — each was
  invisible precisely because nothing asserted on it.

  Rejected: running the fixes now. Also rejected: treating the clean test suite as
  evidence of health. 178 passing tests coexisted with nine real defects, because the
  tests exercise each unit on the shapes it expects and none of these defects live
  there — they live where two components disagree about what a row index means.

- **Day 6** — The four commitments of section 2, attacked deliberately, and what survived.

  The defect review above answers "is the code right". It cannot answer "is the design
  right", so the same tree went through a second, adversarial pass aimed squarely at
  2.1–2.4 — while they are still cheap to change. Days 7 and 8 turn this CLI surface
  into five skills and a report template; after that, a contract change is a rewrite of
  everything built on it. The verdict was *needs-attention*, on all four commitments.
  I verified every finding, and reproduced two as live failures.

  **2.4 pre-registration is not currently enforced, and the log says otherwise.** This is
  the serious one. Nothing marks the moment a plan becomes registered: `embed` takes
  `--stages` directly without reference to a plan, and `rank` reads whatever
  `plan.json` holds at the instant it runs. I edited the weights *after* the metrics
  existed and re-ranked: the winner flipped from `tsne2` to `pca2`, no amendment was
  written, nothing warned. The decision record for that rank reads *"weights were
  declared in the plan before any embedding was computed"* — the fallback rationale at
  `cli.py:462`, asserting as fact the exact guarantee it had just broken. The
  hallucination-control mechanism is presently emitting the falsehood itself. Chronology
  cannot be inferred from a file existing; it needs a registration event the toolbox owns.

  **And fixing the weights would not be enough.** `EvaluationSpec` carries `weights` and
  `justification` under `extra="forbid"`, so the schema cannot express `k`, the seed, or
  the sample cap — while `evaluate` accepts `--k` and `--max-samples` as free flags. I
  scored one candidate at k=5 and another at k=30 and ranked them together; the ranking
  mentions no protocol, no comparability, no caveat. Freezing the weights while leaving
  the measurement adjustable freezes the wrong half. What has to be pre-registered is the
  whole evaluation protocol, not the weighting of it.

  **2.1's escape hatch is wider than one loader.** A CSV with a missing measurement makes
  three of our own files mutually unsatisfiable: the table loader refuses to impute
  ("a preprocessing decision for the agent to make explicitly"), the contract demands
  missingness be "resolved or explicitly encoded by the loader", and the registry has
  eighteen ops and no imputation among them. The only path left open is an adapter
  quietly doing it, after which audited metrics treat fabricated zeros as observations.
  `nan_to_num` passes the contract with nothing recorded; `meta` requires only `name` and
  `source`, so there is nowhere for provenance to live. The contract also accepts
  `labels[::-1]` — equal length is not alignment. Loading is not the only place an unseen
  dataset bites, and imputation is the counter-example that proves it.

  **2.2 holds files, not identities.** Re-running `profile` in an existing run after an
  adapter repair writes a profile of the new matrix while `ensure_cache` keeps the old
  one: measured at 4 features in `profile.json` against 8 in the cache. Planning reads
  the profile, execution reads the cache, and the run id says they agree. Same root cause
  as the `ensure_cache` defect above, but the blast radius is the planning input rather
  than one command. A shared directory is not a shared dataset without an identity
  covering matrix, labels, adapter and options.

  **2.3 was overstated rather than wrong.** No evidence resolver exists yet and the
  report generator is day 8, so nothing here is broken code. But the claim that citation
  pinning means the agent "structurally cannot claim a decision it did not make" is more
  than key validation can carry. A real key with a false reading passes — silhouette 0.7
  *is* 0.7, and "confirms distinct biological cell types" is not thereby supported — and
  `log_decision` validates nothing, so `chosen` is bound to no executed outcome. The
  mechanism buys citation integrity. That is worth having and worth claiming; it is not
  proof of a rationale, and the report should not imply it is. The rank falsehood above
  is this gap already occurring.

  What survives intact: file-based state and deterministic scoring over a declared
  weighting are sound, and picking weights after fixed profile and recon probes is
  legitimate exploratory planning. The problem is never that the agent chooses; it is
  that nothing currently stops it choosing twice.

  **Day 7 is therefore contract work first, skills second.** Ingestion identity, a
  registration transition, and a registered evaluation protocol. Writing the skills
  against contracts known to be broken would mean writing them twice, and every one of
  these failures is silent — a grader who edits weights and re-ranks gets a system that
  certifies its own pre-registration, and the run looks clean. The schedule cost is real
  and lands on the cut list: GPLVM goes first, as already agreed.

  Rejected: carrying on to day 7 as scheduled and repairing contracts after the skills
  exist. Also rejected: treating 2.3 as a defect — the overstatement is in the notes, and
  the fix is to describe the mechanism accurately rather than to build something larger
  than the deadline allows.

- **Day 7** — The three contracts, and what it took to make them hold.

  Days 5 and 6 left three guarantees the design claims and the code did not enforce.
  All three are now enforced, and each was verified by re-running the reproduction that
  found it rather than by trusting a passing suite. The suite went from 178 to 272.

  *Dataset identity.* `ensure_cache` verified instead of trusting, and `profile`, `recon`
  and `embed` stopped reloading the source when the run already holds a cache. Identity is
  a content digest of the matrix and label codes. Sparse input is canonicalised first, and
  `indices`/`indptr` are hashed at fixed width — measured here, one matrix built with
  unsorted indices, with a stored zero, or with int64 rather than int32 indices gives three
  different digests, and index width is storage, not data. Storage *kind* does stay part of
  identity: normalising sparse against dense would mean densifying to hash, 707 MB for
  pbmc3k, which defeats the representation the cache exists to preserve.

  *Pre-registration.* `validate-plan` is the registration event; `rank` reads the frozen
  copy. The false rationale that asserted "weights were declared in the plan before any
  embedding was computed" is deleted — it was a fallback string the toolbox emitted while
  the guarantee was being broken. Two bypasses were found only by attacking the fix. Running
  `validate-plan` a second time re-registered unconditionally, so the whole contract fell to
  one extra command; and copying `plan.json` over `plan.registered.json` satisfied the
  digest check, which the refusal message had all but suggested by saying "restore the
  registered plan". The freeze now refuses weight changes, base-preprocessing changes and
  candidate removal, and `rank` cross-checks the registration against the last
  `register_plan` record in the decision log. The log is append-only and predates every
  embedding, so it is the authority and the file is not.

  *Comparability.* The neighbourhood is chosen by published rule from the Reference's row
  count, `k = max(1, min(15, ceil(n/2) - 1))`, which holds against scikit-learn for every
  n >= 3 with the boundary tight at 30 and 31. `--k` and `--max-samples` are gone from
  `evaluate`; `k` is a required argument of the battery with no default, because an optional
  override leaves the incomparability available to the next caller. A candidate that
  subsampled below what the registered k can support is refused, naming the subsample,
  rather than quietly scored at a smaller k.

  The retry loop survives all of this, which was the constraint that shaped the freeze.
  A candidate that failed, timed out or crashed can still be diagnosed, revised,
  re-registered and re-run; one that succeeded is frozen and must be re-registered under a
  new id. Invalidation happens before every attempt rather than only when stages change,
  because the commonest retry is the same stages with a bigger budget — reproduced, that
  left a timed-out candidate ranked on the scores of the run before it.

  Rejected: warning instead of refusing, which would have put a caveat in the report where
  a guarantee belongs. Rejected: implementing Amendment now — it stays a defined term with
  no mechanism, and every refusal that would need one says so plainly.

  Two things worth recording because they were only caught late. Five of the nine tasks
  had a fix that reintroduced the class of defect it was closing, which is why every task
  got an independent review. And the mixed-k hole survived all nine of those reviews: the
  reference recorded `settings.k`, `evaluate` read it, and nothing passed it on. Each task
  matched its own brief, so only the whole-branch review could see it. The plan contained
  the hole, not the implementations.

  Day 9 inherits: `rank`'s status check reads the artefact rather than the decision log,
  `prepare-reference` has no freeze, `write_cache` converts twice, `_invalidate_candidate`
  globs an unsanitised candidate id on a delete path, and `recon` still carries its own
  `--k` so the agent can still tune the evidence that justifies its own plan.

- **Day 7** — The schedule re-indexed, and GPLVM spent.

  Day 7 was briefed as the three contracts *and* the five skills. The contracts took the
  day on their own, which the day 7 design had said outright they would. The skills move
  to day 8 and everything after shifts by one.

  That shift needs a slot, and the cut order settled on day 6 supplies it. GPLVM is
  therefore spent rather than merely at risk, and section 7 moves it to future work. What
  a further slip would cost is the hedge, which is worth more: it is the ablation showing
  where the agent's judgment beat a rule engine, and it is how a grader without Claude
  Code still gets a working system.

  The rescheduled day 10 carries more than "fix what day 9 broke". It inherits the nine
  defects queued on day 6 and the five day 7 left behind: `rank`'s status check reads the
  artefact rather than the decision log, `prepare-reference` has no freeze, `write_cache`
  converts twice, `_invalidate_candidate` globs an unsanitised candidate id on a delete
  path, and `recon` still carries its own `--k`, so the agent can tune the evidence that
  justifies its own plan. The entry above was written before the re-index and calls that
  day nine.

  Rejected: moving day 14, and compressing day 13's tests to keep GPLVM. Both were
  foreclosed on day 6 — trading a deliverable the report can demonstrate for one method it
  could only mention is the wrong direction on a rubric that rewards reproducibility.

- **Day 7** — Day 7 lifted from an archived branch, day 8 left on it.

  Days 7 and 8 were attempted once already, on `archive/day-7-8`, and the attempt was
  interrupted partway through day 8: the planning ceremony outgrew the work it was
  planning, and the task stopped being tractable. Both days were parked on that branch
  rather than abandoned, which is why the record of them is not in this file.

  Day 7 is taken. Its five commits — the design, the revision the adversarial pass forced,
  the implementation plan and its one restructuring, and the implementation — are
  cherry-picked onto main, and the 272 tests pass here as they did there. What made that
  liftable is that day 7 had been classified architectural and so had a reviewed spec: the
  work arrives with its reasoning attached rather than as a diff to be reverse-engineered.

  Day 8 is not taken. Thirteen further commits on that branch are groundwork the
  interrupted attempt laid down in the toolbox — status derived rather than described, a
  budget with a consumer and a cap that refuses, attempt counting that survives
  interruption, `log-decision` refusing to forge the records the freeze trusts, citation
  resolution, the re-plan round defined by a grown portfolio rather than by a ranking, and
  reconnaissance's rule offered as the planner's base-preprocessing default. They stay
  where they are. A day whose plan exploded is not a day whose output should be inherited;
  those commits are reference for the second attempt, not its starting point.

  Two things follow. The five skills and `/analyze` exist nowhere — `skills/` and
  `commands/` are empty on that branch too, so day 8's actual deliverable was never
  written, and nothing is being redone twice. And the day 6 item deferred to day 7 —
  reconnaissance's rule as the default *suggestion* for base preprocessing, which the
  planner accepts or overrides with a logged reason — is still open, and belongs to day 8.

  Rejected: merging the branch, which would import a partial day 8 whose shape the
  interruption had already judged wrong. Rejected: rebuilding day 7 alongside it, which
  would spend a day re-finding defects that branch's own review had found — five of its
  nine tasks needed a fix that reintroduced the class of defect it was closing.

- **Day 8** — The candidate ceiling, and the loop three correct components made.

  Day 8's toolbox half was lifted from `archive/day-7-8` and reviewed as a whole against
  this file rather than against the plan that produced it, which is the one way a
  plan-level hole is visible. It found one.

  Attempts were capped per candidate id at two. A replacement — add an id, abandon an id
  — was deliberately not counted as a re-plan round, with a test asserting so. And
  nothing capped the id set. Each of those is right on its own; composed, they gave an
  unbounded loop: exhaust an id, abandon it, register a replacement, get two more,
  forever. Section 3.1 had rejected exactly this as "unbounded in cost". The refusal at
  the exhausted boundary completed it, by advising *"Register a new candidate id for a
  further variant"* — true, helpful, and the instruction that resets the allowance.

  The fix caps the registered candidate set by budget: 8, 7 and 6 for fast, standard and
  thorough. The ceiling shrinks as the per-candidate wall-clock grows, which states the
  trade rather than hiding it. It sits strictly above section 3.4's 3-5 portfolio
  guidance, and that gap is deliberate: a ceiling of 5 would bound the loop by forbidding
  the sanctioned repair, destroying the retry this design calls its best evidence of
  agency. The `embed` refusal now computes the room left and only offers a replacement
  when one exists; at the ceiling it names evaluation and the report instead.

  What makes this cheap is that the counter already existed. Re-registration refuses to
  let a registered id disappear — written for record integrity, not for this — so the id
  set is monotone, already digested, and already in the append-only log. The bound is two
  refusals that were already there plus one comparison, and it needs no new trust.

  A plan may also declare `max_candidates` below the budget's ceiling. That enforces
  nothing by itself, since a bound the agent sets on itself is not a bound, but under the
  hard ceiling it is a checkable claim about self-restraint — the same thing
  pre-registering the weighting buys for the evaluation.

  Section 4's wording is amended rather than implemented: it said one retry per *method*,
  which predates section 3.2's settling of the pipeline as the unit. Enforcing it
  literally would newly forbid something the design requires, since the mandatory PCA
  baseline can share a terminal method with another candidate and would have its
  allowance consumed by it.

  Rejected: a pooled per-run attempt budget, which was the first instinct. A pool asks
  the agent to forecast the failure rate of candidates it has not yet run, and it has no
  basis for that forecast — the council found the two opposite failure shapes, hoarding
  and front-loading, which is itself the diagnosis. A per-candidate allowance asks for no
  forecast: "may I retry this?" is always local. Also rejected: lineage tracking through a
  declared `replaces=` field, because an invariant cannot rest on a field whose omission
  is free. And rejected: bounding it in skill prose, which this defect is the argument
  against — the message that taught the bypass *was* prose, and was correct.

  Known limits, stated rather than left to be found. The ceiling bounds a run, not a
  dataset: nothing stops a fresh run, though that is a separate decision log and a
  separate report, so the flailing stays visible. And it bounds count, not breadth — eight
  ids all running the same method is legal. Section 3.4's family-spanning requirement is
  the answer to that and is still unenforced; it is day 10 work, because as an error at
  re-registration it could refuse a legitimate replacement.

- **Day 8** — Four smaller findings from the same review, and one pattern shared
  between two of them.

  *The re-plan trigger.* Section 3.1 said the round comes "after evaluation"; the code
  offers it once anything has been attempted. The code is right and the sentence was
  amended to match: a run where every candidate failed has nothing to evaluate, and is
  exactly the run that needs to re-plan. Reading it literally would have offered the
  round forever to the runs least able to take it.

  *`status` trusted the file.* It enumerated candidates from `plan.registered.json`
  while taking outcomes, attempts and the re-plan round from the decision log. `rank`
  cross-checks the two and refuses, so a rewritten file could never change a ranking —
  but it could change what `status` reported, and `status` is what the agent reads to
  choose its next move, so a forged candidate would have redirected the run long before
  `rank` refused. The registration record already carries the id list; it is now the
  source, with the file kept only as the fallback for a run that never froze a plan.

  *`_ranked` failed open.* A `ranking.json` with no registration recorded was taken at
  its word. `rank` refuses unless the log records a registration, so that state cannot
  be produced by this toolbox — which makes reading it as ranked a way for a
  hand-written file to declare the run finished. It now fails closed.

  *The in-process gate named its own key.* `--in-process` escapes the wall-clock cap
  and is held behind an environment variable, and both the refusal and the `--help`
  text said which variable. That is the same defect as the retry refusal that advised
  registering a new candidate id: correct, helpful, and the one sentence that hands over
  the bypass. Both now name only the legitimate route.

  The limit is worth stating rather than leaving to be found: the variable still exists,
  because the suite needs it, and an agent with a shell could set it. What changed is
  that nothing in the toolbox tells it so. That is a speed bump, not a barrier — the
  barrier would be removing the flag and having the tests reach `run_pipeline` directly,
  which costs the outcome recording the CLI path gives them.

  Two of the four are the same shape, and it is worth naming because it has now
  appeared three times in two days: a refusal that explains the way around itself. A
  message is the surface the agent acts on, so a true sentence in it is an instruction,
  not a note.

- **Day 8** — Missing values are out of scope, and three files now say so.

  Day 6's adversarial pass found the loader, the contract and the registry mutually
  unsatisfiable on a CSV with a hole in it. `_load_table` refused and called imputation
  "a preprocessing decision for the agent to make explicitly"; the contract demanded
  missingness be "resolved or explicitly encoded by the loader"; and the registry had
  no op that imputes. The only route those three left open was an adapter filling the
  holes in silently, after which audited metrics treat fabricated numbers as
  observations and `meta` has nowhere to say otherwise.

  The measurement that settled it: **none of the ten reductions tolerates NaN.** Every
  one raises, and most name it — "PCA does not accept missing values encoded as NaN",
  "NearestNeighbors does not accept missing values". Nor is there a back door: sklearn's
  `nan_euclidean_distances` works and several methods take precomputed matrices, but no
  executor exposes that route — `spectral.py` hardcodes `affinity="nearest_neighbors"`,
  and the registry offers no distance metric for MDS, Isomap or t-SNE.

  So admitting NaN into the contract would have bought exactly one thing: carrying it
  from the loader to an `impute` stage. That reframed the question as *where imputation
  happens* rather than whether — at load, recorded in `meta`, or in the plan, recorded as
  a Stage that is pre-registered, validated, cited and comparable between candidates.

  The plan-stage version is the better design and is not being built. It costs a
  contract change, two executors, a validator rule, and a clause in reconnaissance's
  probe rule, on a day already over its scope, to serve an input neither graded dataset
  has. PBMC3k is a count matrix and PathMNIST is images; neither has missing values.

  What changes is that the three files agree, and agree on refusal. Both messages
  previously named a route — the contract told the loader to resolve it, the loader
  called it the agent's decision — and those sentences were the whole opening. They now
  say the dataset is out of scope and why filling the holes would be worse than
  stopping. Section 7 lists it as future work with the measurement behind it.

  Rejected: required `meta` provenance recording what an adapter did about missingness.
  It is the honest half-measure and merges with the adapter-digest work, but a mandatory
  field is a declaration, not a verification, and this project already has one
  overstated guarantee it had to walk back. A narrower promise kept exactly is worth
  more here than a wider one qualified in a footnote.

  One thing found while measuring, not fixed: `mds.metric` is the metric-versus-
  non-metric flag, a boolean, while `umap.metric` is a distance-function name. Same key,
  two meanings, in a registry the planner reasons over. Day 10.

- **Day 8** — Adapter provenance, the narrowed 2.3 claim, and a fifteenth day.

  The adapter is the one piece of agent-written code in an analysis, and a run recorded
  only its path. That dates badly: `my_adapter.py` describes a matrix produced by
  whatever that file holds when someone later opens it. `meta['adapter']` is now a
  record carrying the path, a sha256 of the source that ran, and its size.

  It is also assigned rather than defaulted. `setdefault` let an adapter supply its own
  `adapter` key and keep it — the one field saying which code produced the matrix was
  writable by that code. This is the toolbox recording what it executed, not the
  adapter describing itself.

  Provenance is deliberately separate from the dataset digest. Day 7 kept adapter source
  out of identity so an adapter could be tidied without invalidating a run; the
  corollary is that identity cannot then answer which code ran, so provenance has to.
  A test pins both halves: the same matrix from an edited adapter keeps its identity and
  changes its provenance.

  *Section 2.3 is narrowed, which day 6 said to do and nothing did.* It claimed evidence
  keys mean the agent "structurally cannot claim a decision it did not make". They do
  not. `log-decision` refuses a citation that resolves to nothing and records what each
  key resolved to, which buys citation integrity: no rationale rests on a number the run
  never computed, and every claim leads back to an artefact. A real key with a false
  reading still passes, and nothing binds `chosen` to an executed outcome. Day 5's rank
  falsehood was that gap occurring. The notes and the README now claim the narrower
  thing, which is the one the mechanism delivers.

  *And the schedule gained a fifteenth day.* Day 8 was re-scoped to the toolbox surface
  the skills need, and then absorbed the candidate ceiling, four review findings, the
  missing-data decision and this. The skills move to day 9 and everything shifts. Paid
  for with a day rather than with the hedge: the calendar has the room, and the hedge is
  worth more than a day, being both the ablation and the working system for a grader
  without Claude Code. It stays what a further slip would cost.

- **Day 9** — The five skills and `/analyze`, and where an agent's own skills live.

  The deliverable that no previous attempt reached. Each skill reads its position from
  `drtools status` rather than from the conversation, so a resumed session restores
  completely, and each ends by handing off to the next.

  What the prose carries is what the toolbox cannot. The refusals already say what is
  wrong and what to do instead, so the skills do not restate them; they carry the
  judgement that does not reduce to a Capability record — that t-SNE and UMAP cluster
  sizes and inter-cluster distances are not meaningful, that a Reference value is a
  baseline an Embedding may exceed, that a failed Candidate indicts a configuration
  rather than a method, and that spanning families is what makes disagreement between
  Candidates informative.

  *They were written under `.claude/` first, and that was wrong twice over.* `.claude/`
  is the build configuration — instructions to whoever is building dr-agent — so the
  product's runtime prose beside it made the two indistinguishable. The symptom arrived
  immediately: the five skills appeared in the building session's own skill list, which
  is not a feature but a namespace collision. And installation, not cloning, is how this
  will actually be used, so the plugin root is where they belong. Section 8 is rewritten
  to say install is the route rather than a "plus".

  That exposed a dependency worth naming: installing the plugin delivers the prose, not
  the toolbox. `drtools` is a console script from the Python package, and every skill
  calls it. `/analyze` now checks for it first, so the likeliest first-run failure is a
  sentence rather than a shell error the agent improvises around.

  *`CONTEXT.md` gained three terms, and the reason is the day 6 lesson repeating.*
  Writing the skills in the glossary's vocabulary surfaced that day 8 had introduced
  three concepts the prose leans on and the glossary never carried: **Portfolio**, the
  Candidates a Plan registers; **Ceiling**, the largest Portfolio a Run may register;
  and **Attempt**, one execution of a Candidate. Two of them collided with words the
  glossary explicitly told the skills to avoid — `ceiling` under Reference value,
  `allowance` under Budget — which is what made the gap visible. "Portfolio" was already
  in eleven places across the notes, the skills and the code, defined nowhere.

  Writing prose in a glossary's terms is a better test of the glossary than reviewing it
  is: an undefined concept is invisible until something has to be said in its words.

  *The skills got a test.* `tests/test_skills.py` checks that every `drtools` command and
  flag the prose names exists, that the chain holds all five, that frontmatter carries a
  name matching its directory and a description stating when to use the skill, and that
  the manifest declares the version `pyproject.toml` declares. Prose cannot be
  type-checked; the names in it can.

  It was broken when written — it passed with a deliberately bogus `--candidate-id`
  injected, because the flag pattern matched only contiguous flags and stopped at the
  first one taking a value. Found by injecting the error on purpose, which is the only
  thing that makes a test written after the code worth anything.
