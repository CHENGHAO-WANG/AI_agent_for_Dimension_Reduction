# dr-agent — design notes

Running record of decisions and their rationale. Raw material for the manually
written 4-page `report.pdf`. Append as decisions are made; do not rewrite history.

Fourteen work days against a 2026-09-28 deadline. Days are indexed rather than
dated: the build has run ahead of the calendar, and the index is what the schedule
actually tracks.

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

| Day | Work |
|---|---|
| 1 | Env, pinned deps, repo skeleton, these notes, synthetic fixtures |
| 2 | Loader contract, profiler, recon pass, CLI/JSON scaffold |
| 3 | Registry YAML, pipeline engine, linear and spectral executors |
| 4 | Manifold and neighbour-embedding executors; subprocess isolation, timeouts |
| 5 | Metrics battery, rank, pre-registered weighting, plan validator |
| 6 | Viz house style, size-adaptive rendering |
| 7 | The five skills, /analyze, decisions.jsonl |
| 8 | Report template + pandoc render; first end-to-end run, dataset 1 |
| 9 | Fix what day 8 broke; clean run on dataset 1 |
| 10 | End-to-end on dataset 2 (large) |
| 11 | GPLVM, timeboxed |
| 12 | Rules-planner hedge + ablation run |
| 13 | Agent-behaviour tests, ADR, README, CONTEXT.md |
| 14 | Final graded runs, reports, submit |

Synthetic fixtures land on day 1 and become the daily smoke test: every day ends with
the full pipeline running on toy data in under a minute.

Cut order if the schedule slips: GPLVM first, then the hedge. Never the tests, never
day 14.

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
