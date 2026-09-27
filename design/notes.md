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
`decisions.jsonl`, `run.json`, and `results/` -- the report, its figures and the exported
data, which section 5 describes. Each skill's first instruction is "read
`profile.json`", never "recall what you found".

This buys resumability when a long run dies, an auditable trail the grader can open,
a directory that *is* the "generated outputs" deliverable, and the ability for the
rules planner (section 6) to reuse the same executors.

**What a run reproduces, and what it does not.** Everything after registration is a
function of the registered plan, the data and the run's seed, so replaying from
`plan.registered.json` returns every number. The plan itself is not: the candidates are
nominated by the agent's judgment, which no seed governs, and running `/analyze` again on
the same data may register a different portfolio and crown a different winner. A report is
therefore reproducible *conditional on its registered plan*, and its limitations section
should say so in those words rather than leave "seeded" to imply more. The inputs to that
judgment are on record -- profile, reconnaissance, their observations, the capability
records -- and so are its outputs and every refusal on the way; the step between them is
not, beyond the rationale and rejections the agent writes. That is why rejections must cite
evidence that resolves, and why section 6's rules planner matters beyond the ablation: it
is the one way to nominate that is itself reproducible.

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

*Added on day 10.* **Runtime is measured and reported, never weighted.** `runtime_s` leaves
the metrics a weighting may name, for four reasons.
- *By the time ranking happens, the cost is already paid.* Ranking comes after every
  candidate has run, so rewarding the faster one saves the user nothing. Keeping the
  analysis affordable is the Budget's job, and a candidate over its time limit already
  fails. Cost would matter only to a user who re-runs the pipeline, and the deliverable is
  the representation (section 3.7). The contrast with d is the point: d is a property of
  the deliverable, so it is priced in the ranking; runtime is a property of how the
  deliverable was produced.
- *It was scaled within the run, and that produced wrong rankings.* The fastest candidate
  scored 1 and the slowest 0. Under weights of 0.9 on trustworthiness and 0.1 on runtime,
  A (0.80, 1 s) beat B (0.82, 3 s). Adding C (0.60, 100 s), which could not win, reversed
  them and B won, because C stretched the scale. And A (2.0 s) beat B (2.1 s) on a tenth
  of a second, since any gap, however small, is stretched to the whole range.
- *It would make the ranking irreproducible.* Section 2.2 says a replay returns every
  number; runtime is the exception, varying with machine load and thread count, so a
  weight on it lets the winner change on replay.
- *It rewards the wrong things.* A candidate subsampled to 5,000 rows is fast because it
  looked at less data. And under section 3.5 it is not even clear whether the tuning fits
  count.

What replaces it: every candidate's run time goes into the report, beside whether its
pipeline can place new samples in the embedding without refitting -- which is what a user
planning to reuse the pipeline actually needs, and a property the registry should declare
per op. Where that reuse is the purpose of the analysis, cost may separate close
competitors (section 3.8), in the report's words rather than in the score. With runtime
gone, every metric in the battery is on an absolute scale.

*Added on day 10.* In a run whose purpose is visualization, the weighting ranks nothing:
it is the objective inside each candidate's tuning, and nothing more (section 3.11).

*Added on day 10.* **How the weighting is set.** Until now the agent chose the weights
freely, with one sentence of guidance and a free-text justification that nothing checks --
the least constrained input in the plan, and in a representation run the one that picks
the winner. A rejection must cite evidence keys that resolve; a weighting cited nothing.
Now:
- *A default.* Local structure 0.5, split as trustworthiness 0.25 and continuity 0.25, and
  global structure 0.5 on the Shepard correlation. A departure is allowed, but its
  justification must cite evidence keys, resolved at registration as a rejection's are,
  so that every departure shows as one.
- *A focus question in both kinds of run* (section 3.11). In a representation run it asks
  what the representation is for downstream: clustering or other neighbourhood-based
  analysis moves weight toward trustworthiness and continuity; distances, regression or a
  map of the whole moves it toward the Shepard correlation. In a visualization run it asks
  which kind of picture is wanted. Unanswered, the agent sets the focus from the evidence
  or keeps the default when the evidence favours neither, and logs which. *Open:* the
  weights each answer maps to. Proposed: 0.35, 0.35 and 0.3 for a local focus, and 0.15,
  0.15 and 0.7 for a global one.
- *Label metrics only with labels.* kNN label preservation and silhouette need labels.
  Without labels neither may carry weight, and `validate-plan` refuses a weighting that
  names them; today it accepts one and the weight is redistributed silently at ranking
  (defect 20 in the day 10 log). With labels both may carry weight in either kind of run.
- *A second default, for trusted labels.* Added later on day 10, because "may carry
  weight" left the numbers to the agent, the same unconstrained choice the default above
  exists to remove.

  | Metric | No trusted labels | Trusted labels |
  |---|---|---|
  | trustworthiness | 0.25 | 0.175 |
  | continuity | 0.25 | 0.175 |
  | Shepard correlation | 0.50 | 0.35 |
  | kNN label preservation | -- | 0.20 |
  | silhouette | -- | 0.10 |

  Seventy per cent stays on unsupervised fidelity, in its own proportions of 1 : 1 : 2,
  because the deliverable is a faithful representation of the data and the labels are an
  outside check on it, not the goal: weighted heavily, they would reward the candidate
  that best reproduces the known groups, which need not be the one that best keeps the
  data's structure. kNN label preservation outweighs silhouette because it is a local
  agreement measure close in kind to fidelity, read against its reference value, while
  silhouette measures how far apart the labelled groups sit -- in its own capability
  record, partly "a display property rather than a fidelity one". The focus answer
  adjusts the unsupervised share as it does without labels.

  *Trusted* means supplied with the data and not derived from it, and whether the labels
  are trusted is a logged decision with evidence, made before any result exists. Labels
  computed from the same data -- cluster assignments such as the Leiden labels section 7
  lists as a possibility for PBMC3k -- are circular: those clusters were built from a
  neighbour graph, often on PCA, so weighting agreement with them rewards whichever
  candidate most resembles the pipeline that made them. Such labels, and any labels when
  the analysis is meant to find new groups, carry weight 0; both label metrics are then
  reported as descriptive numbers only. *Open:* the numbers in the table are proposed and
  await confirmation.
- `runtime_s` carries no weight in any run (above).

The weighting serves up to three jobs in a representation run: the objective that tunes
every candidate's hyperparameter (section 3.5), the Q(d) of the target-dimension rule for
candidates with no criterion of their own (section 3.7) -- the others choose d by their own
criterion -- and the ranking. How d is traded against fidelity, still open in section 3.7,
is therefore part of the same decision.

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

*Added on day 10.* **What the three roles are, and why breadth comes first.** The roles
belong to each candidate's terminal reduction. The linear baseline keeps the directions
of largest variance and cannot unroll curved structure. A local-structure method --
UMAP, t-SNE, LLE, Laplacian Eigenmaps -- keeps each point's nearest neighbours near and
gives up the distances between far-apart points. A global-structure method -- MDS on
straight-line distances, Isomap on distances along the manifold, Diffusion Maps across
scales -- keeps distances between all pairs and gives up fine local detail. The three
match what the battery can distinguish: trustworthiness and continuity measure local
fidelity, the Shepard correlation global. Each family is an assumption about the data,
so a portfolio sharing one assumption succeeds or fails as a block and cannot say whether
the assumption holds; spread across families, disagreement between candidates becomes the
finding. Breadth is taken within what the evidence permits -- a disconnected graph rules
out the spectral family before spread is sought -- and one kind of depth is deliberate: a
pair of candidates differing in a single factor, like `umap` against `pca50 -> umap`,
which isolates that factor as a controlled comparison would. Section 3.5 strengthens the
case: hyperparameter search now happens inside each candidate, so candidate slots are for
different hypotheses, not for variants of one method.

*Neither the count nor the spread is checked.* Only the ceiling and the PCA baseline are
refused; a plan holding PCA alone passes. Spread cannot be checked yet, because the local
and global axis lives only in the registry's prose `preserves` text -- the `family` labels
do not encode it, and kernel PCA, linear in a kernel space and nonlinear in the original
one, fits no role cleanly. The intended check is a warning keyed on a declared property,
once that property exists.

### 3.5 Hyperparameters

*Rewritten on day 10. The day 10 log records what this section said before and why it
changed.*

**Starting values come from the profile, not the library.** `drtools suggest-params`
derives perplexity from n (n/100, clipped to 5-50) and `n_neighbors` from n and from
reconnaissance's connectivity and density findings, each with its reasoning and evidence
keys. The case for that is unchanged: t-SNE's default perplexity of 30 is plainly wrong on
a 500-sample dataset, and a grader in this field will check exactly that.

The original claim that library defaults were "rejected outright" overstated what was
built. Of nineteen hyperparameters across the ten reductions, five are suggested from the
data, two are computed by their executors when left empty (kernel PCA's `gamma`,
Diffusion Maps' `epsilon`), one is fixed by argument (Diffusion Maps' `alpha = 1`), and
eleven run at the registry default unless the agent sets them. That is acceptable only
once each of the eleven has been placed in one of the kinds below.

**Four kinds of hyperparameter, and only one of them is tuned.**

- *Fidelity parameters* -- perplexity, `n_neighbors`, the kernel width of kernel PCA and
  Diffusion Maps, and Diffusion Maps' time `t`. These govern how much of the data's
  structure the method can represent, so choosing them by the battery's fidelity scores
  is legitimate. These are the ones tuned.
- *Display parameters* -- UMAP's `min_dist`, which the registry itself describes as a
  visual packing parameter with no bearing on what the data is like. Tuning it against
  fidelity would optimise nothing real. Left at its default.
- *Modelling choices* -- UMAP's `metric`, MDS metric against non-metric, kernel PCA's
  kernel, the LLE variant, t-SNE's initialisation, PCA's whitening, sparse PCA's
  penalty. These decide what similarity or structure means for the question, so they are
  chosen by reasoning and logged. A score cannot choose them: the battery measures
  neighbourhoods by Euclidean distance on the reference, so tuning UMAP's metric against
  it would select Euclidean by construction.
- *Computational settings* -- MDS's restarts, t-SNE's iteration count. Set once,
  generously enough that the optimiser converges, and not tuned.

**Tuning happens for every candidate, before ranking.** It has to: the ranking is
computed from tuned candidates, so tuning cannot wait to learn which one wins. d is
chosen in the same procedure, since section 3.7 makes it a per-candidate choice and the
best hyperparameter and the best d depend on each other.

**The procedure.** Declared in the plan and frozen at registration: the grid of d values,
the grid of multipliers applied to the suggested value, the rule for choosing d and its
parameter (section 3.7), the tie-breaks -- smallest d, then the multiplier nearest 1 --
and a cap of two cycles.

1. Draw a tuning subsample of at most 2,000 rows, stratified by label when labels exist,
   as reconnaissance already does.
2. *For a method nested in d* (defined below): fill the whole grid. One fit per
   multiplier at the largest d, truncated to every smaller d, gives every cell. *Revised
   later on day 10:* what happens next depends on what the candidate's d rule reads.
   - *Its own criterion* -- PCA's spectrum, Isomap's residual variance, a spectral gap,
     kernel PCA's eigenvalues. For each multiplier m, the rule reads that one fit's Q(d)
     curve and picks d(m). The battery then scores each pair (m, d(m)) under the
     weighting, and the best pair wins.
   - *The battery.* Profile -- for each d keep the best multiplier -- and apply the d rule
     to that curve.
3. *For a method not nested in d*: alternate, starting from reconnaissance's estimate of
   d. Tune the multiplier at that d; choose d at that multiplier; tune the multiplier
   again at the new d; choose d again. Stop as soon as a step changes nothing, and record
   *converged*. Otherwise stop at the cap and record *stopped at the cap*, then score the
   neighbouring cells, diagonals included, as a check. The multiplier step is scored
   by the battery, and the d step applies the candidate's own d rule at the current
   multiplier. In the registry as it stands the distinction does not bite: every method
   not nested in d either has no criterion of its own -- UMAP, t-SNE, the modified,
   Hessian and LTSA variants of LLE -- so both steps read the battery, or has no fidelity
   hyperparameter -- MDS, whose stress scree picks d and nothing is tuned.
4. Refit once on the candidate's full input, at the chosen d and at the chosen
   multiplier times the suggestion recomputed at the full n, under the run's own seed
   exactly as an untuned candidate is fitted. The battery scores this refit, and the
   ranking uses it alone.
5. Record every cell computed with its score, feasibility and run time, the rule
   applied, the stopping outcome and the derived seeds used. Provenance for both values
   is `tuned`.

Every cell is fitted, and scored on the tuning subsample under the pre-registered
weighting, with seeds *derived* from the run's seed rather than the run's seed itself
(below).

**Why each piece is there.**

- *The battery chooses the hyperparameter, even where a candidate's own criterion chooses
  d.* Added later on day 10. A method's own criterion measures how well the embedding fits
  a target the method builds for itself, and the hyperparameter builds that target.
  Isomap's residual variance compares the embedding with the geodesic distances of the
  graph `n_neighbors` builds: at k = 5 the geodesics follow the manifold, at k = 50 they
  cut toward straight lines, so a low value at each says "this embedding matches its own
  graph" about two different graphs, and neither says how well it matches the data. And
  some criteria have a degenerate optimum: shrink Diffusion Maps' kernel width until the
  graph falls into c pieces and the first c eigenvalues all equal 1, so maximising the
  spectral gap rewards breaking the data apart. The parallel is comparing the fit of two
  models fitted to differently transformed responses. The battery avoids both problems,
  because its target is the Reference, the same for every hyperparameter value and every
  candidate. The same reasoning is why step 2 never stitches an own-criterion curve
  across multipliers: a curve whose points come from different fits would be read by the
  elbow or gap rule as though it were one, the flaw that ruled out a t-SNE sweep stitched
  across two implementations. A profiled battery curve may mix multipliers safely,
  because every point measures the same target.

  *A condition, not a ban.* A candidate's own criterion may choose its hyperparameter if
  its target does not depend on that hyperparameter and it has no degenerate optimum. No
  criterion in the registry meets both, but a method added later might, so the condition
  is what stands.
- *Profiling rather than fixing the hyperparameter first.* A method whose hyperparameter
  is badly set needs more dimensions to reach the same fidelity, so choosing d at an
  untuned value chooses too large a d -- the failure "the smallest d that preserves
  enough" exists to avoid. On an illustrative UMAP surface, fixing the multiplier at 1
  chose d = 8 where the best cell was d = 4.
- *Alternating, for methods not nested in d.* Modelled on one-step GEE and on DESeq2's
  sequence of a rough dispersion estimate, then the means, then the dispersion again.
  Such schemes need two conditions, and both are provided for. A data-driven starting
  point: reconnaissance's intrinsic-dimension estimate plays the part of DESeq2's
  moment estimate, cheap and independent of the model being fitted. Weak coupling
  between the two blocks, as DESeq2's mean and dispersion are near-orthogonal: here it
  holds for a structural reason, since in nearly every method the hyperparameter builds
  the input-space graph, affinities or kernel before d is involved, and d only decides
  how much of the output is kept. Hessian and LTSA LLE are the exception, coupled
  through a neighbour minimum that grows with d. The starting point is not a formality:
  on the illustrative surface, starting from the suggested value stopped at (d = 8,
  multiplier 1) -- a point no single-coordinate step improves, and not the best cell --
  while starting at reconnaissance's d reached (4, 2).
- *The full grid, for methods nested in d.* There it costs one fit per multiplier, which
  is what a single tuning step costs anyway, so alternating would save nothing.
  Alternating pays for methods not nested in d, UMAP chief among them, and the saving
  grows with the grids: 15 fits against 7 to 13 at five values of d and three
  multipliers, 50 against about 27 at ten and five. Its real value is that it affords
  finer grids within the same budget.
- *Refitting, and ranking on the refit.* Keeping the best of g configurations reports
  the largest of g noisy estimates, biased upward even when every configuration is
  equally good. A refit that was not selected on carries no such bias -- the logic of a
  held-out test set.
- *A multiplier, not a raw value.* `n_neighbors` and perplexity are counts, and on a
  subsample a tenth the size the same count reaches much further. The suggestion already
  scales with n, so a multiplier transfers between the subsample and the full data where
  a raw value would not.
- *Feasibility judged at the refit's n as well as the subsample's.* A 2,000-row subsample
  runs t-SNE with Barnes-Hut, which permits d = 3; a refit at 10,000 rows or more runs
  FFT, which refuses it. Hessian LLE's neighbour minimum must hold at both sizes. A cell
  infeasible at either is skipped and recorded as skipped, never scored as zero.
- *Derived seeds, never chosen ones.* A run has one seed, fixed when it is created:
  `run_seed` refuses any other, and a retry reuses it, so the agent cannot re-run a
  stochastic method until it scores well. Tuning must not weaken that. It does need the
  tuning fits to be a different draw from the refit: when n is at most 2,000 the tuning
  subsample is the whole data set, and tuning under the run's seed would make the refit
  reproduce the selected fit exactly and bring the upward bias back. So the toolbox
  derives separate streams for the tuning fits and the tuning metric subsample from the
  run's seed with NumPy's `SeedSequence`, keyed on purpose, and records them. They are
  computed, not chosen, which keeps the property the locked seed exists for; the run's
  own seed keeps exactly its present meaning for every final fit and the final battery.
  Keying on purpose rather than on the candidate keeps all candidates on common random
  numbers, as they are today, which plausibly narrows the noise in comparisons between
  candidates running the same stochastic method. For deterministic methods only the
  metric stream matters, and where n is small enough that nothing is subsampled there is
  no noise to select on.

**Nested in d, defined and checked.** A method is *nested in d* when its d-dimensional
embedding equals the first d columns of its embedding at any larger d, for the same data
and hyperparameters. Methods whose coordinates are eigenvectors of a matrix built from the
data alone are nested, because asking for more components keeps more eigenvectors
without changing any. Methods that optimise a layout in d dimensions are not, and nor are
methods whose matrix is itself built from d.

Checked on a noisy Swiss roll, comparing the fit at d = 2 with the first two columns of
the fit at d = 5: PCA, kernel PCA, Isomap, Laplacian Eigenmaps and standard LLE agree
exactly. Modified LLE does not (column correlations 0.998 and 0.505), nor do Hessian LLE,
LTSA, MDS and exact t-SNE. UMAP agrees at 0.98 under its default spectral
initialisation, which is itself nested, and at 0.23 and 0.56 under random initialisation:
the agreement belongs to the initialiser, not to UMAP. Diffusion Maps is nested by
construction, since its executor keeps the leading eigenvectors of one decomposition.

The check matters because slicing a method that is not nested *understates* fidelity at
the smaller d -- trustworthiness 0.925 against 0.995 for UMAP under random
initialisation, 0.906 against 0.997 for exact t-SNE -- which would push the d rule toward
larger d. So `nested_in_d` is a declared field, and a test runs this comparison for every
op that claims it. It is never assumed from the method's family. Changing LLE's default
variant from `standard` to `modified` would move it from the cheap branch to the
expensive one.

**What this section used to promise, in its right place.** "A small sweep of 2-3 values
only for the top-ranked candidate" cannot feed the ranking: it would give extra tries only
to the candidate already in front, which is what section 2.4 exists to prevent. With every
candidate tuned, each tuning record already shows how that candidate's fidelity moves with
its hyperparameter near the chosen value, which is most of what a sensitivity analysis
would say. A separate full-data sensitivity check on the winner and its close competitors
remains available, reported and never used to re-rank.

**None of this is built yet.** No command runs a sweep of either kind, `suggest-params`
writes nothing into the run, and provenance distinguishes only `specified` from
`registry_default`. The day 10 log carries the defects.

### 3.6 Plan validation gate

`drtools validate-plan` hard-rejects incoherent plans — t-SNE with perplexity >= n/3,
Isomap at 107k samples, raw counts into Euclidean MDS, `n_neighbors > n` — and the
agent must revise and resubmit. **Every rejection is logged**, which means the report
can honestly say "the planner proposed X, the validator caught it, the agent revised
to Y". That is a working agent loop demonstrated with evidence.

### 3.7 The target dimension

Sections 3.1-3.6 settle which pipelines run and with which hyperparameters, and leave
one parameter undecided: how many dimensions a candidate's embedding has. Until day 10
it was decided by nothing. Every reduction in the registry defaults `n_components` to 2,
no skill mentions the parameter, and the validator does not check it, so every run
produced 2-D embeddings by inheritance rather than by judgment -- while the recon pass
computed an intrinsic-dimension estimate whose only use was a warning that 2 might be
too few. Evidence gathered and then discarded is the failure section 3.1 exists to
prevent, arriving one stage later than that section looks.

**The deliverable is the best representation, and a picture of it.** Not the best
picture. A run chooses the smallest d that preserves enough of the relevant structure,
and plots that representation separately. This is the decision the rest of the section
depends on: were the deliverable the picture, d = 2 would be correct by definition and
there would be nothing to choose.

**d is chosen per candidate.** The right d for Isomap is not the right d for UMAP, since
the methods differ in how much structure they carry per dimension. A single d shared
across the portfolio is simpler and was rejected: it forces every candidate to the needs
of the most demanding one, and it discards the per-method criteria below.

**Selection varies; scoring does not.** Two things could vary per candidate and only one
should.

- *Selection* -- the rule that picks d, and the quantity Q(d) that rule reads. Varies per
  candidate, declared in the plan, frozen at registration exactly as the weighting is.
- *Scoring* -- the battery, computed at whatever d each candidate selected. Uniform,
  because it is what the ranking reads, and a ranking with two uncontrolled sources of
  variation cannot attribute a difference to either of them.

Varying the selection rule does not weaken pre-registration. What section 2.4 buys is
that a rule cannot be chosen after seeing the result it produces; declaring a different
rule per candidate *before any embedding exists* is as frozen as declaring one. The
argument is about timing, not uniformity.

**Several methods bring their own criterion, and it is usually better than a generic
one.** PCA has the eigenvalue spectrum, Isomap the residual-variance elbow of
Tenenbaum's original paper, Diffusion Maps and Laplacian Eigenmaps a spectral gap, MDS a
stress scree plot. t-SNE, UMAP and LLE have nothing canonical and fall back to the
battery. Forcing all of them onto one generic Q would discard criteria that are standard
in the field and free from an eigendecomposition already computed.

Two familiar quantities cannot serve as that fallback, which is why it is the battery.
Explained variance exists only for the linear methods. Reconstruction error needs an
inverse map: PCA's is exact, kernel PCA's is an approximate pre-image problem, UMAP ships
a learned approximation, and t-SNE, Isomap and LLE have none at all. Trustworthiness,
continuity and the Shepard correlation need only the two coordinate sets row-aligned, so
they are computable for every method in the registry.

**Parsimony has to enter the ranking, or the design defeats itself.** Fidelity to the
reference rises with d, so a candidate that selected d = 9 out-scores one at d = 3 almost
mechanically. Ranking on the battery alone would reward whichever method demanded the
most dimensions, which is the opposite of "the smallest d that preserves enough". So d
enters the comparison as a cost. This was first written "in the way runtime already
does"; section 2.4 has since taken runtime out of the score, because d is a property of
the deliverable and runtime only of how it was produced.

*Open -- how. Settled later on day 10, below: no rate is chosen.* The proposal on the
table is a single parsimony exchange rate declared
once per run: how much normalised fidelity one extra dimension must buy to be worth
taking. It would serve twice, inside the penalised selection rules and as the weight on d
in the ranking, so that a candidate's choice of d and the portfolio's judgment of that
choice cannot disagree. The alternative is to let the selection-time and ranking-time
trade-offs differ. Not settled.

*Open -- the menu of selection rules, and which is the default. Settled later on day
10, below: the elbow, with a fallback.* Three are in play: a
threshold, `min{d : Q(d) >= tau}`; the elbow of the Q(d) curve; and a penalised
objective, `argmax_d [Q(d) - lambda*d]`. The penalised form is recommended, because
lambda states an exchange rate a reader can argue with, where tau is an absolute bar on a
scale whose achievable range depends on the dataset, and an elbow moves when the swept
range of d moves without anything in the record saying so.

*Open -- d_max, and the scale the d cost is normalised on.* It should be absolute rather
than min-maxed across the portfolio, by the reasoning already applied to the battery: a
run where every candidate landed on d = 8 must not still crown a parsimony winner.

**d_max and the grid, decided later on day 10.** Nothing bounded a reduction's output
dimension: the registry gives every reduction `n_components` with a minimum of 1 and no
maximum, and the cap of 100 set in section 3.11 applies only to PCA's output when PCA is
a chain's first stage. A bound is needed without any penalty, for three reasons: section
3.5 declares the grid of d in the plan before anything runs, so it must be finite; cost
grows with it, since nested methods fit once at the largest d and non-nested ones refit
at every grid point; and some limits are structural.
- *d_max = min(100, p - 1, n - 1)* for every reduction in a representation run, matching
  the cap on PCA's first-stage output. A visualization run has d = 2 and no grid.
- *Each method's own structural limit on top*, declared in its capability record and
  checked at registration. Hessian LLE, for one, needs `n_neighbors > d(d + 3)/2`, so
  d = 10 already requires 66 neighbours.
- *A grid that thins at high d*: 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100, truncated at
  the candidate's d_max. A difference between d = 80 and d = 81 is rarely larger than the
  noise, and the thinner grid keeps refits affordable for methods not nested in d.
- *A candidate whose rule stops at d_max is reported as capped*, not as having found its
  dimension there.

*Still open, and the user is still weighing it: the penalty on d. Settled later on day
10, in the next paragraph.* Two further proposals
were discussed on day 10 and neither is adopted. One scales the rate by p, `lambda = mu/p`,
so that the penalty `mu * d/p` lies on the score's own 0-1 scale. Within a run p is a
constant, so this only rescales mu and cannot change a ranking that mu alone would not; it
matters only if one default mu is reused across runs, and on wide data it vanishes -- at
p = 20,000, d = 10 gives d/p = 0.0005. The other removes the chosen rate. Gains within the
noise are handled by the one-standard-error rule, as in CART pruning and glmnet's
`lambda.1se`: within a candidate, the smallest d whose score lies within one standard
error of that candidate's best; across candidates, those within one standard error of the
leader are close competitors and the smaller d wins among them. A genuine trade-off, more
faithful by more than the noise but in more dimensions, is ranked fidelity first, and the
report states the rate at which it would reverse, `lambda* = (S_A - S_B) / (d_A - d_B)`: a
gap of 0.02 over 6 extra dimensions reverses if one dimension is worth more than 0.0033 of
score. The objection that started both: a rate chosen by hand has an effect on the ranking
that nobody can see in advance.

**How d is chosen and priced, decided later on day 10.** No rate per dimension is chosen,
and no rule rests on noise alone.

*Within a candidate*, only for candidates whose d rule reads the battery. In a
representation run that is LLE, in all four variants, and sparse PCA. t-SNE and UMAP never
choose d: they are visualization methods fixed at d = 2 (section 3.11). The other
reductions keep their own criterion, as above.
1. Build the candidate's score curve over the declared grid of d, one curve per
   multiplier (section 3.5).
2. Replace the curve by its running maximum, the best score reached at or below each d.
   This removes dips, and it is the honest reading: a larger d is never worse than
   keeping a smaller one.
3. If the curve is flat -- its best score at most a *flatness threshold* above its score
   at the smallest d -- choose the smallest d. Rescaling a flat curve would only magnify
   its noise into a false elbow. The flatness threshold is its own number, not tied to
   delta below: delta compares two candidates, this compares one candidate across d. Its
   default is set slightly above delta's 0.02 -- *proposed 0.03, to be confirmed*.
4. Otherwise take the elbow by the Kneedle method, with log d on the horizontal axis:
   rescale both axes to [0, 1] and choose the *interior* grid point furthest above the
   straight line joining the curve's two ends. An end of the grid is never an elbow.
5. If no interior point lies above that line, the curve has no point of diminishing
   returns in the grid: choose the smallest d whose score is within 0.10 of the
   candidate's best. When the curve is still rising at the last grid point, the choice
   is reported as capped.

Log d because the elbow then moves much less with the range. Tested on day 10 on the
digits data with the grid ending at 20, 30 and 50, using PCA and Isomap purely as test
curves, since both keep their own criterion: PCA's elbow moved 6, 8, 10 on d and 5, 5, 8
on log d; Isomap's 6, 8, 8 on d and 6, 6, 8 on log d. The range itself is fixed by the grid
the plan declares, so for a given run the elbow is fully determined and cannot be moved
once results exist. The flatness guard was prompted by a flat, bumpy curve -- UMAP's,
used only because its shape was flat -- on which Kneedle alone reported a false elbow.

Then tested on the two candidates this rule actually serves, on the same data. Sparse
PCA's curve rose from 0.762 at d = 2 to 0.985 at d = 30, and the rule chose d = 6. LLE with
12 neighbours scored 0.630, 0.609, 0.621, 0.627, 0.629, 0.706 and 0.758 at d = 2 to 10,
stopping at 10 because standard LLE needs d below `n_neighbors`. Plain Kneedle called
d = 2 the elbow: after the dip the first point sits above the chord. That is what steps 2
and 4 fix. On the running maximum no interior point lies above the chord, the fallback
chooses d = 8, and because the curve was still rising at the last grid point, that choice
is reported as capped.

*Across candidates: a non-inferiority margin, delta = 0.02 by default*, on the weighted
score. The candidates within delta of the leader are close competitors, and among them
the smallest d wins, ties going to the higher score; the rest are ranked by score. delta
says how much fidelity is treated as negligible -- "2 points out of 100" -- which a reader
can argue with directly, where a rate per dimension had an effect on the ranking nobody
could see in advance. It is declared in the plan, and a departure from the default cites
evidence, as a weighting's does. It settles section 3.8's open margin.

*Reported: the path of winners.* For every rate lambda >= 0 at which one dimension might
be valued, the candidate maximising `S - lambda * d` is the winner at that rate; the winner
changes exactly at the slopes of the upper convex hull of the points (d, S), as along a
lasso path, and a candidate off the hull wins at no rate. The report prints each
breakpoint as a sentence -- "if one dimension were worth more than 0.0065 of score,
candidate B would win" -- and names the candidates that win at no rate. For two
candidates the breakpoint is `(S_A - S_B) / (d_A - d_B)`. The report also shows each
battery-reading candidate's curve with its chosen d marked.

*The standard error is reported, not used to choose.* A grouped jackknife over the scored
rows -- ten groups, the same groups for every candidate -- gives each score's precision:
about 0.004 to 0.006 on 1,797 rows. It measures only which rows are scored, holding the
fitted embeddings fixed, so it excludes seed and refit variability and is a lower bound.

*Rejected: the one-standard-error rule*, proposed on day 10 by analogy with CART pruning
and glmnet's `lambda.1se`: within a candidate the smallest d within one standard error of
its best, across candidates the smallest d within one standard error of the leader.
Tested on the digits data with PCA, the score rose steadily from 0.760 at d = 2 to 0.9997
at d = 50, where the standard error was 0.0000, and the rule chose d = 50. In
cross-validation the rule works because error has an interior minimum and a sizable
standard error; fidelity to the Reference has no interior optimum, and its standard error
vanishes as an embedding approaches the Reference. With a couple of thousand scored rows
almost any real gain exceeds one standard error, so statistical significance cannot stand
in for practical importance, and some stated margin has to.

**d is chosen together with the hyperparameter.** Section 3.5 gives the procedure:
the full grid, profiled, for methods nested in d; alternating updates started from
reconnaissance's estimate of d for the rest. Choosing d first, at an untuned
hyperparameter, would pick too large a d, because an under-tuned method needs more
dimensions to reach the same fidelity.

**What the mechanism has to respect.** Established by reading the executors, not the
capability records.

- Most of the portfolio is *nested in d* (defined and checked in section 3.5). PCA,
  kernel PCA, Isomap, Diffusion Maps, Laplacian Eigenmaps and standard LLE are
  eigendecompositions of a matrix that does not depend on d, so one fit yields the whole
  Q(d) curve by truncation. UMAP, t-SNE, MDS under SMACOF, and the modified, Hessian and
  LTSA variants of LLE are not, and need a refit at every d. UMAP can look nested, because
  its default initialisation is a Laplacian Eigenmaps embedding; under random
  initialisation the agreement disappears. That is the difference between a sweep that is
  free and one that is expensive, so it belongs in the capability record as a declared,
  tested field rather than being rediscovered per method.
- t-SNE's output dimension is limited by the approximation the library runs, not by
  t-SNE. openTSNE uses FFT interpolation at 10,000 samples or more, which supports one or
  two dimensions, and Barnes-Hut below that, which supports three and warns of segfaults
  beyond. The validator should enforce that limit as a function of n and name the
  approximation, rather than let a candidate fail in execution. Above it, see the two-op
  split at the end of this section.
- Hessian LLE needs `1 + d(d+3)/2` neighbours -- 6 at d = 2, 45 at d = 8 -- so raising d
  silently invalidates an `n_neighbors` chosen for d = 2.
- MDS reports the SMACOF objective, which the executor already notes is not comparable
  across dimensionalities. A stress scree plot as a criterion needs Kruskal's
  normalisation stated explicitly.
- A sweep runs inside one attempt. Counting it as several would spend section 4's
  allowance before the candidate ever ran at its chosen d.
- The display stays 2-D, and it **cannot** be the first two columns of a d > 2 embedding
  for any neighbour method, whose axes are arbitrary and unordered. A stated 2-D PCA
  projection of the embedding is the honest route, and scoring that projection on the
  same battery turns the cost of looking at the data into a number the report can carry.

*Withdrawn later on day 10.* Section 3.11 makes t-SNE a visualization method, which
never produces a representation, so t-SNE above three dimensions has no use and
`tsne_exact` is not built. The reasoning is kept below as the record of what was decided
and why it lapsed.

**t-SNE above three dimensions: a second op, not a wider limit.** The mathematics does
not stop at three; the approximations do. Rather than widen `tsne`, the registry gains
`tsne_exact` -- scikit-learn's exact gradient, any output dimension, `O(n^2)` in time and
memory. Two ops rather than one entry with a wider range, for two reasons. The
capabilities move with `n_components`: cost, `scales_to` and the upper bound itself all
differ between the approximated and the exact method, and one record cannot describe
both. And a sweep over d must stay inside one implementation: a curve built from openTSNE
below the limit and scikit-learn above it would differ in learning rate, early
exaggeration and initialisation, and the selection rule would read the switch point as
structure. Each op is swept within its own implementation, so neither curve has one.

Scikit-learn already sets the Student-t kernel's degrees of freedom to `d - 1`, van der
Maaten's adjustment for the fact that crowding eases as dimensions are added and a
one-degree tail would otherwise over-separate clusters, so nothing needs adding for it.
`tsne_exact` is stochastic and therefore terminal-only under section 3.9. Its
`scales_to` is to be set by measurement and is expected near MDS's 5,000; on large data it
needs a subsample first, exactly as MDS and Isomap do.

*Argued against first, and the argument did not hold.* The case against was cost, overlap
with UMAP, the kernel tail, and the stitched sweep. Cost cannot disqualify it: MDS,
Isomap, Diffusion Maps and kernel PCA are all quadratic or worse and all in the
portfolio, and the principle they satisfy is that an expensive method earns its place by
giving something no cheaper method gives. Overlap with UMAP is real but is not identity --
the two model neighbourhoods differently, Gaussian affinities calibrated by perplexity
against fuzzy simplicial sets -- and whether that difference matters above three
dimensions is an empirical question of exactly the kind the battery exists to settle.
The kernel tail is handled by the library and the stitched sweep by the split. What
remains against it is scope, and usefulness confined to small data.

A by-product worth having: `tsne` and `tsne_exact` run at d = 2 on the same data measure
how much fidelity the approximation costs, a number the report cannot state today.

### 3.8 Ties, and the winner that is not one

*Scope, added later on day 10.* This section governs representation runs. A
visualization run ranks nothing and names no winner (section 3.11).

`rank` names a single winner and, when the top two scores fall within 0.02, appends a
note saying they "should be treated as tied rather than ranked". Those two statements
live in different places and nothing reconciles them. The tie exists only as English in
a `notes` list; the data model has no representation of it at all. Four consumers read
`winner` and none reads `notes`: the report prints it in bold, `figures` draws the
per-class facet for it, the comparison panels are ordered by it, and -- the one that
matters -- `rank` writes it into the decision log as the answer to "which candidate best
serves the question this analysis is answering". A tie therefore enters the append-only
record as an unqualified choice, with the qualification in a sibling field that entry
does not carry. Section 2.3 is the standard this fails: a claim the numbers do not
support, in the record the report is generated from.

**One winner, plus a computed set of close competitors.** Not a set of joint winners.
`winner` stays a single id, so every consumer above keeps working, and the tie becomes
an additional field rather than a change to an existing one. It is also the more honest
claim: "these two are tied" says the ranking failed to order them, where "UMAP ranked
first, with PCA and Isomap inside the noise" says there is an ordering whose top is not
well separated -- which is what is actually true.

**The set is computed, not proposed.** Membership belongs to `rank`, not to the agent.
Deciding which candidates count as close is a ranking judgment, and section 2.1 puts
ranking inside audited code so that every number in the report comes from something a
reader can check. An agent naming near-winners would be making a numerical claim no code
produced, while looking at which methods they are. The agent's say is over what the
closeness *means*, in section 8 of the report -- that the data is not strongly
nonlinear, that the extra machinery did not pay -- and that is the part no rule can
supply.

**The rule is pre-registered; the outcome varies.** Declared in the plan, frozen at
registration, exactly as the weighting is (section 2.4). Letting the agent set the
margin once the scores exist is worse here than it would be for the weighting, because
**What a close competitor gets in the report, and why a mention is not enough.** Three
tiers exist today. Every successful candidate gets rows in sections 3, 4, 6 and 7, its
own embedding figure, a panel in the comparison figure, and a bar in the metrics figure.
An unscored candidate gets its sections 3 and 4 rows and one line in each of 7 and 9.
Only the winner gets the bold winner line, first position among the comparison panels,
the decision log's `chosen` -- and the part that matters, two diagnostic figures nobody
else gets: the class facet, one panel per class, and the Shepard diagram of distances
before against distances after.

Those two figures go to the close competitors as well. The reasoning is the one that
makes the set worth computing at all. Naming a close competitor says the ranking could
not separate it from the winner, so the separating falls to the reader, and those two
figures are what a reader separates embeddings with. Drawing them only for the nominal
winner hands over the evidence to scrutinise the candidate whose selection is least in
doubt, and withholds it exactly where the judgment is live. The cost is small:
per-candidate figures already exist for every success, the Shepard is capped at 800
points, and the competitor set is capped, so this is at most six extra figures.

The rest stays winner-only -- the bold line, the panel order, and the decision log's
`chosen`, which gains `close_competitors` as a sibling field rather than changing shape.
Section 7 prints each competitor with its margin as a computed value rather than prose,
and marks them in the ranking table itself, so a reader scanning rows sees the tie
without reading the paragraph beneath. Section 9 has to carry the new note too, and
should read a field to decide that: it currently selects which notes are limitations by
testing whether the words "tied" or "noise" appear in them, which is the same
prose-as-data mistake this section exists to correct.

*Open -- how the margin is set. Settled later on day 10 in section 3.7: a
non-inferiority margin of 0.02 on the weighted score, the smallest d winning among the
candidates inside it. The measured standard error proposed below is reported for
precision but does not set the margin, since it vanishes as an embedding approaches the
Reference.* The 0.02 in the code is a constant with nothing behind
it: an absolute margin on a weighted score, described as the noise of stochastic methods
and repeated subsampling, where no such noise was ever measured. It can be measured, and
cheaply. The battery scores each embedding on a subsample capped at 2000 points under
one seed, and the embeddings are already on disk, so re-scoring a candidate at three or
five further subsample seeds costs seconds where re-embedding would cost minutes. The
spread of those scores is an empirical standard error for the comparison, and the margin
becomes "within one spread of the leader". Two limits to state alongside it: it measures
metric-estimation noise rather than embedding noise, so for a stochastic method it is a
lower bound on the real variability; and the subsample seeds must be shared across
candidates or the comparison is between different draws. A measured lower bound with its
scope stated is still far better than a constant, and it earns the report a sentence it
cannot write today -- that scores are reproducible to within some stated tolerance.

*Open -- how many competitors.* Not a fixed count. The rule decides membership and a cap
bounds it, proposed at three. Above the cap the note changes rather than the list
growing: four or more candidates inside the noise means the ranking did not discriminate,
which is a finding about the portfolio or the weighting and reads better as one sentence
than as a list of five.

*Open -- whether section 8 is made to address the competitors.* Section 8 is where a
close competitor has to be dealt with in words: UMAP ranked first, Isomap is
indistinguishable from it and preserves global distances better, which matters if the
layout is to be read as a map. Nothing requires that today. The agent reads `winner` and
will write about the winner, leaving the competitor set as decoration. Section 8 sits
outside the locked core by design (section 2.1), so prose in `write-report` is the
obvious lever -- and prose instruction is not enforcement, which is the lesson this
project keeps relearning. One enforceable option exists: `report --refresh` refusing a
document whose section 8 does not name each competitor id. That is a substring test on a
literal id, not the free-prose number parser the report contract rejected when it turned
down *write then verify*. It is gameable, since an id can be named and nothing said, but
it turns a silent omission into a deliberate one.

**Cost among close competitors.** Runtime is not in the score (section 2.4), but it may
still decide between candidates the score cannot separate. When the user says at the
checkpoint that the pipeline will be reused, on new samples or a larger cohort, the
report may name the cheaper of the winner and its close competitors, and say why: they
are indistinguishable on fidelity, so cost is a fair way to choose between them. The
winner does not change; the report states a reason a reader may act on. Where reuse is
not the purpose, cost is reported and nothing more.

**Sequenced after section 3.7, deliberately.** Once d enters the ranking as a cost, a
tie on fidelity between a candidate at d = 3 and one at d = 9 stops being a tie. Some
fraction of today's ties dissolve once parsimony is priced, so fixing the margin before
the scoring function changes would tune it against a target that is about to move.

### 3.9 What may occupy a non-terminal stage

Section 3.2 made the candidate a chain and the registry a place to declare which ops
may appear where. This section settles what that declaration means, why only PCA
carries it, and what the declaration is *not* able to say.

**What the role does, concretely.** `OpSpec.can_be_intermediate()` returns true when an
op declares `intermediate` in `roles`, or is preprocessing. `validate_stages` walks the
stage list and raises when a stage that is not last fails that check. The refusal fires
at plan validation and again at execution, since `run_pipeline` re-validates. That is
the whole mechanism: a gate on position in the stage list. It does not look at the
downstream op, at the cost, or at the shape of the data. `pca -> pca` passes it;
`kernel_pca -> umap` does not. Neither verdict involves the pair.

**A first stage serves one of two purposes, and the tests differ.** It can bring the
next stage inside the Budget, or it can improve what the next stage receives. Calling
these the *Budget purpose* and the *representation purpose* below; they are descriptions
rather than new vocabulary. Ruling a method out on cost closes only the first door, and
an argument that forgets this excludes methods it has not actually tested.

**The cost condition is relative, and it is not "cheaper than what it feeds".** For a
chain `A -> B` on data of shape (n, d), where A emits k dimensions:

    cost_A(n, d) + cost_B(n, k)  <  cost_B(n, d)

The pre-step must cost less than the saving it creates downstream. The shorter phrasing
-- that the intermediate be cheaper than the method after it -- is a different and weaker
claim, and an absolute threshold such as "subquadratic in n" is weaker still: it is what
the inequality reduces to when the downstream is the cheapest method available. Both
shortcuts give the right verdict against this registry, because its quadratic methods
are also its most expensive ones. That is a fact about this method list and not a
principle, and it will stop holding the moment a more expensive terminal is added.

**Reducing d and reducing n are different purchases.** Reducing d buys speed for
anything computing distances, taking `O(n^2 d)` to `O(n^2 k)`. Reducing n buys
feasibility for anything holding an n-by-n matrix, and only that crosses a hard wall:
Isomap at n = 107,000 needs 91.6 GB whether d is 32,000 or 50. A dimension-reducing
pre-step makes such a method faster and cannot make it possible. Only `subsample` can,
which is why that op is preprocessing rather than a reduction.

**Determinism, and the argument that actually carries it.** "It depends on a seed"
cannot be the objection on its own: stochastic methods are fine as terminal stages, and
a run fixes one seed for everything it does, so every result is reproducible. Two things carry it
instead. First, the variance becomes invisible -- a stochastic terminal's variance sits
in the artefact the battery scores, where it can be looked at, while a stochastic
intermediate's variance sits in the *input* to what is scored and no stage record says
which draw it got. Second, some draws are not noisy but catastrophic: LLE collapses,
which is why its executor computes a coordinate-spread ratio and warns above fifty-fold,
and SMACOF lands in local minima, which is why MDS restarts four times. A collapsed
intermediate feeds nonsense downstream while the terminal's record looks clean, so the
terminal takes the blame.

*This rests on a condition, which is recorded so that it can expire.* A run executes a
candidate once -- the second attempt of section 4 is a repair after failure, not a repeat
for variance -- so an effect attributable to a pre-step cannot be separated from a draw
of that pre-step. Were candidates repeated across seeds, a stochastic intermediate would
become admissible and this exclusion would lapse.

**The metric test, and where the refusal message overclaims.** The refusal says a
terminal method's output has "no meaningful metric for a downstream method to consume".
True of t-SNE and UMAP, which optimise neighbourhood agreement for display: a k-NN graph
built on their coordinates is built from coordinates fitted to satisfy a *different* k-NN
graph, so it measures the first method's artefacts. True of Laplacian Eigenmaps on its
own executor's caveat, that the embedding's scale is arbitrary and inter-cluster
distances are not meaningful. True of LLE, whose coordinate spreads can differ fifty-fold.

It is false for three. Diffusion Maps is constructed so that Euclidean distance in the
embedding equals diffusion distance in the data -- a theorem, not a heuristic. Isomap is
classical MDS on a geodesic distance matrix. Kernel PCA is an orthogonal projection whose
distances are meaningful in the kernel metric. All three pass a test the message claims
they fail, and the message should not be read as having settled them.

**Construction redundancy, which is a property of the pair.** If the upstream op builds
a neighbour graph and the downstream op builds one too, the second is constructed from
coordinates derived from the first: the input is largely recovered and paid for twice.
`pca -> isomap` has none of this and is the most useful pairing available; `isomap ->
umap` has it. Same upstream in one case, same downstream in the other, so this cannot be
attached to either method alone.

**The verdict.** PCA keeps the role; nothing else takes it. Two filters get there and
neither suffices alone:

| | passes the cost filter | deterministic |
|---|---|---|
| pca | yes | yes |
| laplacian_eigenmaps, lle, sparse_pca, tsne, umap | yes | no |
| kernel_pca, diffusion_maps, isomap | no | yes |
| mds | no | no |

The intersection is `pca`. The three that are deterministic but too expensive
-- kernel PCA, Diffusion Maps, Isomap -- are excluded by judgment rather than by any
declared property: Diffusion Maps and Isomap are redundant with the only downstreams
that would want them, and kernel PCA's single non-redundant pairing, in front of a
neighbour embedding, has a weak prior and would spend one of three to five candidate
slots. *That exclusion expires when the Budget grows enough for the portfolio to afford
the experiment*, at which point the question is settled by registering the candidate and
reading the battery rather than by argument, which is section 3.2's own principle.

Sparse PCA would be excluded even were it deterministic: its sparse loadings are
interpretable in the original features, and composing it with anything destroys the only
property it is chosen for.

**What the role is, stated honestly.** A screen, not the condition. Both the cost test
and construction redundancy turn on the pair, and a per-op field cannot express a pair
constraint, so `roles` approximates one: it says an op may be intermediate against *some*
downstream, never that a particular chain is sound. It is therefore too tight across
methods -- it forecloses pairings the real condition would permit -- and too loose within
the one method it admits, since `pca(50) -> pca(2)` passes and buys nothing.

**At most two reductions per candidate, and not the same one twice.** Decided on day 10,
and it closes the looseness just named. Counting only reduction stages -- preprocessing
does not count -- a candidate holds one or two, and when it holds two, the second is a
different op from the first. `pca(50) -> umap` is accepted; `pca(50) -> pca(20)` and
`pca(50) -> pca(20) -> umap` are refused. The reasons: PCA is nested in d (section 3.5),
so a PCA of a PCA is the smaller PCA taken directly, and the extra stage spends a place in
the chain while changing nothing; and a third reduction has no useful occupant, since only
PCA may stand before the terminal stage and a second PCA adds nothing. A limit also keeps
every candidate readable as a hypothesis -- at most "one pre-step, then one method" -- and
keeps single-factor pairs such as `umap` against `pca50 -> umap` clean. The check is
structural and needs no data, so it belongs in `validate_stages`, which runs at
registration and again at execution; its refusal should name the stage and say to drop
the repeated or surplus reduction.

With the roles as they stand this has a corollary: every two-reduction candidate is
`pca -> X` with X not PCA, so any candidate ending in PCA holds exactly one reduction. That
narrows the gap in `no_linear_baseline` but does not close it -- were kernel PCA ever made
intermediate, `kernel_pca -> pca` would pass this rule and still pass as the linear
baseline, so the baseline check still needs strengthening as queued.

**One test, and it is an implication.** Assert that every op declaring `intermediate` is
`stochastic: false`. That is true independently of any downstream, so it is not a
judgment, and it catches the plausible future mistake: Laplacian Eigenmaps and LLE are
cheap and look like reasonable pre-steps, and widening the role for one of them would
otherwise land silently. The test is vacuous against the registry as it stands, which is
the point -- its value is prospective.

*Rejected: a parity test asserting the role equals `deterministic and cheap`.* That
formula is the absolute cost threshold this section has already rejected, and yet its
result -- `{pca}` -- is exactly right today. A test would pass, keep passing, and thereby
convert a coincidence into an invariant; the first method added afterwards would find the
test demanding the wrong edit with a green light behind it. A field that agrees with a
formula known to be flawed is the most dangerous moment to write a test, not the safest.

The division that follows: the field carries judgment, the test carries what is not
judgment, and what neither can settle is settled by running the experiment.

**Two declarations this calls for**, each on its own merits rather than bundled. A
machine-readable `cost_in_n`, because `complexity` is prose and no check can read it. And
`requires_connected_graph`, replacing the hardcoded `{isomap, lle, laplacian_eigenmaps}`
in the validator -- and deliberately *not* named for building a neighbour graph, which is
a different predicate: UMAP builds one and tolerates disconnection, t-SNE uses a dense
affinity, Diffusion Maps uses a dense kernel. One flag for both jobs would repeat the
conflation the change is meant to remove.

**Deferred to section 7, and recorded as the correct mechanism rather than a nicety.** A
per-chain cost check inside `validate-plan`, evaluating the inequality above against the
profile. It would refuse `kernel_pca -> umap` by naming the actual problem -- this chain
costs more than the method it wraps -- instead of refusing a category, and it would admit
any future pairing that does satisfy the inequality without a data edit. It needs per-op
cost functions rather than complexity strings, which is why it is not day 11 work. When
it lands it takes over the Budget purpose entirely, and `roles` shrinks to whatever
remains of the representation purpose -- possibly nothing, at which point the field
should be deleted rather than left standing.

### 3.10 Preprocessing: what every candidate shares, and what only some do

*Added on day 10.* Preprocessing has two layers. The **base preprocessing** is shared by
every candidate, and its output is the Reference every candidate is scored against.
**Candidate-specific stages** sit between the base preprocessing and a candidate's first
reduction. Both layers are now fixed by rule rather than chosen freely. The agent decides
the two facts the rules take as input -- whether the values are raw counts, and whether
the features are all of one type -- and logs each decision with the evidence keys it rests
on.

**The base preprocessing, in order.**

1. *Drop constant features, always, and first.* A feature is constant when

   `max(x) - min(x) <= eps * max(1, max_i |x_i|)`, with `eps = 1e-12`.

   The test compares the range with the magnitude, not the variance with zero. A variance
   computed in floating point is rarely exactly zero for a constant non-zero column
   (defect 19 in the day 10 log), whereas the range of stored identical values is exactly
   zero. The tolerance also catches a column that is constant only up to rounding in some
   earlier arithmetic. The floor of 1 inside the maximum makes the test absolute for
   features whose values are all below 1 in magnitude, which catches a column that is
   zero up to rounding; a purely relative test would keep it. The floor has a cost: a
   feature measured in units so small that its genuine range is below `1e-12` is dropped.
   Tested on day 10: values near `1e-13` varying by half were dropped, while values near
   `1e-9` and a sparse column holding 5 in a tenth of its rows were kept; a column equal
   to 0.1 only up to rounding was dropped, as intended. So the record lists every
   feature the tolerance dropped whose range was not exactly zero, which makes a wrong
   drop visible. The step comes first so that no later transform or z-score ever sees a
   constant column. The profile's count of constant features uses the same test.
2. *When the values are raw counts: `normalise_total`, then `log1p`.* Kept in scope. The
   agent tells the user it is doing this, twice. At planning, in the conversation and
   before anything runs, it says that the values look like raw counts, cites the evidence
   (non-negative integers, sample totals varying so many-fold), and says what it will do:
   rescale each sample to the median total, then take `log(1 + x)`. It adds that data
   already transformed should say so, or be supplied transformed. The report's
   preprocessing section repeats this. Overriding the default still needs a logged
   reason, as before.
3. *When the features are of mixed types: z-score every feature.*

**Deciding whether the features are of one type.** Examples of one type are genes, or
intensities at a series of wavelengths. An example of mixed types is age, sex, height and
weight in one table. The matrix alone cannot settle this. The profile records the signals
available:
- feature names and the file format, where an `.h5ad` file of gene symbols is strong
  evidence of one type;
- whether binary or categorical columns sit alongside continuous ones, which means mixed
  types;
- the spread of feature standard deviations, which is weak evidence, because gene
  expression spans orders of magnitude and mixed features can share a scale by chance.

The agent makes the call as a logged decision citing those keys, and a user who declares
the answer when starting the analysis overrides it. The decision is made after profiling
and *before* reconnaissance, because the Probe representation depends on it (below). So
where the evidence is unclear and the user has declared nothing, the agent asks before
reconnaissance rather than at the checkpoint that follows it; the question counts toward
the checkpoint's two. Under `--auto` it takes the default and records it. When the
evidence is unclear the default is *mixed*, because the two errors are not equally bad. Wrongly z-scoring
features of one type inflates noise features. Wrongly leaving mixed features unscaled lets
the feature measured in the largest units dominate every distance and every component.

*A known limitation, deliberately not fixed.* A binary or categorical column such as sex
is z-scored like a continuous one. Euclidean distance on mixed data is a compromise, and
the registry holds no method made for categorical features. The report's limitations
section states this whenever the features were judged mixed.

**Candidate-specific stages, by rule.** *Revised later on day 10; the day 10 log gives
the earlier rule, which selected only before a linear first reduction.* The rule looks at
the candidate's *first* dimension-lowering stage -- a reduction or a visualization method
-- because selection happens once, before that stage, and shapes only what it sees. In
`pca -> kernel_pca(kernel=poly)` the first stage is PCA, so selection applies, and the
polynomial kernel then sees PCA's output rather than the raw features.

*Which first stages select.* Those that work through the data's Euclidean geometry. For
two independent samples the expected squared distance is a sum over features,
`E ||x_i - x_j||^2 = 2 * sum_f Var(f)`, so a feature's share of the distances equals its
share of the variance. Ranking by variance therefore ranks features by how much they shape
the distances, and dropping the low-variance ones changes the distances little while
removing their noise. A dot-product kernel breaks that link, since a dot product depends on
the features' means as well as their spread, and so does a cosine or a correlation, which
rescale each sample.

| First stage | Selects |
|---|---|
| `pca`, `sparse_pca` -- linear | yes |
| `mds`, `isomap`, `lle`, `laplacian_eigenmaps`, `diffusion_maps`, `tsne` -- built on Euclidean distances in these executors | yes |
| `kernel_pca` with `kernel = rbf` | yes |
| `umap` with `metric = euclidean` | yes |
| `phate`, `trimap`, `pacmap` at a Euclidean metric, their default; each capability record declares the metric used | yes |
| `kernel_pca` with `kernel` poly, cosine or sigmoid | no |
| `umap` with `metric` cosine, correlation or manhattan -- manhattan sums unsquared differences, whose link to variance is only approximate | no |
| a GPLVM, not in the registry, recorded for when one is added | no |

So the rule turns on the parameters actually set, not only on the op. The rational
quadratic kernel, which also contains a Euclidean distance, was considered for kernel PCA
and not added.

| Features | First stage | More than 2,000 features | Candidate-specific stages |
|---|---|---|---|
| Mixed types | any | either | none (z-scored in the base) |
| One type | does not select | either | none |
| One type | selects | no | none |
| One type | selects | yes | `select_variable_features(n_features=2000)`, then z-score |
| Linear baseline | `pca` | either | none |

Selection ranks features by variance, which is now its only criterion: `dispersion` is
dropped. It runs after `log1p` where the counts rule applies, as the registry's caution on
selection already requires.

*The linear baseline never selects features*, so it stays a pure anchor: the base
preprocessing followed by a single `pca`. Whether it may carry any candidate-specific
stage at all, a subsample for instance, is the question about `no_linear_baseline` left
open on day 10.

*Why the scores stay comparable.* Candidate-specific stages do not change the Reference.
A candidate that keeps 2,000 of 20,000 features is still scored against the full-feature
Reference, so the battery measures the cost of discarding the rest rather than hiding it.

*The asymmetry is deliberate.* Features of one type keep their native scale when nothing
is selected, since their variances are comparable and meaningful. After selection they are
z-scored. This is by design. One consequence is accepted with it: a selecting PCA
candidate and the linear baseline differ in two factors at once, selection and scaling, so
the pair measures the combined effect and cannot isolate either.

**Enforced, not advised.** These rules are invariants, so `validate-plan` refuses a
candidate that departs from them:
- one missing the selection and z-score the rule requires;
- one carrying a selection or z-score the rule forbids;
- one selecting any number of features other than 2,000.

Fixing the number also closes one route to a degenerate baseline:
`select_variable_features(n_features=2) -> pca` registered on day 10 with no finding.

**Z-scoring means centring and scaling.** Dividing each feature by its standard deviation
without centring was considered. It gives the same result for PCA, which centres its input
itself, and for every method that works from Euclidean distances, which do not change when
a constant is added to a feature; and it keeps a sparse matrix sparse. It was not adopted.
The cost of that is memory. Centring makes every zero non-zero, so at n = 107,000 the
2,000 selected features become a dense 1.7 GB matrix against the 2 GB budget, and PCA's
own centred copy doubles it. The validator's memory estimate must count the dense output
of z-scoring, and refuse such a candidate at registration rather than let it fail in
execution.

**The Probe representation follows the same rule.** Decided on day 10. Reconnaissance
chose its Probe representation by a rule of its own: raw counts, then normalise and log;
feature standard deviations spanning more than 100-fold, then z-score. The base
preprocessing now keys on the feature-type decision rather than that spread, and left
alone the two rules would disagree -- a table of age, sex, height and weight whose scales
happened to span less than 100-fold would be probed unscaled and analysed z-scored, so the
evidence would describe a matrix the analysis never uses. The Probe representation
therefore applies the base preprocessing's default exactly: drop constant features by the
range test, normalise and log raw counts, z-score features of mixed types. Day 6's "one
rule for both" holds again, and `suggest_base` offers that one rule. The two remain
different things: the Probe representation is fixed by the rule and discarded once the
measuring is done, while the base preprocessing is the agent's, may be overridden with a
logged reason, and survives as the Reference. The spread of standard deviations stays in
the profile as one weak signal toward the feature-type decision. A feature-type decision
changed after reconnaissance has run -- by the user at the checkpoint, say -- makes that
reconnaissance stale, and it is run again before planning.

### 3.11 Two purposes: a representation, or a picture

*Added on day 10.* A run has one of two purposes. Its **purpose is representation** when
the deliverable is a low-dimensional representation that downstream analysis can use --
clustering, regression, testing. Its **purpose is visualization** when the deliverable is a
picture. The two call for different methods, different comparisons and different reports.

**Two classes of method, fixed by the registry.** Each reduction op is declared one of two
things, and the declaration is a property of the op, not a judgment made per run.
- *Reductions*, the methods whose output is a representation: PCA, sparse PCA, kernel PCA,
  MDS, Isomap, Diffusion Maps, Laplacian Eigenmaps, LLE.
- *Visualization methods*, whose output is coordinates for viewing only: t-SNE, UMAP,
  PHATE, TriMap, PaCMAP.

Some of these placements are arguable outside this agent. UMAP is run at 10 to 50
dimensions before density-based clustering in common practice, and TriMap, PaCMAP and
PHATE are designed to keep more global structure than t-SNE. Inside this agent the
classification is strict all the same: the five are visualization methods without
exception, and the reductions are reductions without exception -- a reduction run at d = 2
to draw a picture is still a reduction.

**Class and emphasis are separate axes, and must not be confused.** Class says what the
output may be used for. Emphasis says which distances a method keeps: short ones, which
show who sits next to whom, or long ones, which show how far apart the groups are. The
axes are independent: t-SNE is a visualization method emphasising local structure,
PaCMAP a visualization method balancing the two, LLE a reduction emphasising local
structure, Isomap a reduction emphasising global structure. The focus question below asks
about emphasis. It never changes a method's class, and it never changes which methods are
eligible.

**The checkpoint asks at most three questions**, raised from two:
1. whether the features are of one type or of mixed types -- asked before reconnaissance,
   and only when the evidence is unclear (section 3.10);
2. the purpose -- representation or visualization, with representation the default;
3. the focus. In a visualization run: *clusters and neighbourhoods*, or *the overall
   layout*. In a representation run, framed by the downstream use: clustering or other
   neighbourhood-based analysis, or distances, regression and a map of the whole. Either
   way it moves the weighting (section 2.4).

Each unanswered question takes its default, which the agent chooses from the evidence and
records with its evidence keys; under `--auto` every default is taken. The purpose may
also be given as an argument to `/analyze`.

**A representation run.**
- *Portfolio.* Reductions only. Visualization methods are excluded by rule rather than
  rejected by judgment, so the requirement that every eligible method be nominated or
  rejected (defect 17 in the day 10 log) covers the reductions.
- *Ranking.* As sections 3.7 and 3.8 describe: d is priced, and the report names one
  winner and its close competitors.
- *Chains.* At most two reductions, the first PCA (section 3.9).
- *PCA's output dimension in a chain is tuned.* Call it k. It becomes a third block in
  section 3.5's procedure: a short grid declared in the plan, starting from the spectrum
  suggestion, capped at 100. k carries no parsimony cost, because the k-dimensional output
  is not delivered and only the final output's score counts. PCA is nested in k, so one
  PCA fit gives every k, though each k needs its own fits of the second stage. When the
  second stage is nested in d, the grid over k and the multiplier is filled at the largest
  d and profiled. When it is not, the alternating scheme gains k as a third step -- but k
  and the multiplier both shape the input neighbour graph, so the two are strongly
  coupled, not weakly, and the search runs to its cap and checks the neighbouring cells
  rather than stopping at the first step that changes nothing.
- *Two plots per candidate.*
  - **Plot A** shows the candidate's own representation on the two principal axes of the
    delivered embedding. That is a rotation, so it changes no distance. It equals the
    first two coordinates wherever those are already ordered by importance -- PCA, Isomap,
    classical MDS -- and gives a meaningful view where they are not, as in metric MDS,
    whose axes have no order. The plot states the share of the embedding's variance its
    two axes carry.
  - **Plot B** shows UMAP fitted on that representation, fixed by rule: the same method,
    `n_neighbors` from the size-aware suggestion, the default `min_dist`, the run's seed,
    identical for every candidate, so that differences between two candidates' B plots
    come from their representations and not from settings. Rejected: the agent choosing
    the method, which after the results exist is choosing whichever picture looks best,
    and before them has nothing to choose from; and fitting both t-SNE and UMAP and
    keeping the higher score, which is a hidden rule for t-SNE, since it usually wins on
    trustworthiness. Plot B is not tuned, not a candidate and not ranked. It is omitted
    when the candidate's d is 2, since plot A is then the whole representation and a
    second layout only adds distortion.
  - Both plots are scored against the representation they draw, by trustworthiness and
    continuity. The battery already scores the representation against the Reference, so
    the report carries the whole chain: data, representation, picture.
  - Plot B stands outside section 3.9's chain rules. Those rules exist because a first
    stage adds its cost to a candidate and hides its variance in what is scored. Here the
    representation is computed anyway and plot B is not ranked, so neither applies.
- *Diagnostics.* The class facet and the Shepard diagram go to the winner and the close
  competitors, as section 3.8 has it.

**A visualization run.**
- *Dimension.* d = 2 throughout, so nothing in section 3.7 applies.
- *Portfolio.* The visualization methods, and the reductions at d = 2, with `pca` at d = 2
  the required baseline. A reduction at d = 2 keeps its class.
- *Chains.* Zero or one reduction before one visualization method, and that reduction is
  PCA: its cost and its variance belong to a scored candidate, so section 3.9 applies in
  full. Sparse PCA, Diffusion Maps and a GPLVM were considered and rejected as first
  stages. Sparse PCA is stochastic, and a method after it discards the sparse loadings
  that are its reason for being chosen. A GPLVM is not in the registry and fails both
  filters. Diffusion Maps is the serious case -- deterministic, with precedent in
  single-cell trajectory work -- but it holds an n-by-n kernel, it duplicates the
  neighbour graph the next method builds, and PHATE is essentially that pipeline as one
  method. It would be reconsidered for trajectory data on which PHATE is unavailable, or
  once the budget allows `diffusion_maps -> umap` to be tested against PHATE directly.
- *PCA's k is picked, not tuned.* It is the spectrum suggestion -- the elbow of the
  cumulative-variance curve, raised toward the count reaching 90% of variance, capped at
  100 -- and the agent may override it with a logged reason.
- *Nomination* works exactly as in a representation run, over the larger eligible set, and
  section 3.4's three roles apply across it.
- *Tuning.* Each candidate's fidelity hyperparameter is tuned by section 3.5 with d fixed.
  The objective is the pre-registered weighting, which follows the focus answer: heavy on
  trustworthiness and continuity for clusters and neighbourhoods, heavy on the Shepard
  correlation for the overall layout. Inside tuning, the weighting trades metrics across
  one method's possible pictures, which says what kind of picture is wanted. That is its
  only job in a visualization run.
- *No ranking.* A weighted total across candidates is not computed, because the weighting
  would decide the method by itself: weight local structure and t-SNE or UMAP wins,
  weight global structure and PCA or MDS wins. The individual metrics still compare, since
  each is on an absolute scale and computed identically for every candidate, so the
  report shows every candidate's scores metric by metric, in a fixed order with the PCA
  baseline first.
- *One plot per candidate*, its own 2D embedding. The class facet and the Shepard diagram
  go to every candidate, since there is no winner to reserve them for.
- *A second checkpoint, after the results.* The user is shown the pictures and chooses
  the one to adopt. With no reply, or under `--auto`, the agent chooses. A choice made
  after the results is not what pre-registration protects, and it is acceptable only
  labelled as what it is:
  - it goes in the decision log as a judgment, with reasons citing the metric profile and
    the focus answer -- "PaCMAP keeps neighbourhoods nearly as well as t-SNE, 0.93
    against 0.95, and global distances far better, 0.71 against 0.32";
  - the report says who chose -- "chosen by the agent after seeing the results; the user
    did not choose" -- so that it cannot be read as a measured ranking;
  - the chosen picture takes the headline position, and every other candidate's plot and
    scores still appear;
  - `rank` plays no part, so this choice is the only step that singles a candidate out.
- *Caveats per method.* What a picture's gaps and cluster sizes do and do not mean comes
  from each method's capability record, not from one fixed sentence about t-SNE and UMAP.
  The five claim different things, and none of them makes cluster sizes meaningful.

**Consequences elsewhere.**
- `tsne_exact` (section 3.7) is withdrawn. Its only use was t-SNE as a representation above
  three dimensions, and t-SNE no longer produces representations.
- In a representation run, section 3.4's local-structure role is filled only by LLE and
  Laplacian Eigenmaps. Both need a connected neighbour graph and both are stochastic, so on
  some data the role will be empty, and the run records that as a finding.
- The single-factor pair `umap` against `pca50 -> umap` belongs to visualization runs. With
  section 3.10's revised rule, a Euclidean `umap` selects and scales just as the PCA chain
  does, so the pair again differs in the PCA step alone. Under the earlier rule it differed
  in three factors -- selection, z-scoring and PCA -- which is why it was briefly described
  as the standard pipeline against the direct path. With a non-Euclidean metric the direct
  `umap` does not select, and the pair differs in three factors again; the agent should
  keep the metric Euclidean when the pair is meant to isolate PCA.
- An op is preprocessing, a reduction, or a visualization method. `CONTEXT.md` gains
  **Purpose** and **Visualization method**, and **Terminal method** is redefined by
  position alone: the old definition, "coordinates for viewing rather than a
  representation", ran position and purpose together, and it was wrong for Isomap, MDS,
  kernel PCA and Diffusion Maps, which are terminal-only yet produce representations.
- A gap in section 3.5: it names both Diffusion Maps' kernel width and its time `t` as
  fidelity parameters, but its procedure tunes one multiplier on one parameter. Which of
  the two is tuned is open.

*Open, deferred.* The cost of a direct path on many features. Measured on day 10 at
n = 6,000: UMAP's neighbour graph took 0.3 s at 2,000 features and 6.7 s at 20,000, its
layout about 7 to 8 s either way, and a whole direct fit 17.7 s at 20,000 features against
about 9 s after PCA to 50 or after selecting 2,000 features. At n = 107,000 and 20,000
features the dense matrix alone is 8.6 GB. Below 4,096 samples UMAP computes exact pairwise
distances instead, at a cost proportional to n^2 p, so benchmarks at small n measure a
different algorithm.

### 3.12 Every row covered, or the method refused

*Added on day 10.* A method that holds an n-by-n matrix -- MDS, Isomap, kernel PCA,
Diffusion Maps -- cannot run on a large dataset at all: at n = 107,000 the matrix alone is
91.6 GB whatever the number of features, so a pre-step that reduces d makes such a method
faster and never makes it possible (section 3.9). Until now the answer was a `subsample`
stage, and a candidate that took one produced an embedding of only the rows it kept.
Isomap on PathMNIST would have delivered a representation of 5,000 of 107,000 images,
been scored on rows drawn from those 5,000 while the other candidates were scored on rows
drawn from all 107,000, and could have won.

**The rule: fit on the subsample, then project every row; a method that cannot project is
refused above its limit.**
- *A method with a `transform`* is fitted on the subsample, and every remaining row is
  passed through the fitted pipeline, stage by stage, with the parameters the fit
  produced: the features selected, the z-score means and standard deviations, the PCA
  loadings, then the method's own `transform`. In scikit-learn that covers PCA, sparse
  PCA, kernel PCA, Isomap and LLE; UMAP and openTSNE have one too. Whether PHATE, TriMap
  and PaCMAP do is checked when their capability records are written.
- *Laplacian Eigenmaps and Diffusion Maps* get the Nyström extension, the standard way to
  place new points in a spectral embedding: a new row's coordinates are computed from its
  kernel similarities to the fitted rows and the fitted eigenvectors. Written in the
  toolbox, since neither implementation ships one.
- *A method with no transform and no standard extension* -- MDS, in the registry as it
  stands -- is refused when n exceeds its `scales_to`, and the refusal is recorded as a
  rejection citing `profile.shape.n_samples`. Placing new points one at a time against a
  fixed MDS configuration is known but not standard, and a stand-in nobody would
  recognise is worse than an honest absence. A generic fallback -- each new row at the
  weighted mean of its nearest fitted rows -- was also rejected: averaging pulls projected
  points toward their neighbours and shrinks the spread, which quietly degrades exactly
  the global structure these methods are chosen for.

**Subsampling only where it is needed.** A candidate may subsample only when its first
dimension-lowering stage would otherwise receive more rows than that op's `scales_to`.
PCA's limit is 2,000,000, so `subsample(n_samples=50) -> pca` -- one route to a degenerate
baseline, registered on day 10 without a finding -- is refused.

**What this does to scoring.** Every candidate now covers every row, so the battery's
2,000 scored rows are the same rows for every candidate: they are drawn from the same row
count under the same seed and labels. The comparison is paired, and the noise from scoring
candidates on different rows is gone. The score also describes what is delivered -- the
fitted rows and the projected ones together -- so a poor projection costs its candidate in
the ranking rather than hiding outside it. A candidate fitted on 5,000 rows still carries
the effect of having seen fewer rows, which is right: the subsample is part of its
pipeline, and the pipeline is what is compared. Tuning is unaffected; section 3.5's tuning
sample keeps its own derived stream. Rejected, and no longer needed: nesting the
subsamples of different sizes and drawing one shared scoring sample from the smallest.
Tested on day 10, stratified subsamples of 5,000 and 10,000 rows shared only 936 rows,
which is what first showed the problem.

**Cost.** Projection runs in chunks. Projecting 97,000 rows through a kernel PCA fitted on
10,000 needs a 97,000-by-10,000 kernel, about 7.8 GB built whole; Isomap's `transform`
needs the same shape of distance matrix. The fitted objects stay in the worker process
that fitted them, so nothing is pickled across a process boundary.

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

*Added on day 10.* **What the user takes away: `results/`.** The report, its figures and
the exported data live together in `runs/<id>/results/`:

```
runs/<id>/
  results/
    report.md, report.pdf
    figures/
    data/
  profile.json, recon.json, plan.json, plan.registered.json,
  decisions.jsonl, run.json, embeddings/, metrics/, data/
```

The figures move inside it because `report.md` links to them by relative path, so the
folder can be copied, zipped or shared alone with every link intact. The audit trail stays
at the top level, where section 2.2 places it and where the report's evidence keys point.

**The export, in `results/data/`: the winner only.** A representation run exports the
winner's representation at its chosen d; a visualization run exports the adopted
picture's 2D coordinates. Every Candidate's `.npy` stays in `embeddings/` regardless.
- *A CSV* whose first column is the sample identifier, a `fitted` or `projected` column
  saying which rows the method was fitted on (section 3.12), then the coordinates. CSV
  because `.npy` needs an extra package in R.
- *A manifest*: the pipeline and its parameter values, the seed, the chosen d, the
  features kept, the z-score means and standard deviations, whether the method can place
  new samples without refitting, and the number of rows fitted and projected.
- *Loadings* for PCA and sparse PCA, with the feature names attached.
- *No saved model objects.* A pickled scikit-learn or UMAP model often fails to load under
  another library version, so it would look reusable when it may not be; the manifest
  carries what is needed to refit.

The report gains a section pointing to the export. Its prerequisite is defect 21 in the
day 10 log: the loaders keep feature names but discard sample identifiers.

**Interaction.** One checkpoint after profiling, at most 1–2 multiple-choice
questions, each stating a default and never blocking. *Superseded on day 10:* at most
three questions, and a visualization run adds a second checkpoint after the results
(section 3.11). `--auto` suppresses them; the
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

Scheduled late (section 9 has the day), so slipping costs only the ablation.

---

## 7. Scope boundaries

**In:** PBMC3k and PathMNIST. The registry's ten reductions: nine library-backed —
PCA, Kernel PCA, Sparse PCA, MDS both metric and non-metric, Isomap, LLE with its four
variants under one op, Laplacian Eigenmaps, t-SNE, UMAP — plus Diffusion Maps, written
directly after `datafold` proved unusable against modern scikit-learn.

*Added on day 10: three visualization methods*, joining t-SNE and UMAP on that side of the
split between methods whose output serves downstream analysis and methods whose output is
for viewing only:
- **PHATE** (Moon et al., 2019): distances between diffusion potentials, embedded by MDS,
  aimed at preserving trajectories and progressions as well as clusters; package `phate`.
- **TriMap** (Amid and Warmuth, 2019): an embedding fitted to preserve the ordering within
  sampled triplets of points, aimed at global structure; package `trimap`.
- **PaCMAP** (Wang, Huang, Rudin and Shaposhnik, 2021): an embedding fitted on three kinds
  of pair -- near pairs, mid-near pairs and further pairs -- to balance local and global
  structure; package `pacmap`.

All three are stochastic, so section 3.9 makes them terminal-only. They are candidates only
in runs whose purpose is visualization; plot B in a representation run stays UMAP, fixed by
rule. Each needs a capability record, an executor, a size-aware suggestion for its
fidelity hyperparameter (section 3.5) -- `knn` for PHATE, `n_inliers` for TriMap,
`n_neighbors` for PaCMAP -- and a `scales_to` set by measurement.

*Installation, checked on day 10 without installing.* All three resolve against this
environment (scikit-learn 1.9.1, NumPy 2.5.3, SciPy 1.18.1): `phate` 2.0.0, `trimap` 1.2.0,
`pacmap` 0.9.1. PaCMAP needs `faiss-cpu` and `numba`, both with Windows wheels. PHATE's
dependencies are pure Python. **TriMap is at risk.** It requires `annoy`, which publishes
no wheel for Windows on Python 3.12, and no C++ compiler is on the path here, so installing
it means building `annoy` from source or finding another route. It also requires `torch`.
Resolving is not the same as working: `datafold` resolved too, and failed at import time
against modern scikit-learn. Each method is therefore to be imported and fitted once on the
synthetic data before its capability record is written.

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

- **Day 9** — An external review of days 7 to 9, and the five defects it found.

  A Codex review over `847d138...HEAD` — the whole of days 7, 8 and 9, 47 files and
  some 7,500 inserted lines. Five findings, all five real when checked against the
  code, two of them changing a decision rather than repairing an implementation of one.
  No false positives, which is worth recording: the previous external pass on day 6
  also landed, and reviewing a whole scope at once is what keeps finding the defects
  that sit between correct pieces.

  *A refusal that destroyed the evidence it was refusing to replace.* `embed` cleared
  the previous attempt's embeddings and metrics and then resolved the seed, which can
  refuse. A retry carrying a seed the run was not created with deleted the record of
  the previous failure and then declined to produce a new one, and the decision log
  does not close the gap: it records how an attempt ended, not why. The op, the error
  type and the message existed only in `embeddings/<id>.json`.

  The invariant was already written thirty lines above, over the `--in-process`
  refusal — a request the run cannot honour must leave the run exactly as it found it —
  and the budget and attempt-count refusals sit above the invalidation and keep it.
  One refusal sat below it. Every individual rule here was correct; the defect was
  only in where one of them sat relative to a destructive step. That is the third time
  this project has found a defect at a seam rather than inside a rule.

  *`/analyze` opened every new analysis on a refusal.* It said a Run is resumed or
  started and that "either way, the next command is `drtools status`". There is no
  either way: `status` reads a Run through `_require_run`, which refuses a path that
  does not exist. `profile-dataset` had the same circularity at the top — read your
  position from `status`, and only then, at step 1, run the `profile` that creates the
  directory `status` needs. Both now branch on the case. The refusal carries the route
  out too, which `_open_run` already did for its own case and `_require_run` did not.

  Prose only, for the two documents. The branch a skill takes is not mechanically
  checkable, and a test that pattern-matched the wording would pass for the wrong
  reason; the refusal is tested instead.

  *A declared ceiling was not frozen, and this changes a decision.* Day 8 recorded that
  a plan declaring `max_candidates` below the budget's ceiling "enforces nothing by
  itself, since a bound the agent sets on itself is not a bound, but under the hard
  ceiling it is a checkable claim about self-restraint — the same thing pre-registering
  the weighting buys for the evaluation". Re-registration froze the budget, the
  weighting and the base preprocessing and left this open, so a run could declare 2,
  register 2, then re-register at 5 and spend 5. The `register_plan` record carried the
  digest, the weights and the candidate ids but not the declaration, and
  `plan.registered.json` is overwritten by the next registration — so afterwards
  nothing showed that 2 had ever been claimed.

  That day 8 sentence is superseded, and both halves of it are why. *Checkable* needs
  the claim to survive in the append-only log. And the comparison to the weighting is
  not decoration: the weighting is frozen by refusal, so a field said to buy the same
  thing cannot be freely rewritten. The declaration is made before any candidate has
  run, which is what separates a claim about restraint from a report of what the run
  turned out to need. Loosening now refuses, tightening stays legal, omitting the field
  counts as loosening rather than as saying nothing, and `max_candidates` joins
  `LIFECYCLE_FIELDS` so the agent's own write route cannot forge it.

  Rejected: recording the declaration and leaving it loosenable, which was the smaller
  change and keeps the day 8 sentence intact. It buys a record of the loosening and
  nothing that stops it, and a pre-registration the run may revise is the thing this
  project has refused everywhere else.

  *One re-plan round, enforced rather than reported.* Section 3.1 permits one round
  once the portfolio has been attempted, and `evaluate-embeddings` says so. Nothing
  refused a second. `status` computed `replan_round_spent` correctly and spent it on
  one decision — whether a run with no successes goes back to plan or on to report —
  while `validate-plan` registered a third and fourth extension without comment.

  The refusal reuses `status`'s own scan rather than restating the rule. The round has
  four distinctions in it and every one lives at an edge, so a second implementation
  would disagree with the first somewhere and the two commands would then describe
  different runs.

  This exposed a test that was not testing what it said. The exhaust-abandon-replace
  cycle test described exhausting an id, abandoning it and registering a replacement,
  and never recorded the abandonment — which under the design's own definition made
  every cycle a re-plan round, and it passed only because nothing refused one. It now
  abandons through `log-decision` and still ends where it did, at the Ceiling. The
  Ceiling remains what bounds the replacement route; the round bounds the other one.

  *Malformed JSON escaped the error contract.* `--plan` and `log-decision --json` both
  parsed with a bare `json.loads`, and `main` catches neither `JSONDecodeError` nor the
  `ValueError` it derives from, so a mistyped brace left a traceback and exit 1 — not
  one of the four documented exit codes, and no route out. `_plan_from` had already
  closed this class for pydantic's `ValidationError`, which is raised only once the
  document has parsed. The message names the source, because `@path` and an inline
  document are two of them and fixing the wrong one is a loop.

  Day 10's brief is unchanged: the report template and the first end-to-end run. These
  five were repairs to days 7 to 9 and are recorded here rather than spending a day.

  385 tests.

- **Day 10** — The report contract, and what the toolbox contributes to a document the
  agent writes.

  Briefed as "report template + pandoc render". The notes fixed the requirement and not
  the mechanism — section 5 fixes the nine-section skeleton and names pandoc, section
  2.3 fixes that the report is generated from the Decision log, and nothing anywhere
  said whether `drtools` contributes to the document at all. That is the *make X hold*
  shape, so the day was architectural and got a design before any code:
  `design/specs/2026-09-20-report-contract.md`, then a plan, then six tasks.

  *The decision: the toolbox emits the numbers; the agent writes the prose.* Generated
  sections carry fenced blocks the toolbox owns and can regenerate; everything outside a
  fence is the agent's. The property this buys is that **the agent never retypes a
  number** — a figure in the report cannot disagree with the artefact it came from,
  because there was no opportunity to transcribe it. That is stronger than citation
  integrity and composes with it: citation integrity says a cited key resolves, and this
  says the value printed is the value it resolved to.

  Rejected: *render only*, the agent writing everything and `render` just calling pandoc,
  which leaves the grounding resting on prose discipline and makes every number a
  transcription. Rejected: *fully generated*, which cannot write sections 8 and 9 without
  squeezing paragraphs of interpretation through a field meant for one-line rationales,
  inverting which of the log and the report is the record. Rejected: *write then verify*,
  which needs a number-extraction parser over free prose — and a parser that misses one
  case is worse than no parser, because it reports a clean check over a document it did
  not fully read. Emitting achieves the same end with no parser at all.

  Section 8, Interpretation, has no block. Nothing in the Run grounds one, and a
  generated contribution there would lend the appearance of derivation to the one section
  that is entirely the agent's judgment — the failure mode section 2.3 already had to
  narrow a claim over once.

  *Rendering last is a property, not an instruction.* Two chains go stale and ordering
  fixes only one: the PDF against the Markdown, and the Markdown against the Run. The
  second is the dangerous one, because it yields a clean, current-looking PDF of numbers
  the Run has moved past. So `render` recomputes what each block would hold and refuses
  if any would change. It does not refresh on the agent's behalf — silently changing the
  numbers while producing the PDF would let the two artefacts a reader compares differ,
  and section 5 already settles that the Markdown is the source of truth.

  *Reading real artefacts corrected the design twice, and simplified it once.* The spec
  said section 4's hyperparameters come from the registered Plan. They come from
  `embeddings/<id>.json`: the Plan holds what was asked for, and the record holds what
  ran, with registry defaults filled in and `param_provenance` marking every value
  `specified` or `registry_default`. Only the record can say which hyperparameters the
  agent chose. And `figures.json` stores absolute paths, so the block relativises them —
  a report carrying an absolute path breaks the moment the Run is moved and pandoc cannot
  resolve the image.

  The simplification: the spec carried an exception for a Run where every Candidate
  failed. It needs no code. `status._next_stage` already returns `report` once nothing
  has succeeded and the re-plan round is spent, and `plan` while the round is unspent.
  Both are the right answer, so `report` checks one condition and has no special case —
  which is the same discipline as reading `status` for readiness at all, rather than
  asking the question a second way and letting the two answers drift.

  *A defect in the fence contract, found by the tests of the task after it.* `fence`
  wrote the body verbatim, but `parse_blocks` strips trailing newlines, because the
  closing comment sits on its own line. So any body ending in a blank line failed its own
  digest the instant it was read back, and `--refresh` refused a block nobody had
  touched. `_block_ranking` produces exactly such a body whenever a Run has no ranking
  notes. The task that introduced it had six tests and every one used a body ending
  mid-line, so the round trip was never exercised at its edge.

  The fix is one canonical form applied wherever a body is written, hashed or compared.
  Half of it would have been worse than none: normalising only the digest makes the
  hand-edit check pass while leaving the staleness comparison weighing a stripped parsed
  body against an unstripped generated one, so every block reads as changed on every
  refresh and `refreshed` stops meaning anything the agent can act on. Two tests pin the
  two halves.

  Day 10's second half — the first end-to-end run on PBMC3k — is not done. It carries one
  decision the design deliberately left open: whether PBMC3k gets derived Leiden
  reference labels. Labels from a PCA-neighbour-graph-Leiden pipeline would flatter
  embeddings preserving that same neighbourhood structure, so using them as a metric
  input would bias the ranking toward methods resembling the pipeline that produced them.

  *A design pass on the target dimension, and five defects, all found by reading.*
  Mid-day, before the PBMC3k run. A question about how the agent picks the number of
  dimensions to keep established that nothing picks it: the value is the registry default
  throughout, and the intrinsic-dimension estimate that should inform it only ever
  produced a warning. Section 3.7 is the result. It is new intent rather than a record of
  anything built, it carries three questions still open, and the work lands after day 10.

  Queued with it, none of them caught by a failing test:

  - Nothing refuses a terminal `n_components` above 3, so a candidate can be scored at
    full width and plotted from its first two columns -- the numbers and the picture
    describing different objects. Absorbed by section 3.7 rather than fixed beside it.
  - Nothing forbids a reduction inside `base_preprocessing`. The pipeline asserts the
    invariant in a comment, "by construction made only of preprocessing", and no code
    checks it. A `pca` there would make the reference an embedding, and every candidate
    would be scored against 2-D coordinates. Day 5's silent `prepare-reference` failure
    is the same shape: a reference that quietly becomes the wrong thing yields a full set
    of believable numbers.
  - `no_linear_baseline` tests only that some candidate's last op is spelled `pca`, so it
    enforces less than its own message claims. `[subsample(50), pca]` satisfies it.
  - Section 3.2 says the registry declares `can_be_intermediate` for PCA *and* kernel
    PCA; the registry gives `kernel_pca` `roles: [terminal]`. The field is terminal-only
    in the first version of that file and no commit message or log entry mentions it, so
    a judgment about kernel PCA cannot be told from the value its eight neighbours
    already had. Leaning to widen, which is what this document already specifies -- but
    only after the baseline check above is strengthened, since widening makes
    `[kernel_pca, pca]` legal and that candidate would pass as the linear baseline while
    being nothing of the kind.
  - The registry tells the agent a null `gamma` means `1/n_features`. The executor
    substitutes a median-pairwise-distance heuristic and records the provenance. The
    registry is the agent's only view of the library, so a capability record wrong about
    its own behaviour is a defect in itself.
  - Tie detection reads only the top two. `_notes` takes `scored[0]` and `scored[1]`
    and compares that one pair, so with scores 0.800, 0.790 and 0.785 the third
    candidate sits within 0.015 of the leader and goes unmentioned. The note then
    understates the tie it exists to report. It should walk down the list while the
    margin from the leader stays inside the threshold.
  - A tie has no representation in the data model, which section 3.8 now takes up. The
    defect proper is that `rank` detects the tie, states it in English, and then writes a
    single-winner record that four consumers act on -- including the decision log entry,
    where an unqualified `chosen` enters the append-only record.
  - `_block_limitations` decides which ranking notes are limitations by testing whether
    the words "tied" or "noise" occur in them. English is being used as a data field, so
    rewording a note silently drops it from section 9. The notes should carry a kind.
  - The registry's header and the day 3 entry both claim that adding a method is one
    entry plus one executor. It is not. `plan.py` holds a seven-name `EUCLIDEAN_METHODS`
    frozenset and an inline `{isomap, lle, laplacian_eigenmaps}`, and `heuristics.py`
    branches on four more names. A new method that missed them would escape every check
    meant to govern it, silently, and the executor-parity test covers none of it -- it
    was written for the declaration-implementation pair and the predicate sets
    accumulated after. Section 3.9 replaces one of these sets and the claim needs either
    a wider test or a narrower wording.
  - The registry caps `tsne` at `n_components = 3`, which is not the limit of the library
    that runs it. openTSNE's automatic choice uses FFT interpolation at 10,000 samples or
    more, and FFT refuses three dimensions with a runtime error, so `tsne(n_components=3)`
    on large data passes `validate-plan` and fails in execution. Below 10,000 it uses
    Barnes-Hut, which accepts more than three and only warns of segfaults, and a segfault
    here becomes a `crashed` outcome with nothing recorded. The real limit is a function
    of n and of the approximation. Section 3.7 now says what to do above it.
  - The suggested `n_neighbors` for LLE contradicts its own rationale. On a graph that
    reconnaissance found disconnected at k = 15, the connectivity rule raises the value
    to 45, the LLE cap then clamps it to 12, and the rationale still says 45 -- so the
    suggestion reproduces a configuration already known to fail. The two rules genuinely
    conflict on that data, and the honest output is that LLE is unsuitable, which is a
    rejection with a reason rather than a clamped number. The same cap of 12 also breaks
    Hessian LLE's neighbour minimum from d = 4 upward, once d varies per candidate.
    Verified by running the suggestion on a synthetic profile.
  - The registry tells the agent that an empty Diffusion Maps bandwidth means "the median
    squared k-NN distance". That is the heuristic the executor's own docstring documents as
    failing -- correlation 0.22 with the manifold parameter at n = 400 -- and the executor
    uses the kernel-sum scaling criterion instead. Like the `gamma` entry above, but worse:
    the text does not lag the code, it advertises the rejected method. Both exist because
    each default is described twice, once in prose and once in code, and the executor
    already records which method it used in a `*_source` field that the registry could
    point at instead.
  - "Override a suggestion with a logged reason" cannot be checked. `suggest-params`
    returns its answer to the agent and writes nothing into the run, and provenance has two
    states, so a value that followed the suggestion and one that overrode it are both
    recorded as `specified`. Persist the suggestion and give provenance a state for each.
  - `validate-plan` does not check LLE's neighbour minimum, although the variant, d and
    `n_neighbors` are all in the plan. The executor refuses instead, so a mistake visible
    before anything runs spends one of the candidate's two attempts -- the same shape as
    the t-SNE limit.
  - `validate-plan` never resolves the evidence keys a plan cites. Only `log-decision`
    does. A rejection citing `recon.this.key.does.not.exist` registered with no finding,
    and the report prints rejection evidence verbatim, so a broken citation goes straight
    into section 3 -- which `CONTEXT.md` calls "a broken rationale, not a missing value".
    Refuse unresolved keys at registration, as `log-decision` already does.
  - A rejection's method name is not checked against the registry: the same test plan
    rejected `not_a_method` without a finding.
  - Nothing requires every reduction to be accounted for. A plan with a single PCA
    candidate and nothing rejected registered as valid with no findings, so nine methods
    went unmentioned and the selection the rejections exist to show was invisible. With
    ten reductions, requiring each to appear in a candidate or in `rejected` is small.
  - Nomination is held to a lower standard than rejection. A rejection must cite evidence
    or draw a warning; a candidate cannot cite any, since `CandidateSpec` has only an
    optional free-text `rationale`. The skill's first rule, argue from the evidence, is
    enforced weakly for what is not run and not at all for what is. Give candidates an
    `evidence` field, resolved like the rest.
  - `drop_constant` keeps a feature when its variance is above zero, and floating-point
    rounding leaves a constant non-zero column with a variance just above zero. Over
    1,000 rows the dense variance of a column of 0.1 was 2.0e-30, and of 7.7, 1.5e-26; the
    sparse formula, mean of squares minus squared mean, gave 1.5e-15 for 0.3 and 9.4e-13
    for 7.7. Each such column survives, and a later z-score divides by a standard
    deviation near 1e-7, turning rounding error into a feature with unit variance. Nothing
    downstream notices, because the result is not NaN. The profile's count of constant
    features tests `std == 0` and has the same flaw. Section 3.10 replaces both with a test
    on the range.
  - A weighting may name kNN label preservation or silhouette on data with no labels.
    `_check_weighting` checks names, signs and the sum, and never whether the metrics it
    weights can be computed; the profile already knows whether labels exist. The weight is
    then dropped and redistributed at ranking, with a note, so the ranking answers a
    different question from the one registered. Refuse it at registration.
  - The loaders keep feature names -- a DataFrame's columns, an AnnData's `var_names` --
    and discard sample identifiers: the DataFrame's index, `obs_names`. An embedding's
    rows follow the loaded order, so matching them back to samples rests on that order
    never changing, and nothing exported can name its rows. Carry the identifiers through
    the loader contract and the cache.

  Also noted, not defects: section 3.4 now says what spanning families means and that it
  is unchecked; section 2.2 now says a report is reproducible only conditional on its
  registered plan, which the report's limitations should state -- a line for
  `_block_limitations` when the pass closes; and section 6's "day 12" was stale after the
  re-index and now points at section 9.

  *A limit on chain length, decided.* Nothing capped how many reductions a candidate may
  chain, and `pca(50) -> pca(20)` registered. Now: at most two reduction stages per
  candidate, and the second a different op from the first, enforced in `validate_stages`.
  Section 3.9 has the reasoning. Rejected: a cap with no rule against repeats, which would
  still admit the PCA-of-PCA stage that changes nothing.

  *Preprocessing, decided.* Section 3.10 fixes both layers by rule. The base preprocessing
  drops constant features by a range test with a tolerance, applies `normalise_total` and
  `log1p` to raw counts and tells the user so, and z-scores features of mixed types.
  Selecting 2,000 features by variance, then z-scoring, applies only to features of one
  type feeding a linear first reduction, and never to the linear baseline. The rules are
  enforced at registration. Considered and withdrawn: removing the value transforms as
  outside a dimension-reduction agent's scope. Users do supply raw counts, so the
  transforms stay and the agent says when it applies them. Not adopted: scaling without
  centring, despite its memory saving. Dropped: the `dispersion` criterion. Rejected for
  the constant test: exact equality, which misses a column constant only up to rounding,
  and a purely relative tolerance, which misses a column zero only up to rounding. The
  Probe representation now follows the same rule, which moves the feature-type decision
  ahead of reconnaissance. Rejected: keeping reconnaissance's own scale-spread rule, under
  which the evidence could describe a representation the analysis never uses.

  *Runtime out of the score, decided.* `runtime_s` was a weightable battery metric, the
  only one scaled within the run. Tested through `rank_candidates` at weights 0.9
  trustworthiness and 0.1 runtime: adding a slow, poor candidate reversed the order of
  the two ahead of it, and 0.1 s outweighed a 0.02 lead in trustworthiness. Now measured
  and reported for every candidate, with whether the pipeline can embed new samples, and
  never weighted; section 2.4 has the reasons, section 3.8 the one place cost may speak,
  between close competitors when reuse is the purpose. Rejected: keeping runtime with an
  absolute scale, which would fix the reversal but not the sunk cost, the replay, or the
  reward for subsampling.

  *Three visualization methods added.* PHATE, TriMap and PaCMAP join the registry as
  visualization methods, candidates only when a run's purpose is visualization. Section 7
  records what each is and what installing them involves; TriMap's hard dependency on
  `annoy`, which has no Windows wheel for Python 3.12, is the one known obstacle.

  *Two purposes, decided.* Section 3.11. A run's purpose is representation, the default,
  or visualization, asked at the checkpoint, whose limit rises from two questions to
  three. The registry classes each reduction op as a reduction or a visualization method,
  strictly, and that class is kept apart from what structure a method emphasises. A
  representation run nominates reductions only, ranks as before, tunes PCA's output
  dimension in a chain with a cap of 100, and draws two plots per candidate, the second a
  UMAP fixed by rule. A visualization run works at d = 2, admits zero or one PCA before a
  visualization method with its output dimension picked rather than tuned, tunes by a
  weighting that follows the user's focus, ranks nothing, and adopts one picture at a
  second checkpoint after the results -- the user's choice, or the agent's labelled
  judgment. `tsne_exact` is withdrawn. Rejected: plot B chosen by the agent or by score;
  ranking visualization candidates on a weighted total; Sparse PCA, Diffusion Maps and a
  GPLVM as first stages; classing methods by their emphasis rather than by declaration.

  *Selection rule widened.* Section 3.10 first said: select 2,000 features by variance and
  z-score them only when the features are of one type, number more than 2,000, and feed a
  *linear* first reduction; a nonlinear first reduction, kernel PCA included, got neither.
  Now selection applies wherever the first dimension-lowering stage works through the
  data's Euclidean geometry -- the linear methods, every method built on Euclidean
  distances, kernel PCA with the RBF kernel, and UMAP and the other visualization methods
  at a Euclidean metric -- and not before kernel PCA with a dot-product or cosine kernel,
  UMAP at a non-Euclidean metric, or a GPLVM. The reason is that a feature's expected
  contribution to squared Euclidean distance is twice its variance, so variance ranks
  features by their effect exactly where Euclidean distance is used. In a chain only the
  first stage counts. The other conditions stand: one type, more than 2,000 features, keep
  2,000, then z-score, and never for the linear baseline. Rejected: adding a rational
  quadratic kernel to kernel PCA; requiring every stage of a chain to qualify.

  *How the weighting is set, decided.* Section 2.4. A default of 0.25 trustworthiness,
  0.25 continuity and 0.5 Shepard correlation; departures justified by evidence keys that
  resolve; a focus question in both kinds of run, framed in a representation run by what
  the representation is for; the label metrics weighted only when labels exist, and then
  allowed in either kind of run. Rejected: dropping silhouette from representation runs
  as a display property -- kept, since with labels it measures how separated the known
  groups are, which a downstream classification can use. Open: the weights each focus
  answer maps to.

  *Tuning objective and d rule, separated.* Section 3.5 profiled every nested method's grid
  by the battery and then applied the d rule to the profiled curve. For a candidate whose d
  rule reads its own criterion, that curve is stitched from different fits -- Isomap's
  residual variance at k = 5 for some d and at k = 20 for others -- and the rule reads the
  switch points as structure. Now the battery chooses the hyperparameter for every
  candidate, each candidate's own rule chooses d on one fit's curve, and the ranking stays
  the battery. Rejected: letting the own criterion choose the hyperparameter as well, since
  the hyperparameter builds the criterion's target and some criteria reward degenerate
  settings; it stays open as a condition for methods added later. Also corrected: section
  2.4 had said the weighting is every candidate's Q(d).

  *Coverage and the results folder, decided.* Section 3.12: a subsampled candidate is
  projected to every row by its `transform`, or by a Nyström extension for Laplacian
  Eigenmaps and Diffusion Maps; a method with neither, MDS today, is refused above its
  limit; subsampling is allowed only above a method's limit. Every candidate is then
  scored on the same rows. Section 5: `results/` holds the report, its figures and the
  winner's export, with sample identifiers, a manifest and PCA loadings. Rejected: a PCA
  pre-step as a way to fit MDS on large data, since it reduces d and not n; exporting
  only the fitted rows; a nearest-neighbour fallback for methods without a transform;
  nested subsamples with a shared scoring sample; pickled models in the export.

  *d_max and the grid, decided.* Section 3.7: d_max = min(100, p - 1, n - 1) for every
  reduction, each method's structural limit on top, a grid thinning from 2 to 100, and a
  candidate stopped at d_max reported as capped. The penalty on d stays open: section 3.7
  records the two proposals discussed, a rate scaled by p and the one-standard-error rule
  with a reported reversal rate, and why the question is hard.

  *How d is chosen and priced, decided.* Section 3.7. Within a candidate whose d rule
  reads the battery -- LLE and sparse PCA in a representation run -- the Kneedle elbow on
  log d, taken on the curve's running maximum and never at an end of the grid; a curve
  flat to within its own flatness threshold, proposed 0.03 and not tied to delta, takes
  the smallest d; with no interior elbow, the smallest d within 0.10 of the candidate's
  best, reported as capped when the curve is still rising at the grid's end. Plain
  Kneedle was tried first and chose d = 2 on LLE's dipping curve. Across candidates: a
  non-inferiority margin of 0.02, the smallest d winning among those inside it, which also
  settles section 3.8's margin. Reported: the path of winners over the value of a
  dimension, as sentences, and the jackknife standard error of each score. Rejected: a
  chosen rate per dimension, scaled by p or not, since its effect on the ranking cannot be
  seen in advance; and the one-standard-error rule, which on the digits data chose d = 50
  because fidelity rises to the Reference and its standard error vanishes there.

  *Weights with labels, decided.* Section 2.4 gains a second default for trusted labels:
  0.175, 0.175 and 0.35 on the unsupervised metrics, 0.20 on kNN label preservation and
  0.10 on silhouette, the numbers awaiting confirmation. Trusted means supplied with the
  data and not derived from it; derived labels, and labels in an analysis meant to find
  new groups, carry no weight and are reported only. Rejected: leaving the label weights
  to the agent, which the unsupervised default was introduced to stop.

  *Section 3.5 rewritten.* It said: profile-derived defaults the agent may override with a
  logged reason; "a small sweep of 2-3 values only for the top-ranked candidate, budget
  permitting"; and that library defaults were "rejected outright". The first stands. The
  sweep for the top-ranked candidate alone could not have fed the ranking without giving
  extra tries to the one already ahead, and it was never built; it becomes tuning for every
  candidate before ranking, with d chosen in the same procedure, plus an optional
  sensitivity check that never re-ranks. The third overstated what exists -- eleven of
  nineteen hyperparameters run at the registry default -- and is replaced by an account of
  which kinds of hyperparameter should be tuned at all. The alternating procedure for
  methods not nested in d is modelled on one-step GEE and on DESeq2's dispersion-mean-
  dispersion sequence. Rejected: a full grid for every method, which costs the same as
  alternating only where the method is nested in d; choosing d first at the suggested
  hyperparameter, which picks too large a d; and tuning the raw value on a subsample,
  which does not transfer to the full n.

  *Seeds in section 3.5 corrected.* As first written it gave the refit "a fresh seed" and
  the tuning cells "a metric-subsample seed different from the one the final battery
  uses". Both contradicted the run's rule of one immutable seed, and a seed chosen afresh
  would reopen the route the locked seed closes -- re-running a stochastic method until it
  scores well, the seed analogue of p-hacking, invisible in the record because every
  attempt looks legitimate. Corrected to leave the run's seed exactly as it is for final
  fits and the battery, and give tuning separate streams derived from it and recorded.

  Queued rather than fixed, deliberately: the question pass that produced them is still
  running, and the agreement is to collect what it finds and implement once. Sections 3.5
  and 3.7-3.9 together are far larger than the day 11 row anticipates, and they are
  ordered: 3.8 after 3.7, because pricing d changes the scores whose margin it measures,
  and the tuning in 3.5 shares its machinery with 3.7's choice of d. The schedule needs
  revisiting once the pass closes.

  416 tests, 61s.
