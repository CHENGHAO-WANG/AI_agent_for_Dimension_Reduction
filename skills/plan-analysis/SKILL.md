---
name: plan-analysis
description: Use when a dr-agent Run has been profiled and reconnoitred and `drtools status` reports `plan` as the next stage, including when an earlier plan was refused and needs revising.
---

# Plan the analysis

Declare what will be compared, what will not be run and why, and how the results will
be judged — all before any Embedding exists. Registration freezes the weighting, so
everything here is a commitment made in advance.

## Argue from the evidence, not from the dataset's name

Read `profile.json` and `recon.json` from the Run. "n=2700 and sparse, therefore t-SNE"
is a lookup table wearing a costume. The spectrum's elbow, the intrinsic dimension, and
the neighbourhood graph's component count each change what is worth running:

| Evidence | What it licenses |
|---|---|
| Fast-decaying spectrum | Structure is largely linear; manifold methods will add little |
| Low intrinsic dimension, connected graph | Isomap and Diffusion Maps are justified |
| Disconnected neighbourhood graph | Rules out spectral methods before they crash |
| Sample totals varying many-fold | Normalise before anything Euclidean |

## Build the Plan

**Base preprocessing.** It follows a rule from the data decision `recon` recorded:
`drop_constant` first, always; then `normalise_total` and `log1p` for raw counts; then
`standardise` for features of mixed types. `drtools suggest-base --run-dir runs/<id>`
returns that base, the same rule Reconnaissance probed under, and persists it in the
Run. A base that differs from the rule registers only with `base_departure: {"reason":
..., "evidence": [...]}`, and `drop_constant` stays first whatever the reason. Its output
becomes the Reference every Candidate is scored against, so it holds preprocessing
only: a Reduction or Visualization method there is refused.

**When the values are raw counts, tell the user before registering.** Say that the
values look like raw counts and cite the evidence — non-negative integers, sample totals
varying so many-fold — and say what will be done: each sample rescaled to the median
total, then log(1 + x). Add that data already transformed should be declared as such,
or supplied transformed. Under `--auto`, put the same statement in the Decision log.
Report section 2 repeats it from the Run.

**Candidates.** Three to five, each a Stage list ending in a Reduction or a
Visualization method, deliberately spanning families: one Linear baseline, one
local-structure, one global-structure. The Linear baseline is required, and it is a
Candidate whose Stages are exactly one `pca` — without it there is nothing to measure
the nonlinear methods against. Every other Candidate cites the Evidence keys that make it
worth running, in its `evidence`, as a Rejection does.

**Feature selection, by rule.** When the features are of one type, more than 2,000
remain after the constants go, and a Candidate's first method works through Euclidean
geometry at the parameters set — `drtools methods` shows each method's `euclidean` —
the Candidate carries `select_variable_features(n_features=2000)`, then `standardise`,
directly before that method. Otherwise it carries neither, and the Linear baseline
never does. Registration refuses a Candidate that departs from this, and any other
`n_features`.

**Memory.** Registration estimates each Candidate's peak memory — the loaded data, plus
its heaviest Stage's input, output and PCA's centred copy — and holds it to half this
machine's memory. Over the limit, a Candidate with a method that can place new rows
fits on a `subsample`, which is then allowed; one whose method cannot is refused and
belongs in `rejected`. The finding names the Stage.

A Candidate holds at most two methods: one `pca` pre-step, then a different method. A
`subsample` is allowed only in front of a method that would otherwise receive more rows
than its `scales_to`; a method whose `new_rows` is `none` cannot be subsampled at all, so
above its limit it is a Rejection citing `profile.shape.n_samples`.

Read `drtools methods` for what each Op preserves, assumes, destroys, and where it
stops scaling. You cannot introspect the library; the Capability records are what you
reason over.

The best experiment available here is entering both `umap` on the Reference and
`pca50 -> umap` as separate Candidates, and letting the Battery settle whether the
pre-step helped.

**Hyperparameters.** `drtools suggest-params --op <op> --run-dir runs/<id>` gives
profile-derived starting values with the reasoning behind them, and persists them in the
Run. Pass `--params` with the Stage's other settings when the Suggestion depends on them
— LLE's neighbour minimum grows with `method` and `n_components`. Library defaults are
usually wrong for the data at hand — a perplexity of 30 on 500 samples is not a
judgement, it is an oversight.

Registration compares every value against the persisted Suggestion and records it as
`suggested`, `overridden`, `specified` (nothing was suggested for it) or
`registry_default` (not given, and no different Suggestion). A parameter left unset
while its Suggestion differs from the default is an Override too. An Override is
refused unless the Stage carries its reason:

```json
{"op": "tsne", "params": {"perplexity": 50},
 "overrides": {"perplexity": {"reason": "...", "evidence": ["recon.neighbourhood.k"]}}}
```

**Rejections.** Every Reduction and Visualization method in `drtools methods` appears in
a Candidate, in `rejected`, or both; one that appears in neither is refused. Each
Rejection carries a reason and the Evidence keys behind it, and an uncited one is
refused. `"MDS rejected — O(n^2) at n=107,000; PCA already captures the global variance
structure it would recover"` is evidence of judgement. A method may be both rejected and
run when the Rejection rules out one configuration — `umap` on the raw features — and
every Candidate running it puts another method in front, as `pca50 -> umap` does. `pca`
is never rejected, since the Linear baseline runs it alone.

**The weighting.** Declare a weight per metric in the Battery, summing to 1, with a
justification tied to what the user asked for and what the Profile says. You are
choosing emphasis before you can see who wins — that is the point, and after
registration the whole `evaluation` block is frozen: weights, justification and
evidence.

| Data | Default | Needs `evaluation.evidence` |
|---|---|---|
| No labels | trustworthiness 0.25, continuity 0.25, shepard_correlation 0.5 | only for a departure |
| Labels | the unlabelled default, or 0.175, 0.175, 0.35, knn_label_preservation 0.20, silhouette 0.10 | always |

With labels, choosing a default decides whether the labels are trusted: the labelled
default says they were supplied with the data and not derived from it, the unlabelled
one that they carry no weight — derived labels, or an analysis meant to find new
groups. Cite what settles it. Any other weighting is a departure, allowed when its
Evidence keys argue for it. The label metrics need labels, and `runtime_s` carries no
weight: it is reported, never scored.

## Register it

Write `plan.json` into the Run, then:

```
drtools validate-plan --run-dir runs/<id>
```

This is the registration event. It needs the data decision `recon` recorded, and
simulates the Plan against the Profile — tracking sample count, feature count, sparsity,
memory, and whether values are still raw counts — so it knows what each method will
actually receive. `"Isomap on 107,000 points"` is refused
where `"subsample to 3,000, then Isomap"` is not.

Every Evidence key the Plan cites is resolved here, against the Run's `profile`,
`recon`, `suggestions`, `metrics` and `ranking`, and one that does not resolve is
refused. `plan.*` resolves because the Plan says so, so a Plan cites the artefacts that
measured the data, never itself.

A report with `valid: false` lists every finding at once, each with a `fix`. Revise and
resubmit. **Every refusal is recorded**, and that record is worth having: the report
can honestly say the planner proposed X, the validator caught it, and the plan became
Y. Do not work around a finding — repair the Plan it describes.

Then hand off to **execute-plan**.

## What does not reduce to a table

- t-SNE and UMAP cluster sizes, and the distances between clusters, are not meaningful.
  Do not interpret them, and say so in the report.
- A method failing is information about this configuration, not about the method.
- Spanning families is what makes disagreement informative: when UMAP and PCA disagree,
  that disagreement is itself a finding about the data's nonlinearity.
