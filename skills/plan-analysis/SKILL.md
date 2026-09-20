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

**Base preprocessing.** `drtools suggest-base --run-dir runs/<id>` returns Stages, a
rationale, and Evidence keys, derived from the same rule Reconnaissance used to choose
its Probe representation. Adopt it or override it — an override needs a logged reason.
Its output becomes the Reference every Candidate is scored against.

**Candidates.** Three to five, each a Stage list ending in a Reduction, deliberately
spanning families: one linear baseline, one local-structure, one global-structure. A
Candidate whose terminal Stage is plain `pca` is required — without a linear baseline
there is nothing to measure the nonlinear methods against.

Read `drtools methods` for what each Op preserves, assumes, destroys, and where it
stops scaling. You cannot introspect the library; the Capability records are what you
reason over.

The best experiment available here is entering both `umap` on the Reference and
`pca50 -> umap` as separate Candidates, and letting the Battery settle whether the
pre-step helped.

**Hyperparameters.** `drtools suggest-params --op <op> --run-dir runs/<id>` gives
profile-derived starting values with the reasoning behind them. Library defaults are
usually wrong for the data at hand — a perplexity of 30 on 500 samples is not a
judgement, it is an oversight. Override a suggestion with a logged reason.

**Rejections.** Every method you considered and did not run goes in `rejected`, with a
reason and the Evidence keys behind it. `"MDS rejected — O(n^2) at n=107,000; PCA
already captures the global variance structure it would recover"` is evidence of
judgement. An uncited rejection carries no weight, and the validator will say so.

**The weighting.** Declare a weight per metric in the Battery, summing to 1, with a
justification tied to what the user asked for and what the Profile says. You are
choosing emphasis before you can see who wins — that is the point, and after
registration it cannot move.

## Register it

Write `plan.json` into the Run, then:

```
drtools validate-plan --run-dir runs/<id>
```

This is the registration event. It simulates the Plan against the Profile — tracking
sample count, feature count, sparsity, and whether values are still raw counts — so it
knows what each method will actually receive. `"Isomap on 107,000 points"` is refused
where `"subsample to 3,000, then Isomap"` is not.

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
