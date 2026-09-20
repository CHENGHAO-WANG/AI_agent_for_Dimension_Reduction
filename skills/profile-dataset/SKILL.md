---
name: profile-dataset
description: Use when beginning a dr-agent analysis of a dataset, or when `drtools status` reports `profile` or `recon` as the next stage of a Run.
---

# Profile the dataset

First stage of the `/analyze` chain. Measure what the data is, then probe how it is
structured, so that planning argues from evidence rather than from the dataset's name.

## Read your position from the Run

A new Run has no position to read. Step 1 below is what creates the Run directory,
and until it has run every command that reads a Run refuses the path. Starting a new
analysis, go straight to step 1.

Resuming a Run that exists, read the position first:

```
drtools status --run-dir runs/<id>
```

`next` names the stage with work outstanding. Act on that rather than on what you
remember doing — the Run is the only state, and a resumed session remembers nothing.

## Steps

1. **`drtools profile --data <spec> --runs-root runs --run-id <id>`**
   Creates the Run and measures the dataset. `drtools datasets` lists what loads
   without a Loader of your own.

2. **Read `observations`.** Each one states what a measurement implies for the
   analysis and carries the Evidence keys behind it. `"87% of entries are zero and
   sample totals vary 24-fold, so raw Euclidean distance mostly measures sequencing
   depth"` is something you can plan against; `sparsity: 0.87` is not. These
   observations, and the keys they cite, are the raw material for every later
   rationale.

3. **`drtools recon --run-dir runs/<id>`**
   The cheap structural probes: PCA spectrum and its elbow, an intrinsic dimension
   estimate, neighbourhood-graph connectivity, and a thumbnail image. Open the
   thumbnail and look at it — it is evidence, not decoration.

4. **Read `probe_representation` before reading anything else in `recon`.** Every
   reconnaissance number is conditional on that transform. When you later cite an
   intrinsic dimension or a component count, the claim is about the probe, and the
   report must say so.

## When the Loader refuses

The refusal names what is wrong and what to do instead. Three of them are decisions,
not defects:

- **Missing values** are out of scope. Do not write a Loader that fills them in —
  fabricated numbers would enter every metric and figure as though measured.
- **Non-numeric columns** must be encoded, dropped, or named with `--label-column`.
- **An unrecognised format** is the one case where you may write a Loader of your own,
  passed with `--adapter`. It is contract-checked exactly like a built-in one, and the
  Run records its path and a digest of the source that ran.

## The checkpoint

Once profiling and reconnaissance are done, ask the user at most two multiple-choice
questions — what the analysis is for, and anything the profile leaves genuinely
ambiguous. State a default for each and proceed on it if there is no answer. Under
`--auto`, skip this entirely and record the defaults you took.

## Log what you decided

Anything you chose rather than read — the dataset spec, an adapter, a label column,
a checkpoint default — goes in the Decision log:

```
drtools log-decision --run-dir runs/<id> --json @decision.json
```

Cite Evidence keys from `profile` and `recon`; the command refuses a key that does not
resolve, and records what each one resolved to. Use your own stage names. The
lifecycle stages the toolbox writes for itself are reserved, and it will say so.

Then hand off to **plan-analysis**.
