---
name: profile-dataset
description: Use when beginning a dr-agent analysis of a dataset, or when `drtools status` reports `profile`, `recon` or `checkpoint` as the next stage of a Run.
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

3. **Decide the two facts the preprocessing rules read.** Are the values raw counts?
   Are the features all of one type — genes, intensities at a series of wavelengths —
   or mixed, like age, sex, height and weight in one table? The matrix alone cannot
   settle the second. Read `profile.features.column_kinds` (binary, integer and
   continuous columns side by side mean mixed types), `source_format`, `names_head`,
   and `profile.values.suspected_kind`. The spread of standard deviations is weak
   evidence either way.
   - The user declared either fact when starting (`/analyze --values`, `--features`):
     that answer stands, `decided_by: user`.
   - The evidence settles it: your call, `decided_by: agent`, citing the keys.
   - The evidence is unclear and nothing was declared: ask the user now, before
     reconnaissance, since the probes depend on the answer. The question counts
     toward the checkpoint's three. Under `--auto`, take the default the refusal below
     states — raw counts when the profile suspects them, features mixed — as
     `decided_by: default`.

4. **`drtools recon --run-dir runs/<id> --decision @runs/<id>/inputs/data_decision.json`**
   ```json
   {"values": "raw_counts", "features": "one_type", "decided_by": "agent",
    "rationale": "...", "evidence": ["profile.features.column_kinds"]}
   ```
   The cheap structural probes, run on the base preprocessing the decision gives:
   PCA spectrum and its elbow, an intrinsic dimension estimate, neighbourhood-graph
   connectivity, and a thumbnail image. Open the thumbnail and look at it — it is
   evidence, not decoration. Without `--decision` it refuses and states the default.
   The decision changes only by running `recon` again, which a changed answer at the
   checkpoint requires.

5. **Read `probe_representation` before reading anything else in `recon`.** Every
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

Once profiling and reconnaissance are done, ask the user at most three multiple-choice
questions in all, the feature-type question of step 3 included. State a default for each
and proceed on it if there is no answer.

- **The Purpose.** Is the deliverable a representation downstream analysis will use —
  clustering, regression, testing — or a picture? Representation is the default. An
  `/analyze --purpose` argument is the user's answer already given.
- **The focus**, asked in the Purpose's terms:

  | Focus | Representation: used downstream for | Visualization: the picture should show |
  |---|---|---|
  | `local` | clustering, other neighbourhood-based analysis | clusters and neighbourhoods |
  | `global` | distances, regression, a map of the whole | the overall layout |
  | `balanced` | neither favoured | neither favoured |

  Unanswered, set it from the evidence and cite the keys, or take `balanced` when the
  evidence favours neither.

Record the answers, whoever gave them:

```
drtools checkpoint --run-dir runs/<id> --answers @runs/<id>/inputs/checkpoint.json
```
```json
{"purpose": "representation", "purpose_decided_by": "user",
 "focus": "local", "focus_decided_by": "agent",
 "rationale": "...", "evidence": ["recon.neighbourhood.n_connected_components"]}
```

An answer that is yours, `decided_by: agent`, cites the evidence it rests on. Under
`--auto`, ask nothing and record the defaults as `decided_by: default`. Registration
refuses a Plan until the checkpoint is recorded, and freezes it once registered.

## Log what you decided

Anything else you chose rather than read — the dataset spec, an adapter, a label
column — goes in the Decision log:

```
drtools log-decision --run-dir runs/<id> --json @runs/<id>/inputs/log_entry.json
```

A decision needs all four of these, and is refused without `question` or `chosen`:

```json
{"stage": "profile", "question": "Which column holds the labels?",
 "chosen": "cell_type", "rationale": "...", "evidence": ["profile.labels.present"]}
```

Cite Evidence keys from `profile` and `recon`; the command refuses a key that does not
resolve, and records what each one resolved to. Use your own stage names. The
lifecycle stages the toolbox writes for itself are reserved, and it will say so.

Then hand off to **plan-analysis**.
