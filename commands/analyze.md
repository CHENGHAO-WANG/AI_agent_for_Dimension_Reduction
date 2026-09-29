---
description: Analyse a dataset by dimension reduction, end to end, and write up what was decided and why.
argument-hint: <dataset spec> [--auto] [--purpose representation|visualization] [--budget fast|standard|thorough] [--values raw_counts|not_counts] [--features one_type|mixed] [--run-id <id>]
---

Analyse `$ARGUMENTS` by dimension reduction, from profiling through to a written report.

## How this runs

Five skills, in order, each reading its position from the Run rather than from this
conversation:

```
profile-dataset -> plan-analysis -> execute-plan -> evaluate-embeddings -> write-report
```

Invoke each skill when you reach its stage, and follow it. They are the instructions;
this command only orders them.

## Check the toolbox is there

Installing this plugin delivers these instructions. It does not deliver `drtools`,
which is a Python package. Before anything else:

```
drtools datasets
```

If that is not found, stop and say so: the skills are installed but the toolbox is not.
It installs with `pip install dr-agent`, or from a clone of the repository with
`pip install -r requirements.txt`; TriMap then takes `pip install --no-deps
trimap==1.2.0`, and without it only TriMap is unavailable. Do not improvise around a missing `drtools` — every
number in the report has to come from it.

## Start or resume

If a `--run-id` was given and that Run exists, or if `runs/` holds an unfinished Run for
this dataset, this is a resumption. Otherwise it is a new Run. The two open
differently, because `status` reads a Run and cannot create one.

**A new Run.** Enter **profile-dataset** and start at its first step.
`drtools profile` is what creates the Run directory, and until it has run there is
no position to read: every other command refuses a path that does not exist.

**A resumed Run.** Read the position first:

```
drtools status --run-dir runs/<id>
```

`next` names the stage to enter: `profile`, `recon` or `checkpoint` (all
**profile-dataset**), `plan`, `execute`, `evaluate` or `report`. Enter that stage's
skill.

From there the two are the same. Re-read `status` after each stage rather than
assuming the next one — a Candidate that failed changes what comes next.

Nothing about the Run lives in this conversation. If context is lost, `status` restores
the position completely.

## Arguments

- **dataset spec** — a built-in name, or a path. `drtools datasets` lists the built-in
  ones. For a format nothing recognises, write a Loader and pass `--adapter`.
- **`--auto`** — ask nothing at either checkpoint: take the stated defaults at the one
  after profiling and record them, and adopt no picture at the one after a
  Visualization run's results. The graded Runs use this, so the reports are produced
  with no intervention.
- **`--purpose representation|visualization`** — whether the deliverable is a
  representation for downstream analysis or a picture. Given here, it is the user's
  answer to the checkpoint's Purpose question. Default representation.
- **`--budget fast|standard|thorough`** — the compute this Run may spend. It caps each
  Candidate's wall-clock and how many Candidates the Run may ever register, and it is
  frozen when the Plan is registered. Default `standard`.
- **`--values raw_counts|not_counts`**, **`--features one_type|mixed`** — declare either
  fact the preprocessing rules read, when you know it. A declaration stands over the
  agent's reading of the profile, and is recorded as the user's.
- **`--run-id`** — name the Run directory. Otherwise it is named for the dataset and
  the time.

## What you may not do

Method selection, execution, evaluation and ranking go through `drtools` and nowhere
else. Do not compute an embedding, a metric or a ranking yourself, and do not edit the
artefacts under `runs/<id>/` by hand: every number in the report has to come from code
that can be read. Writing a Loader for an unrecognised format is the one exception, and
it is contract-checked like any other.

When a command refuses, the message names the problem and the route out. Repair what it
describes rather than working around it — the refusals are recorded, and a plan that was
caught and revised is worth more in the report than one that happened to be right.

## Finish

The Run directory is the deliverable: artefacts, figures, the Decision log, and the
report. Tell the user the report and the export are in `runs/<id>/results/`, and what
came out: in a Representation run what won, with the caveats the ranking's notes carry;
in a Visualization run what you recommended, labelled as your judgment, and what the
user adopted, if anything.
