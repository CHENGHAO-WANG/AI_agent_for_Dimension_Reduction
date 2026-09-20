---
name: execute-plan
description: Use when a dr-agent Run has a registered Plan and `drtools status` reports `execute` as the next stage, including when a Candidate has failed, timed out or crashed and needs diagnosing.
---

# Execute the plan

Run each registered Candidate in a process that can be killed, and turn every failure
into a record you can act on. A Candidate that fails is not a problem with the Run —
diagnosing one and repairing it is the clearest evidence of judgement this system can
produce.

## Read your position from the Run

```
drtools status --run-dir runs/<id>
```

`candidates` gives each one's `outcome`, how many `attempts` it has had, whether a
`retry_available`, and whether it was `abandoned`. Trust this over memory: it is
derived from the append-only Decision log, not from what you recall running.

## Steps

1. **`drtools prepare-reference --run-dir runs/<id>`** — runs the registered Base
   preprocessing once. Every Candidate is scored against its output, so this happens
   before any scoring, and the neighbourhood size the Battery uses is fixed from it.

2. **`drtools embed --run-dir runs/<id> --id <candidate>`** for each Candidate. Stages
   come from the registered Plan; you name the Candidate, not the pipeline.

The wall-clock cap comes from the Run's registered Budget. `--timeout` may lower it and
will be refused if it raises it, because every Candidate was planned and rejected under
that Budget.

## When a Candidate does not produce an Embedding

Four Outcomes, and they are not the same thing:

| Outcome | What happened |
|---|---|
| `ok` | An Embedding exists |
| `failed` | It raised; the record names the exception type, the Stage and the parameters |
| `timeout` | It did not finish inside the Budget |
| `crashed` | The process died without recording anything — usually memory |

Read the record. `MemoryError` and `LinAlgError` call for different repairs, and the
record names the Stage and the parameters it ran with, which is what you revise from.

**You get one diagnose-and-retry per Candidate.** Two attempts is the whole allowance.
To retry with different parameters, re-register the Plan with that Candidate's Stages
revised — the freeze permits revising a Candidate that failed, timed out or crashed,
and permits adding one. It refuses to move the weighting, the Base preprocessing, or
the Stages of a Candidate that already succeeded.

The worked case: Laplacian Eigenmaps fails on a disconnected neighbourhood graph, you
raise `n_neighbors` well above the `k` Reconnaissance probed at, and it succeeds. Log
both halves — the diagnosis and the repair.

**When the allowance is spent**, record a permanent failure and move on to the
remaining Candidates. You may instead register a differently-shaped Candidate under a
new id and abandon the old one, but that spends from the Run's ceiling on how many
Candidates it may ever register, and the ceiling is what makes this terminate. When
`embed` refuses, it says how much room is left. A Run that reaches its ceiling has
spent its scope of work: score what succeeded and report it.

Mark giving up explicitly, or `status` will keep offering a retry nobody intends to
take:

```
drtools log-decision --run-dir runs/<id> --json @decision.json
```

with `"candidate": "<id>"` and `"abandoned": true`, alongside the reason.

## What not to reach for

`--in-process` is refused. It runs with no wall-clock cap, so no Budget can bind it and
a method that exhausts memory takes the toolbox down with it, leaving no record of the
attempt. The isolated route is the one that produces Outcomes.

Then hand off to **evaluate-embeddings**.
