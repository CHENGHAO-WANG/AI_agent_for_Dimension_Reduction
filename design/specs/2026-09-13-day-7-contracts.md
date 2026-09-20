# Day 7 — the three contracts

Status: design, approved in outline, not yet implemented.
Supersedes nothing. Feeds the day 7 implementation plan.

## Why

Two reviews on day 6 found that three guarantees the design claims are not
enforced by the code. Each was reproduced:

1. **Dataset identity.** Re-running `profile` in an existing Run after repairing a
   Loader wrote a Profile of the new matrix while the cache kept the old one —
   `n_features` 4 against 8. Planning reads the Profile, Candidates read the cache.
2. **Pre-registration.** Editing the weighting after metrics existed and re-ranking
   flipped the winner with no Amendment and no warning, and the Decision log recorded
   *"weights were declared in the plan before any embedding was computed"*.
3. **Comparability.** Candidates scored at k=5 and k=30 ranked together with no
   caveat anywhere in `ranking.json`.

All three are silent. None would surface on day 8, and day 9 fixes only what day 8
breaks.

## Decisions

- **Refuse, never warn.** A violation is a hard error that writes nothing. No
  override flag.
- **Amendment stays a term, unimplemented.** `CONTEXT.md` defines it as the only
  sanctioned way for a weighting to move. Day 7 does not build it; `rank` refuses
  and names it as the route that does not yet exist, and the report says the same.
- **Identity is content.** A digest over the matrix and label codes — not the
  Loader's source, which must stay free to be tidied without invalidating a Run.
- **The Battery's settings are the toolbox's, by published rule.** The agent weights
  the metrics; it does not choose the neighbourhood size any more than it chooses
  which metrics to look at. `--k` leaves `evaluate`.
- **`validate-plan` is the registration event.** It is the only command that already
  reads the Plan and writes a verdict, and it is where the agent has said "this is
  the plan". No new command to forget.
- **Enforcement lives with the artefact.** `cache.py` owns identity, `plan.py` the
  freeze, `rank.py` comparability, with one shared `content_hash()` and refusal
  through the existing `ContractError`.

## 1. Dataset identity

`write_cache` computes a digest and stores it in `meta.json` as `dataset_digest`.

Sparse input **must be canonicalised first** — `.tocsr()`, `.sum_duplicates()`,
`.sort_indices()`, `.eliminate_zeros()` — and `indices`/`indptr` must then be hashed at
a **fixed width and fixed endianness** (`astype('<i8')`), independent of how they were
stored. The digest folds in shape and the *value* dtype; it must not fold in the index
dtype. Measured in this venv, the same matrix built with unsorted indices, with an
explicitly stored zero, or with int64 rather than int32 indices gives three different
digests, and `sp.csr_matrix()` does not sort — and index width is not a property of the
data, so a Loader that switches to int64 must not invalidate a Run. With the fixed-width
cast all three constructions agree while genuinely different data still differs.

Canonicalisation happens inside `content_hash()`, so the write path and the check path
cannot drift apart.

The digest covers the matrix and the label codes only. `name`, `source` and
`label_names` stay out — they must be free to change — but are recorded in cache meta
and re-emitted into the manifest when `--data` is absent.

`ensure_cache` stops early-returning. It digests the incoming data and raises on
mismatch, naming both digests and the recovery: a changed dataset needs a new Run.

**The larger half is that `profile`, `recon` and `embed` stop loading from source when
the Run has a cache.** They read the cache. `--data` becomes optional when `--run-dir`
names a Run that has one; passing it anyway is a verification that refuses on
mismatch. This closes the Profile/cache split and the `embed` source/cache divergence
together.

Digesting happens on write and inside `ensure_cache` only — never in `read_cache`,
which runs in every Candidate subprocess, in `figures`, and twice in `_reference_for`.
Measured: 36 ms for a pbmc3k-shaped CSR, 2.0 s for a pathmnist-shaped dense matrix
streamed from a memmap. Once is free; per-read would cost 20+ seconds a Run and defeat
the memmap that exists so Candidates do not each pay for the matrix. Stream in row
blocks rather than calling `.tobytes()` on a memmapped gigabyte.

A refusal must write nothing. Today `_cmd_profile` calls `write_manifest` before
`ensure_cache`, so a rejected re-profile has already overwritten the manifest's
command, spec and timestamp. Digest-check first; write after.

One `_resolve_run(args)` requires either an existing Run with a cache or an explicit
`--data`, and refuses a `--run-dir` that does not exist for every command but
`profile`. Today a mistyped path silently creates an empty Run, and
`evaluate --id x` without `--run-dir` raises `AttributeError` rather than a usage
error.

## 2. Plan registration

`validate-plan` writes `plan.registered.json` and a `plan_digest`, and appends a
registration record to the Decision log carrying the digest, the **full weighting**
and the Candidate ids — so the report's pre-registration claim is checkable from the
log alone.

**The freeze is per artefact, not wholesale.** A wholesale freeze would forbid the
loop `design/notes.md` §4 calls *"the most convincing single piece of evidence of
agency the system can produce"* — one diagnose-and-retry per method, the worked
example being Laplacian Eigenmaps failing on a disconnected graph and succeeding at
`n_neighbors=30`. The Plan must be able to carry that retry.

Permanently immutable once registered:

- `evaluation.weights` and `evaluation.justification` — always, no exceptions
- `base_preprocessing`
- the stages of any Candidate that has an Embedding with Outcome `ok`

Changeable by re-registering, which bumps a sequence number and appends a record:

- adding a Candidate id
- changing the stages of a Candidate with no Embedding, or whose Outcome is `failed`,
  `timeout` or `crashed`

Removing a Candidate refuses: the Rejections are evidence, and a Candidate that ran
and lost is part of the record.

**Invalidation happens before every embed attempt, not at re-registration.** Tying
cleanup to a stage change misses the commonest retry of all — re-running a Candidate
whose stages are fine but whose Budget was too small. Reproduced: a Candidate that had
succeeded and been evaluated, re-embedded with a 1s timeout, ends with Outcome
`timeout` while `metrics/<id>.json` survives; `rank` then lists it among the ranked
Candidates on its previous scores and does not even report it as failed.

So:

- Re-embedding a Candidate whose Outcome is `ok` **refuses**. Its stages are immutable
  and its result already exists; there is nothing to gain and a replacement would make
  the Embedding disagree with the record.
- Re-embedding a Candidate whose Outcome is `failed`, `timeout` or `crashed` is the
  retry loop, and the **toolbox** deletes every dependent artefact first —
  `embeddings/<id>.*` and `metrics/<id>.json` — rather than leaving it to the agent.
- `rank` additionally excludes any Candidate whose current Outcome is not `ok`,
  whatever `metrics/` holds. Belt and braces, because the ordering above is the kind of
  invariant that a later refactor quietly breaks.

Together these close the queued day-9 defect where a reused Candidate id dropped only
its Outcome record.

**The freeze anchors in the Decision log, not in the filesystem.** `embed` appends an
outcome-bearing record per attempt, and re-registration refuses only where that
Candidate has a recorded **successful** attempt — not on the presence of any `embed`
record, which would freeze exactly the failed Candidates the retry loop exists to
revise. Anchoring in files is defeated by deleting them; anchoring in an append-only
log raises tampering from editing one JSON to forging a consistent log. It also fixes
an ambiguity: a timeout writes `embeddings/<id>.json` but produces no Embedding, so a
file-based test would freeze the Plan on a Candidate that never ran.

An attempt killed before it could record an Outcome — the run died, the machine
rebooted — leaves a started record with no completion. Treat it as `crashed`: the
Candidate is revisable and re-embeddable, which is the same state the isolation layer
already calls "died without recording anything".

`embed` requires a registered Plan, `--id` must name a registered Candidate, and the
stages come from `plan.stages_for(candidate)` rather than `--stages`. This is a
`cli.py`-only change: `run_candidate` and `_worker.py` take a stage list and neither
knows where it came from.

`rank` reads `plan.registered.json`, refuses when `plan.json` has diverged, and
stamps the digest into `ranking.json`.

**Delete the fallback rationale at `cli.py:462-463`** — the string that asserts
pre-registration whenever no justification was given. It is one line, it is false, and
the report quotes it.

## 3. The Battery's settings

`DEFAULT_K` is replaced by a published rule:

```
k = max(1, min(15, ceil(n / 2) - 1))
```

Checked against scikit-learn 1.9.1 across n = 2..1000 for `trustworthiness`, its
transposed form, `NearestNeighbors(k+1)` and `silhouette_score`. It holds for every
n >= 3 and the boundary is tight: n=30 gives k=14 against a limit of 15.0, n=31 gives
k=15 against 15.5. n=2 is the only failure, and metrics are unavailable there.

**k is derived once from the Reference's row count and fixed there**, not computed per
Candidate. Per-Candidate derivation makes two Candidates that subsample differently
get different k, which `rank` then refuses with no legal next command — the agent
cannot change a toolbox-derived setting. One k makes mixed settings impossible by
construction, and gives the report a number to quote.

It is fixed when the Reference is, not at registration: `validate-plan` precedes
`prepare-reference`, so the Reference's row count does not exist yet at registration
time. `prepare-reference` computes `k = rule(reference rows)` and records it, with the
sample cap and the seed, in `reference.json`; the Reference is immutable once created,
so its settings are too. When `base_preprocessing` is empty the Reference is the cache
and k derives from the cache's row count. `evaluate` reads those settings and refuses
a Candidate whose row count cannot support the recorded k, naming the subsample.

**The Run's seed is fixed when the Run is created and never moves.** Reading it from
the manifest is not the same as freezing it: `_cmd_profile` rewrites the manifest with
`args.seed`, and re-profiling *identical* data passes the digest check, so an agent can
evaluate one Candidate, re-profile with a different seed, and evaluate the next on a
different metric subsample without the registered Plan changing. Reproduced: re-profile
moved the seed 0 → 7, and the 2000-row metric subsample of a 3000-row dataset kept only
1347 rows in common. `profile` therefore preserves the recorded seed and refuses a
`--seed` that differs from it, and every evaluation setting derives from that immutable
state — including on the empty-base path, where the Reference is the cache and no
`reference.json` records them.

`evaluate` loses `--k` and `--max-samples`; the seed comes from that immutable Run
state. The
`k=` parameter leaves `evaluate_embedding`'s signature rather than becoming an
override — an override leaves the incomparability open for the next caller. Five call
sites in `tests/test_metrics_rank_plan.py` change; all four scenarios were re-checked
at k=15 and every assertion still holds.

Two guards are independent of k and must be separate:

- `silhouette_score` requires `1 < n_classes < n_used`. Verified: n=3 with 3 classes
  and n=4 with 4 classes both raise regardless of k, so the n<3 carve-out does not
  cover them.
- kNN label agreement requires `k + 1 <= n_used`.

Both return `None` with a note rather than raising. `rank` already drops a metric
absent for every Candidate.

Below `n_used < 20` the local metrics return `None`. The rule yields k=1 at n=3 and
k=2 at n=5; trustworthiness over one neighbour is noise, and `rank` would weight it at
face value. Absent is safer than a number.

**Deferred to day 9:** `reference_digest`, per-record digests and the mixed-digest
check. Two cheap invariants close the real silence without that machinery:

- `prepare-reference` drops `--stages` and reads `base_preprocessing` from the
  registered Plan. Two sources for one invariant is how a Reference comes to be
  something other than the Base preprocessing.
- `evaluate` refuses when the registered Plan declares a non-empty
  `base_preprocessing` and no `reference.json` exists. Today `_reference_for` falls
  back to the cache, so Candidates that ran `standardise → pca` are scored against raw
  counts while the record affirms `"loaded dataset as cached"`.

The Reference freeze narrows to refusing a *change* to an existing Reference.
Creating one where none exists is always legal — it is fully determined by the cache
digest and the registered base stages — which removes an ordering edge and a wedge
together.

When `base_preprocessing` is empty, `prepare-reference` writes `reference.json` — the
settings, and a marker that the Reference is the cache — but no `reference.npy`. That
answers the identity question and sidesteps the queued defect where
`np.asarray(sparse)` writes an unloadable 0-d object array.

An agent may reasonably skip `prepare-reference` when the Plan declares no base, so
`evaluate` must not require `reference.json` in that case. The two branches:

- `base_preprocessing` non-empty and no `reference.json` → refuse, naming
  `prepare-reference`
- `base_preprocessing` empty and no `reference.json` → the Reference is the cache and
  the settings derive from its row count by the same rule, deterministically. No
  refusal, no wedge.

## 4. Tests

There is no CLI test harness. Nothing under `tests/` imports `drtools.cli`, which is
why 178 passing tests coexisted with nine defects — and all three failures here live
in `cli.py`. Building that harness is part of this work, not a follow-on.

One regression test per contract, each built from its reproduction:

- re-profile a Run with changed data; assert refusal and that the manifest is untouched
- register, embed, rewrite the weighting, rank; assert refusal and that the Decision
  log carries no pre-registration claim
- register, embed, retry a failed Candidate with new stages; assert the retry is
  allowed, the stale metrics are gone, and the weights could not be touched

And four more, each from a hole the design review found rather than from a reproduction
of the original three:

- re-embed an evaluated Candidate with a Budget too small to finish; assert the stale
  metrics are gone and that `rank` does not score it
- re-embed a Candidate whose Outcome is `ok`; assert refusal
- evaluate a Candidate, re-profile with a different `--seed`, evaluate another; assert
  the seed did not move
- re-register after a `failed`, a `timeout` and a `crashed` Outcome; assert each is
  revisable, and that a Candidate with a successful attempt is not

Plus unit tests for `content_hash` canonicality — the three constructions that
currently differ, including int32 against int64 indices, which must now agree — and for
the k rule and the two guards at n = 2, 3, 4, 5, 19, 20, 30, 31.

## Ordering

```
profile (--data) → [recon] → write plan.json → validate-plan (registers)
                 → prepare-reference → embed --id ×N → evaluate --id ×N → rank → figures
```

Acyclic. Every refusal message must name its recovery route, or the skills will encode
a loop around it: `cli.py` prints the message and nothing more.

## Cost

Day 7 was the five skills, `/analyze` and `decisions.jsonl`. It cannot also be three
contracts and a test harness. Sections 1, 2 and 4 plus the k-rule half of 3 land on
day 7; the Reference-digest machinery moves to day 9, beside the two defects it
collides with. GPLVM absorbs the slip, which is what the cut order is for.

The three regression tests do not move. The day-6 conclusion was that the tests passed
because nothing exercised the seams; these are the seams.

## Naming

Two concepts have no term in `CONTEXT.md`. `dataset_digest` and the settings recorded
with the Reference are provisional keys, to be named by `domain-modeling` as this
closes:

- the identity of the ingested dataset within a Run
- the settings the Battery is computed under — note the glossary lists "protocol" as a
  word to avoid, so the review's "evaluation protocol" is not available
