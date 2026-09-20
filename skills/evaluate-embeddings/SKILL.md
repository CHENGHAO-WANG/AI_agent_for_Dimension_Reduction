---
name: evaluate-embeddings
description: Use when a dr-agent Run has Candidates that produced Embeddings and `drtools status` reports `evaluate` as the next stage, or when a ranking needs recomputing after the portfolio grew.
---

# Evaluate the embeddings

Score every Candidate on the same Battery, rank them under the weighting that was
registered before any of this existed, and draw the figures. The scoring is
deterministic; your work is reading what comes back.

## Read your position from the Run

```
drtools status --run-dir runs/<id>
```

Evaluate every Candidate whose `outcome` is `ok` and whose `metrics` is false. `ranked`
is false whenever the existing ranking does not belong to the currently registered
Plan — a grown portfolio invalidates the previous ranking, and this is how you know.

## Steps

1. **`drtools evaluate --run-dir runs/<id> --id <candidate>`** for each successful
   Candidate. The neighbourhood size, the sample cap and the seed all come from the
   Run; there is nothing to choose, which is what makes the Candidates comparable.

2. **`drtools rank --run-dir runs/<id>`** — combines the Battery under the registered
   weighting.

3. **`drtools figures --run-dir runs/<id>`** — draws the standard set. Open them and
   look. A collapsed Embedding, a mislabelled panel or an empty view is visible in the
   image and invisible in the numbers.

## Reading what comes back

**Reference values are a baseline, not a ceiling.** A metric computed on the Reference
says what the representation Candidates were reduced *from* achieves. An Embedding can
beat it: in high dimensions distances concentrate, so the Reference's own neighbourhoods
are noisy, and discarding noisy directions can recover structure the full-dimensional
space obscured. Report that as what it is. Do not divide one by the other and call the
result a percentage retained.

**Read the `notes` on the ranking.** They carry the caveats that change what the
ranking means:

- A metric that could not be computed for every Candidate is dropped for all of them,
  and the weight is redistributed. The ranking then answers a narrower question than
  the weighting intended, and the note says how much weight moved.
- Two Candidates separated by a hair are tied, not ranked. Stochastic methods and
  repeated subsampling move scores by more than that margin.
- Candidates that produced no Embedding are excluded rather than scored as zero.
  Exclusion is a judgement on that configuration, not on the method.

**Runtime is scaled within the cohort**, because two seconds is fast or slow only
relative to the alternatives. Every other metric is on an absolute scale, so a field of
uniformly poor Embeddings does not produce a winner that looks excellent.

## The weighting does not move

If the results make you wish the weighting were different, that is the mechanism
working. Pre-registration exists so that emphasis cannot be chosen to flatter whichever
Candidate happened to win. `rank` refuses a weighting that has changed since
registration, and the sanctioned route — an Amendment — is defined but not implemented,
so a changed weighting means a new Run. Say in the report what you would have weighted
differently, and why; that is an honest finding.

## One re-plan, if the evidence asks for it

A single round of extending the portfolio is permitted, and `status` reports whether it
is spent. Extend it when the results reveal something the Profile did not — a structure
no Candidate was shaped to capture. Do not extend it to try harder at winning.

Then hand off to **write-report**.
