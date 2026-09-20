---
name: write-report
description: Use when a dr-agent Run has been ranked and `drtools status` reports `report` as the next stage, including a Run where every Candidate failed and the outcome still has to be written up.
---

# Write the report

Generate the report **from the Decision log**, not from memory of the session. Every
claim traces to a record the Run actually wrote, and every number to an artefact that
actually holds it.

## Sources, in this order

```
runs/<id>/decisions.jsonl   every choice, its reasoning, its Evidence keys
runs/<id>/ranking.json      scores, weights applied, and the notes that qualify them
runs/<id>/profile.json      what the data is
runs/<id>/recon.json        what its structure looked like before planning
runs/<id>/metrics/          per-Candidate scores
runs/<id>/figures/          the drawn figures
```

Read the Decision log first and let it drive the narrative. Every record carries what
was asked, what was chosen, why, and what its Evidence keys resolved to at the time.
A section you cannot source from a record is a section you should not write.

## Fixed sections, in this order

Both generated reports use the same skeleton so they can be compared:

1. Dataset profile
2. Preprocessing decisions, and why
3. Methods selected **and rejected**, with reasons
4. Hyperparameter choices, and why
5. Figures
6. Quantitative comparison
7. Ranking, with the weighting justification
8. Interpretation
9. Limitations

**Section 3 is the highest-value part of the document.** The Rejections are the most
direct evidence that the agent selected rather than sprayed, and each one already
carries its reason and its Evidence keys in the log. Give them the same weight as the
methods that ran.

Markdown in the Run directory is the source of truth; rendering to PDF is a separate
step.

## What you may claim

Citation integrity is what the Evidence mechanism buys: no rationale rests on a number
the Run never computed, and a reader can follow every claim back to the artefact behind
it. **It is not proof that a rationale is true.** A real key with a false reading passes
— a silhouette of 0.7 is 0.7, and "confirms distinct biological cell types" is not
thereby supported. Claim the narrower thing.

Three specific restraints:

- **t-SNE and UMAP** cluster sizes, and the distances between clusters, are not
  meaningful. Say so where such a figure appears.
- **A Reference value is a baseline, not a ceiling.** An Embedding that exceeds it has
  not retained more than 100% of anything.
- **A failed Candidate** says this configuration did not work. It does not say the
  method is unsuitable, unless the failure itself is the evidence — a disconnected
  neighbourhood graph is a fact about the data.

## What belongs in Limitations

The things a reader would otherwise have to discover:

- Metrics dropped from the ranking, and how much weight moved with them.
- Candidates separated by less than the noise, reported as tied.
- Any subsampling, and what the metrics therefore describe.
- Figures where identity is carried by printed class names rather than colour, and why:
  above three classes, no categorical palette clears the separation floors for
  scatter-like forms under simulated colour-vision deficiency.
- A weighting you would now choose differently, if the results suggested one.

## A Run where everything failed

Still a report. What was tried, what each failure was, what the failures say about the
data, and what you would try next. A clean account of an analysis that did not work is
worth more than a ranking of nothing.
