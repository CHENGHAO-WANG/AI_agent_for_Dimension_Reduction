# Day 10 — the report contract

Status: design, approved in outline, not yet implemented.
Supersedes nothing. Feeds the day 10 implementation plan.

## Why

Day 10 is briefed as "report template + pandoc render; first end-to-end run, dataset 1".
The notes fix the requirement and not the mechanism, which is what makes this a design
pass rather than an implementation of one.

What is already settled. Section 5 fixes the nine-section skeleton and names pandoc as
the renderer. Section 2.3 fixes that the report is generated from the Decision log
rather than written freehand, and narrows what that buys to citation integrity: no
rationale rests on a number the Run never computed, and nothing more. Day 9's
`write-report` skill carries the skeleton, the source order and the claim discipline.

What is not settled, and is the whole of this document: whether `drtools` contributes
anything to the report at all. The CLI stops at `figures`. No command reads
`ranking.json` and emits a document, and no command calls pandoc. So the boundary
between what the toolbox produces and what the agent writes is open, and every day from
11 to the graded Runs on 15 reads whatever it is set to.

## The decision

**The toolbox emits the skeleton with every number already in it; the agent writes the
prose between the numbers.** Generated sections carry fenced *facts blocks* that the
toolbox owns and can regenerate. Everything outside a fence belongs to the agent, and
the toolbox never touches it.

The property this buys: **the agent never retypes a number.** A figure in the report
cannot disagree with the artefact it came from, because the agent never had the
opportunity to transcribe it. That is a stronger guarantee than citation integrity and
it composes with it — citation integrity says a cited key resolves, and this says the
value printed is the value it resolved to.

Three alternatives were rejected.

*Render only* — the agent writes the whole document by hand and `render` just calls
pandoc. The smallest change, and the most faithful to "the agent decides". Rejected
because it leaves the grounding resting entirely on the agent following prose, and this
project has already recorded that prose to an agent is not enforcement. Every number in
the report would be a transcription, and a transcription is exactly where a stale or
mistyped value enters a document that claims to be generated from the log.

*Fully generated* — the toolbox assembles the entire document, and the agent's prose
reaches the page only through the `rationale` strings it logged at each choice point.
Maximal auditability. Rejected because sections 8 and 9 cannot be written that way.
Interpretation is not in the Run, by construction, and squeezing a paragraph of it
through a field meant for a one-line rationale would corrupt the Decision log to serve
the report — inverting which of the two is the record.

*Agent writes, toolbox verifies* — the agent writes everything and `report --check`
refuses any number that does not appear in an artefact. Grounding as a gate rather than
as construction. Rejected on cost and fragility: it needs a number-extraction parser
over free prose, and a figure quoted to two decimals, a percentage, a rounded count and
a number inside a sentence all have to be recognised. A parser that misses one is worse
than no parser, because it reports a clean check over a document it did not fully read.
Emitting the numbers achieves the same end with no parser at all.

## 1. Two commands

```
drtools report  --run-dir <run>              emits report.md
drtools report  --refresh --run-dir <run>    rewrites the facts blocks in place
drtools render  --run-dir <run>              report.md -> report.pdf via pandoc
```

Assembly and rendering are separate commands because they fail for unrelated reasons.
Pandoc being absent says nothing about whether the Run supports the report; a single
command that did both would give an environment problem and a Run problem the same
shape, and the agent's next move differs completely between them.

`render` takes the file as it finds it. It does not refresh first: a render that
silently changed the document's numbers would let the PDF and the Markdown a reader
compares differ, and which of the two is the source of truth is already settled — the
Markdown is.

## 2. The facts blocks

Fence syntax, with a stable id and a digest:

```markdown
<!-- drtools:ranking sha256=9f3a1c... -->
| candidate | score | ... |
<!-- /drtools:ranking -->
```

Keyed by id rather than by position, so `--refresh` still finds a block after the agent
has written three paragraphs above it. The skeleton's section order is fixed by the
design; the toolbox does not enforce it and does not need to in order to refresh.

The inventory, by section:

| § | Section | Block | Source |
|---|---|---|---|
| 1 | Dataset profile | shape, modality, value kind, sparsity, dataset digest, adapter provenance | `profile.json`, cache meta |
| 2 | Preprocessing, and why | base preprocessing as registered; `plan`-stage records with what their Evidence resolved to | `plan.registered.json`, log |
| 3 | Methods selected and rejected | Candidates with Stages and rationale; Rejections with reason and Evidence | `plan.registered.json`, log |
| 4 | Hyperparameters | per Candidate, the params that differ from the op default, with the `suggest-params` reading where one was logged | `plan.registered.json`, log |
| 5 | Figures | the figures actually drawn, by path | `figures/` |
| 6 | Quantitative comparison | the metrics table, marked with which metrics were dropped from the ranking | `metrics/*.json` |
| 7 | Ranking | scores, `weights_declared` against `weights_applied`, `weights_dropped`, the tie notes, `failed_candidates` | `ranking.json` |
| 8 | Interpretation | **none** | — |
| 9 | Limitations | the mechanical ones only: metrics dropped and the weight that moved with them, Candidates tied inside the noise, any subsampling and what the metrics therefore describe | `ranking.json`, `metrics/*.json` |

**Section 8 has no block, and that is the point of the split.** Nothing in the Run
grounds an interpretation. A toolbox contribution there would lend the appearance of
derivation to the one section that is entirely the agent's judgment — the failure mode
section 2.3 already had to narrow a claim over once.

Section 7 is nearly free. `rank` already emits `weights_declared`, `weights_applied`,
`weights_dropped`, `failed_candidates`, and a `notes` list carrying the tie note when
the top margin falls inside the noise of stochastic methods and repeated subsampling.
Those are precisely the qualifications `write-report` instructs the agent to carry,
already structured, and today they reach the report only if the agent remembers them.

Section 9's block carries the mechanical limitations and no others. The judgment ones —
a weighting the agent would now choose differently, what the failures say about the
data — stay prose. The subsampling line comes from `metrics.py`, which already records
`subsampled` with the counts behind it, per Candidate.

A block whose source is absent — no figures drawn, no reconnaissance — prints an
explicit line saying the Run did not produce it, rather than being omitted. An omitted
block is indistinguishable from a section the agent has not reached, and the difference
matters to a reader deciding whether something was skipped or was never available.

## 3. Refusals

- **`report` refuses unless the Run is at the report stage.** `status` already derives
  `next`; this reads it rather than recomputing the condition. One exception, which
  `write-report` already carries: a Run where every Candidate failed still gets a
  report, and sections 6 and 7 then hold the failure record instead of a table.

- **`report` without `--refresh` refuses an existing `report.md`,** and names
  `--refresh`. Overwriting is how the agent's prose would be lost.

- **`--refresh` refuses a block that was edited by hand.** The digest in the fence is
  over the content the toolbox last wrote; if the content no longer matches, the block
  was edited, and regenerating it would destroy that edit while doing something else.
  This is the defect class found in `embed` on day 9 — a command destroying work as a
  side effect of doing something unrelated. The message names the block and says prose
  belongs outside the fence.

- **`--refresh` reports a block the document no longer holds; it does not re-insert
  it.** The toolbox cannot know where in the agent's prose a deleted fence belonged,
  and guessing would drop a table into the middle of a paragraph. The report names the
  missing ids so the agent can paste the fence back where it wants it, and running
  `report` on a fresh file is the other route.

- **`render` refuses a missing pandoc and a missing PDF engine separately.** They are
  two different installs and two different fixes. Both are present in this environment
  (pandoc 3.9; MiKTeX with pdflatex, xelatex and lualatex; `pandoc x.md -o x.pdf`
  succeeds), so the refusals are tested by faking absence rather than by an environment
  that exhibits it.

No override flag on any of these, consistent with day 7's "refuse, never warn".

## 4. What this changes elsewhere

`skills/write-report/SKILL.md` gains the two commands. Day 9 wrote "rendering to PDF is
a separate step" without naming what performs it, and told the agent to write the
document without saying that most of its numbers arrive already written.

`tests/test_skills.py` covers the new commands and flags automatically, since it scans
the prose for every `drtools <command> --flag` it names.

## 5. Testing

- A block refreshes in place with the prose around it preserved byte for byte.
- A block whose content was hand-edited refuses to refresh, and the message names it.
- `--refresh` reports which blocks changed, so the agent knows what to re-read.
- Every number a block prints traces to a key in an artefact of the Run.
- A Run where every Candidate failed produces a report, with the failure record in
  sections 6 and 7.
- `report` refuses a Run that is not at the report stage.
- `report` refuses to overwrite, and names `--refresh`.
- `render` refuses a missing pandoc, and a missing PDF engine, with different messages.
- The rendered PDF exists and is non-trivial; its content is not asserted.

## 6. Out of scope

**Whether PBMC3k gets derived Leiden reference labels.** Deliberately deferred by
section 7 of the notes to when it is reached, and it is a question about the Run rather
than about the report. It needs its own decision: labels derived from a
PCA-neighbour-graph-Leiden pipeline would flatter embeddings preserving that same
neighbourhood structure, so using them as a metric input would bias the ranking toward
methods resembling the pipeline that produced the labels. Raised when the end-to-end run
starts.

**The manually written four-page `report.pdf`** of section 8's deliverables. That is a
human document about the project, not a generated one about a Run.

**An Amendment.** Still defined in `CONTEXT.md` and still unimplemented; the report says
so where a weighting would have moved.
