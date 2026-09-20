# dr-agent

A Claude Code plugin that runs dimension-reduction analyses autonomously, built over
fourteen work days against a 2026-09-28 deadline. The schedule in `design/notes.md` is
indexed by day number, not dated: the build has run ahead of the calendar, and the index
is what it tracks.

## Where decisions live

**`design/notes.md` is this project's authoritative record.** Read it and `CONTEXT.md`
before exploring the codebase: the notes carry the architecture, the decisions and the
reasoning behind them, while `CONTEXT.md` carries the vocabulary they are written in.
Append to the notes' decision log as each work day closes, keyed by day index — what
was decided, why, and what was rejected. That log is how this project records a
decision, and it stands in for the ADR the global rule asks of bounded work.
`docs/adr/` is created on **day 13** for ADR-0001, on the locked-core/open-adapter
action space, and is the whole of `docs/` — there is no prose documentation tree.

`CONTEXT.md` is the glossary, and it governs rather than describes: the five skills are
prose the agent reads at runtime, so they must name concepts in its terms.

## Scoping a work day

The global rule's `design/notes.md` branch governs here, and it turns on whether the
notes carry *this* day's design — not on whether the project has notes. The test: can
you name the mechanism from the notes, or only the requirement?

Most days pass it. The notes settle the design up front, so the day is bounded whatever
its interface surface, and the work is to implement it and append to the decision log.

A day briefed as *make X hold* — the notes fixing the property and leaving the mechanism
open — is architectural, and its size is not the deciding fact. Contract work reached
that way is what every later day is built on, so the design is worth reviewing before
any code exists.
