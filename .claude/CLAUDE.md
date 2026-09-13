# dr-agent

A Claude Code plugin that runs dimension-reduction analyses autonomously, built over
fourteen work days against a 2026-09-28 deadline. The schedule in `design/notes.md` is
indexed by day number, not dated: the build has run ahead of the calendar, and the index
is what it tracks.

## Where decisions live

**`design/notes.md` is this project's authoritative record.** Read it and `CONTEXT.md`
before exploring the codebase: the notes carry the architecture, the decisions and the
reasoning behind them, while `CONTEXT.md` carries the vocabulary they are written in.
Append to the notes' decision log as each work day closes, keyed by day index — what was
decided, why, and what was rejected.

`CONTEXT.md` is the glossary, and it is live from day 6 — the five skills are written
on days 7 and 8 and must name concepts in its terms. `docs/adr/` is created on **day
13** for ADR-0001, on the locked-core/open-adapter action space, and holds nothing
else. There is no prose `docs/` tree: an architecture page would be a third copy of
what the README, the four-page report and the notes already carry between them.

This defers the global rule that architectural work is unfinished until `docs/` reflects
it. Every day of this build is architectural by that rule's own test — a new package, a
new pipeline, new interface functions — so applying it per session would mean keeping two
records of the same decisions for fourteen days and reconciling them at the end. One pass
at day 13 costs less and contradicts less.
