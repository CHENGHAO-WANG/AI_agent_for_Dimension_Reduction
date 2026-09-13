# dr-agent

A Claude Code plugin that runs dimension-reduction analyses autonomously, built over
fourteen work days against a 2026-09-28 deadline. The schedule in `design/notes.md` is
indexed by day number, not dated: the build has run ahead of the calendar, and the index
is what it tracks.

## Where decisions live

**`design/notes.md` is this project's authoritative record.** Read it before exploring
the codebase: it carries the architecture, the decisions, and the reasoning behind them,
standing in for the `CONTEXT.md` and `docs/adr/` that the global conventions expect.
Append to its decision log as each work day closes, keyed by day index — what was
decided, why, and what was rejected.

`docs/`, `docs/adr/` and `CONTEXT.md` are written on **day 13** in a single pass from the
notes. ADR-0001, on the locked-core/open-adapter action space, is already scheduled
there. Until that day the notes are the only record, and `docs/adr/` stays empty.

This defers the global rule that architectural work is unfinished until `docs/` reflects
it. Every day of this build is architectural by that rule's own test — a new package, a
new pipeline, new interface functions — so applying it per session would mean keeping two
records of the same decisions for fourteen days and reconciling them at the end. One pass
at day 13 costs less and contradicts less.
