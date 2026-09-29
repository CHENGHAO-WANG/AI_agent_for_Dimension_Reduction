"""The report: what the toolbox writes into it, and what it must never touch.

The document has two owners. Every number belongs to the toolbox and lives inside a
fenced block; everything else is the agent's prose. The split exists so that the agent
never retypes a number -- a figure in the report cannot disagree with the artefact it
came from, because there was no opportunity to transcribe it. That is a stronger
guarantee than citation integrity and composes with it: citation integrity says a cited
key resolves, and this says the value printed is the value it resolved to.

The fence carries a digest of the body the toolbox last wrote. That is what lets a
refresh tell "this block is mine to regenerate" from "the agent edited this", and
refuse the second rather than silently destroying the edit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drtools import jsonio
from drtools.export import exported_candidate
from drtools.plots import PLOT_B_METHOD
from drtools.registry import load_registry
from drtools.runs import RunDir
from drtools.tuning import d_curve

BLOCK_IDS = (
    "profile",
    "preprocessing",
    "methods",
    "hyperparameters",
    "figures",
    "metrics",
    "ranking",
    "limitations",
    "export",
)
"""The nine generated blocks, in section order.

Section 8, Interpretation, is absent deliberately: nothing in the run grounds it, and a
generated block there would lend the appearance of derivation to the one section that
is entirely the agent's judgment.
"""

_OPEN = re.compile(
    r"<!-- drtools:(?P<id>[a-z_]+) sha256=(?P<digest>[0-9a-f]{16}) -->\r?\n"
)
_DIGEST_CHARS = 16


def _normalise(body: str) -> str:
    """The one canonical form of a block body, used everywhere a body is compared.

    Line endings collapse because git rewrites them on checkout in this repo, and a
    block that read as hand-edited after a clone would make `--refresh` useless on a
    fresh checkout.

    Trailing blank lines go because `parse_blocks` cannot keep them: the closing comment
    sits on its own line, so reading a body back always strips what precedes it. A
    generator that ends its body with a blank line -- `_block_ranking` does, whenever a
    run produced no ranking notes -- would otherwise write a body that does not match
    its own digest the moment it is read back, and every refresh would refuse a block
    nobody had touched. Normalising in one place is what keeps written, parsed and
    regenerated bodies comparable.
    """
    return body.replace("\r\n", "\n").rstrip("\n")


def _digest(body: str) -> str:
    """Over the body in canonical form."""
    return hashlib.sha256(_normalise(body).encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


@dataclass(frozen=True)
class ParsedBlock:
    """One fenced region as found in a document.

    `start` and `end` span the whole region, opening comment through closing comment,
    so that replacing it leaves every byte outside untouched.
    """

    id: str
    body: str
    digest: str
    start: int
    end: int


def fence(block_id: str, body: str) -> str:
    """One complete fenced region: opening comment with digest, body, closing comment.

    The body is written in canonical form, so that what is written is exactly what
    `parse_blocks` reads back and exactly what the digest covers.
    """
    body = _normalise(body)
    return (
        f"<!-- drtools:{block_id} sha256={_digest(body)} -->\n"
        f"{body}\n"
        f"<!-- /drtools:{block_id} -->"
    )


def parse_blocks(document: str) -> dict[str, ParsedBlock]:
    """Every fenced region in the document, keyed by id.

    The closing comment is searched for by id rather than by a generic pattern, so a
    body that happens to contain the word `drtools` -- a path in a metrics note, a
    command quoted from a refusal -- does not end the block early.
    """
    blocks: dict[str, ParsedBlock] = {}
    for match in _OPEN.finditer(document):
        block_id = match.group("id")
        closing = f"<!-- /drtools:{block_id} -->"
        close_at = document.find(closing, match.end())
        if close_at == -1:
            continue
        body = document[match.end() : close_at].rstrip("\r\n")
        blocks[block_id] = ParsedBlock(
            id=block_id,
            body=body,
            digest=match.group("digest"),
            start=match.start(),
            end=close_at + len(closing),
        )
    return blocks


def edited(block: ParsedBlock) -> bool:
    """Whether the body no longer matches the digest the toolbox wrote beside it."""
    return _digest(block.body) != block.digest


def replace_block(document: str, block_id: str, body: str) -> str:
    """Swap one block's fenced region, leaving every other byte of the document alone."""
    blocks = parse_blocks(document)
    if block_id not in blocks:
        raise KeyError(block_id)
    block = blocks[block_id]
    return document[: block.start] + fence(block_id, body) + document[block.end :]


# ------------------------------------------------------------------ block contents


NOT_PRODUCED = "_This Run did not produce this._"
"""What a block prints when its source does not exist.

Printed rather than omitted. An omitted block is indistinguishable from a section the
agent has not reached yet, and the difference matters to a reader deciding whether
something was skipped or was never available.
"""


def _read(run: RunDir, *parts: str) -> Any | None:
    path = run.path.joinpath(*parts)
    return jsonio.read(path) if path.exists() else None


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _candidate_ids(run: RunDir) -> list[str]:
    plan = _read(run, "plan.registered.json")
    return [candidate["id"] for candidate in plan["candidates"]] if plan else []


def _block_profile(run: RunDir) -> str:
    profile = _read(run, "profile.json")
    if profile is None:
        return NOT_PRODUCED
    meta = _read(run, "data", "meta.json") or {}
    shape, values, labels = profile["shape"], profile["values"], profile["labels"]

    rows = [
        ["Dataset", f"{profile['name']} (`{profile['spec']}`, {profile['source']})"],
        ["Samples", f"{shape['n_samples']}"],
        ["Features", f"{shape['n_features']}"],
        ["Storage", f"{shape['storage']}, {shape['dtype']}, {shape['memory_mb']} MB"],
        ["Sparsity", f"{values['sparsity']:.4f}"],
        ["Values", f"{values['suspected_kind']}"],
        ["Identity", f"`{profile['dataset_digest'][:12]}`"],
    ]
    if profile.get("modality"):
        rows.insert(1, ["Modality", profile["modality"]])
    if labels["present"]:
        rows.append(
            [
                "Labels",
                f"{labels['n_classes']} classes ({labels['kind']}), "
                f"balance ratio {labels['balance_ratio']:.2f}",
            ]
        )
    else:
        rows.append(["Labels", "none"])
    # The adapter is the one piece of agent-written code in an analysis. A path alone
    # dates badly, so the record carries a digest of the source that actually ran.
    adapter = meta.get("adapter")
    if adapter:
        rows.append(
            [
                "Adapter",
                f"`{adapter['path']}` (`{adapter['sha256'][:12]}`, "
                f"{adapter['size']} bytes)",
            ]
        )

    body = _table(["Property", "Value"], rows)
    observations = profile.get("observations") or []
    if observations:
        body += "\n\n" + "\n".join(
            f"- {observation['statement']}"
            if isinstance(observation, dict) and "statement" in observation
            else f"- {observation}"
            for observation in observations
        )
    return body


def _decision_prose(run: RunDir, stage: str) -> str:
    """Records the agent logged at one stage, with what their Evidence resolved to.

    `evidence_resolved` is written by `log-decision` and is the reading at the time the
    decision was made. Carrying it is the difference between a citation and a claim
    that the citation still says what it said.
    """
    lines: list[str] = []
    for record in run.decisions():
        if record.get("stage") != stage:
            continue
        lines.append(f"**{record['question']}**")
        lines.append(f"Chosen: {record['chosen']}. {record['rationale']}")
        for key, value in (record.get("evidence_resolved") or {}).items():
            lines.append(f"- `{key}` = {value}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _block_preprocessing(run: RunDir) -> str:
    """The data decision, the base the rule gave or the departure from it, and what ran.

    Section 3.10 has the agent tell the user twice when raw counts are transformed: at
    planning, in the conversation, and here. The sentence is fixed and the target is
    the one `normalise_total` recorded when the Reference was prepared, so the report
    states what was done rather than what was meant.
    """
    plan = _read(run, "plan.registered.json")
    if plan is None:
        return NOT_PRODUCED
    parts: list[str] = []

    decision = (_read(run, "recon.json") or {}).get("data_decision")
    if decision:
        line = (
            f"Data decision: values `{decision['values']}`, features "
            f"`{decision['features']}`, decided by {decision['decided_by']}"
        )
        if decision.get("evidence"):
            line += ", citing " + ", ".join(f"`{key}`" for key in decision["evidence"])
        parts.append(line + ".")

    registration = next(
        (r for r in reversed(run.decisions()) if r.get("stage") == "register_plan"), {}
    )
    if registration.get("base") == "rule":
        parts.append("The base follows the rule for that decision.")
    elif registration.get("base") == "departure":
        departure = registration.get("base_departure") or {}
        cited = ", ".join(f"`{key}`" for key in departure.get("evidence") or [])
        parts.append(
            f"The base departs from the rule for that decision: {departure.get('reason')}"
            + (f" (citing {cited})" if cited else "")
            + "."
        )

    base = plan.get("base_preprocessing") or []
    if base:
        parts.append(
            "Base preprocessing, applied to every Candidate and to the reference:\n\n"
            + "\n".join(
                f"{i}. `{stage['op']}`"
                + (f" {stage['params']}" if stage.get("params") else "")
                for i, stage in enumerate(base, start=1)
            )
        )
    else:
        parts.append(
            "No base preprocessing: Candidates were scored against the cached matrix."
        )

    records = (_read(run, "data", "reference.json") or {}).get("stage_records") or []
    notes = {record["op"]: record.get("notes") or {} for record in records}
    if "normalise_total" in notes and "log1p" in notes:
        target = notes["normalise_total"].get("target_total")
        parts.append(
            "The values were judged to be raw counts, so each sample was rescaled to the "
            f"median total ({target:,.6g}) and then transformed by log(1 + x). Data "
            "already transformed should be declared as such when the analysis starts, "
            "or supplied transformed."
        )

    tolerated = notes.get("drop_constant", {}).get("dropped_by_tolerance") or []
    if tolerated:
        names = (_read(run, "data", "meta.json") or {}).get("feature_names")
        shown = [
            f"`{names[index]}`" if names is not None and index < len(names) else f"column {index}"
            for index in tolerated
        ]
        parts.append(
            f"{len(tolerated)} feature(s) were dropped as constant only by the "
            "tolerance, their range not exactly zero but within 1e-12 of their "
            "magnitude: " + ", ".join(shown) + ". A genuine feature measured in very "
            "small units would be dropped this way, so check they are what they seem."
        )

    decisions = _decision_prose(run, "plan")
    if decisions:
        parts.append(decisions)
    return "\n\n".join(parts)


def _block_methods(run: RunDir) -> str:
    plan = _read(run, "plan.registered.json")
    if plan is None:
        return NOT_PRODUCED

    selected = _table(
        ["Candidate", "Pipeline", "Why"],
        [
            [
                candidate["id"],
                " -> ".join(f"`{stage['op']}`" for stage in candidate["stages"]),
                candidate.get("rationale") or "—",
            ]
            for candidate in plan["candidates"]
        ],
    )

    rejected = plan.get("rejected") or []
    if rejected:
        rejected_table = _table(
            ["Method", "Why not", "Evidence"],
            [
                [
                    rejection["method"],
                    rejection["reason"],
                    ", ".join(f"`{key}`" for key in rejection.get("evidence") or [])
                    or "—",
                ]
                for rejection in rejected
            ],
        )
    else:
        # Printed rather than omitted: an empty section 3 would read as an oversight,
        # and this is the section the design calls the clearest evidence that the agent
        # selected rather than sprayed.
        rejected_table = "_No method was rejected in this Run._"

    return f"**Selected**\n\n{selected}\n\n**Rejected**\n\n{rejected_table}"


def _block_hyperparameters(run: RunDir) -> str:
    """What each Candidate actually ran with, and which values the agent chose.

    From the embedding record rather than the Plan: the Plan holds what was asked for,
    and the record holds what ran, with registry defaults filled in and
    `param_provenance` marking which is which: `registry_default`, `suggested`,
    `overridden` -- printed with the suggestion and the reason -- or `specified`.
    """
    ids = _candidate_ids(run)
    if not ids:
        return NOT_PRODUCED

    sections = []
    for candidate_id in ids:
        record = _read(run, "embeddings", f"{candidate_id}.json")
        if record is None:
            sections.append(f"**{candidate_id}** — not run.")
            continue
        rows = []
        for stage in record["stages"]:
            provenance = stage.get("param_provenance") or {}
            overrides = stage.get("param_overrides") or {}
            for name, value in (stage.get("params") or {}).items():
                source = provenance.get(name, "unrecorded")
                if name in overrides:
                    override = overrides[name]
                    source = (
                        f"overridden (suggested {override.get('suggested')}): "
                        f"{override.get('reason')}"
                    )
                rows.append([f"`{stage['op']}`", f"`{name}`", f"{value}", source])
        if rows:
            sections.append(
                f"**{candidate_id}**\n\n"
                + _table(["Stage", "Parameter", "Value", "Source"], rows)
            )
        else:
            sections.append(f"**{candidate_id}** — no parameters to record.")
        tuned = _tuning_line(record.get("tuning"))
        if tuned:
            sections.append(tuned)
    return "\n\n".join(sections)


def _tuning_line(record: dict[str, Any] | None) -> str | None:
    """How tuning chose d and the fidelity parameter, from the tuning record."""
    if not record or "chosen" not in record:
        return None
    chosen, method = record["chosen"], record["method"]
    curve = d_curve(record)
    parts = [f"Tuned: d = {chosen['d']}"]
    if curve:
        parts.append(f"chosen by the {curve['rule']} rule on its curve of {curve['label']}")
    else:
        parts.append("fixed, as a picture is drawn at d = 2")
    line = ", ".join(parts)
    base = method.get("base") or {}
    if method.get("param") and base.get("source") == "rule":
        line += (
            f"; `{method['param']}` at {chosen['multiplier']:g} times the width the "
            "executor's rule computes"
        )
    elif method.get("param") and chosen.get("value") is not None:
        line += (
            f"; `{method['param']}` = {chosen['value']}, {chosen['multiplier']:g} times "
            f"the {base.get('source', 'suggestion')} of {base.get('value')}"
        )
        if base.get("at_rows"):
            line += f" at {base['at_rows']:,} rows"
    if chosen.get("at_grid_edge"):
        line += ", at the edge of the multiplier grid, so a value beyond it might score higher"
    return line + "."


def _block_figures(run: RunDir) -> str:
    drawn = _read(run, "results", "figures", "figures.json")
    if not drawn:
        return NOT_PRODUCED

    visualization = run.purpose() == "visualization"
    terminal = _terminal_ops(run)
    competitors = [] if visualization else _shown_competitors(run)
    parts: list[str] = []
    for name, record in drawn.items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        try:
            relative = Path(record["path"]).relative_to(run.results_dir).as_posix()
        except ValueError:
            # Drawn somewhere else entirely. Naming it beats embedding a path that will
            # not resolve from the report.
            parts.append(f"_{name} was drawn outside this Run and is not embedded._")
            continue
        parts.append(f"**{name}**\n\n![{name}]({relative})")
        if record.get("caveat"):
            parts.append(f"_{record['caveat']}_")
        # Section 3.11: what the picture's distances, gaps and sizes mean comes from
        # the method that drew it -- the candidate's own in a visualization run, UMAP
        # under every plot B.
        caption = _caption(run, name, visualization, competitors)
        if caption:
            parts.append(caption)
        if name.startswith("plot_b_"):
            parts.append(_reading_line(PLOT_B_METHOD))
        elif visualization and name.startswith("embedding_"):
            op = terminal.get(name.removeprefix("embedding_"))
            if op:
                parts.append(_reading_line(op))
        channel = record.get("identity_channel")
        if channel and channel != "labels":
            parts.append(f"_Identity is carried by {channel} in this figure._")
    return "\n\n".join(parts) if parts else NOT_PRODUCED


def _caption(
    run: RunDir, name: str, visualization: bool, competitors: list[str]
) -> str | None:
    """What a representation run's figure is, where its name alone does not say."""
    if name == "d_curves":
        return (
            "_Each panel is on its own criterion's scale, so the panels are not compared "
            "with one another. The ringed point is the chosen d; the rest of the curve "
            "is what choosing it gave up or saved._"
        )
    if visualization:
        return None
    for prefix, kind in (("embedding_", "A"), ("plot_b_", "B")):
        if name.startswith(prefix):
            candidate = name.removeprefix(prefix)
            plots = (_read(run, "metrics", f"{candidate}.json") or {}).get("plots") or {}
            if kind == "A" and plots.get("A"):
                return (
                    f"_Plot A: the d = {plots['d']} representation on its first two "
                    f"principal axes, which carry {plots['A']['variance_share']:.0%} of "
                    "its variance. A rotation, so the distances within those two axes "
                    "are the representation's own; the variance beyond them is not "
                    "shown._"
                )
            if kind == "B" and plots.get("B"):
                return (
                    f"_Plot B: {PLOT_B_METHOD} of the representation at the same fixed "
                    f"settings for every candidate (n_neighbors = "
                    f"{plots['B']['n_neighbors']}, the run's seed). A picture of the "
                    "representation, not the representation, and not scored._"
                )
    for prefix in ("class_facet_", "shepard_"):
        if name.startswith(prefix) and name.removeprefix(prefix) in competitors:
            return (
                f"_Drawn for {name.removeprefix(prefix)} because it is a close "
                "competitor: the margin could not separate it from the winner, so these "
                "are the figures a reader separates them with._"
            )
    return None


def _shown_competitors(run: RunDir) -> list[str]:
    """The close competitors given diagnostic figures: the first three (section 3.8)."""
    ranking = _read(run, "ranking.json") or {}
    return [c["id"] for c in ranking.get("close_competitors") or []][:3]


def _block_metrics(run: RunDir) -> str:
    scored = {}
    for candidate_id in _candidate_ids(run):
        record = _read(run, "metrics", f"{candidate_id}.json")
        if record is not None:
            scored[candidate_id] = record
    if not scored:
        return NOT_PRODUCED

    names = sorted({name for record in scored.values() for name in record["values"]})
    body = _table(
        ["Candidate"] + [f"`{name}`" for name in names],
        [
            [candidate_id]
            + [
                # A metric recorded as None -- silhouette without labels -- is a dash.
                f"{record['values'][name]:.4f}"
                if record["values"].get(name) is not None
                else "—"
                for name in names
            ]
            for candidate_id, record in scored.items()
        ],
    )

    ranking = _read(run, "ranking.json") or {}
    dropped = ranking.get("weights_dropped") or {}
    if dropped:
        body += "\n\n" + "\n".join(
            f"- `{name}` is shown here but was excluded from the score: {reason}"
            for name, reason in dropped.items()
        )
    settings = next(iter(scored.values()))["settings"]
    body += (
        f"\n\nMeasured at k={settings['k']}, capped at {settings['max_samples']} "
        f"samples, seed {settings['seed']}."
    )
    return body + "\n\n" + _coverage_table(run, list(scored))


def _coverage_table(run: RunDir, candidates: list[str]) -> str:
    """Rows fitted and projected, how the rest were placed, and what each cost."""
    rows = []
    for candidate_id in candidates:
        record = _read(run, "embeddings", f"{candidate_id}.json") or {}
        coverage = record.get("rows") or {}
        placed = "—"
        if coverage.get("n_projected"):
            kinds = [
                stage["projection"]["kind"]
                for stage in record.get("stages", [])
                if stage.get("projection")
            ]
            placed = kinds[-1] if kinds else "unrecorded"
        tuning = (record.get("tuning") or {}).get("duration_s")
        rows.append([
            candidate_id,
            f"{coverage.get('n_fitted', 0):,}",
            f"{coverage.get('n_projected', 0):,}",
            placed,
            "—" if tuning is None else f"{tuning:.1f}",
            f"{record.get('total_duration_s', 0):.1f}",
        ])
    return (
        "Coverage and run time. Every candidate covers every row; tuning is the search, "
        "and the last column is the fit and projection that produced the Embedding.\n\n"
        + _table(
            ["Candidate", "Rows fitted", "Rows projected", "New rows placed by",
             "Tuning (s)", "Fit and projection (s)"],
            rows,
        )
    )


#: Which kinds of ranking note are limitations, and so repeat in section 9. Every kind
#: `rank` writes is classified here, and a test holds this to `NOTE_KINDS`. The two
#: kinds marked False are limitations too, but section 9 prints them from their own
#: fields rather than from the note.
LIMITATION_KINDS: dict[str, bool] = {
    "weights_dropped": False,
    "failed_candidates": False,
    "close_competitors": True,
    "not_discriminated": True,
    "standard_error_scope": True,
    "standard_error_unavailable": True,
}


def _block_ranking(run: RunDir) -> str:
    if run.purpose() == "visualization":
        return _block_comparison(run)
    ranking = _read(run, "ranking.json")
    if ranking is None:
        return NOT_PRODUCED

    applied = ranking["weights_applied"]
    rows = []
    for entry in ranking["ranking"]:
        contributions = [
            f"{entry['contributions'][name]['contribution']:.4f}"
            if name in entry["contributions"]
            else "—"
            for name in applied
        ]
        if entry["rank"] == 1:
            standing = "winner"
        elif entry.get("close_competitor"):
            standing = f"close, {entry['difference_from_winner']:+.4f}"
        else:
            standing = ""
        rows.append(
            [str(entry["rank"]), entry["id"], str(entry["d"]), f"{entry['score']:.4f}",
             _se(entry.get("se")), _se(entry.get("se_difference")), standing]
            + contributions
        )
    body = _table(
        ["Rank", "Candidate", "d", "Score", "SE", "Paired SE", "Within the margin"]
        + [f"`{name}`" for name in applied],
        rows,
    )

    winner = next(entry for entry in ranking["ranking"] if entry["rank"] == 1)
    lines = ["", f"Winner: **{ranking['winner']}** (d = {winner['d']}).", ""]
    competitors = ranking.get("close_competitors") or []
    if competitors and ranking.get("discriminated", True):
        lines.append(
            f"Close competitors, within {ranking['margin']} of the leader "
            f"{ranking['leader']}: "
            + "; ".join(
                f"{c['id']} (d = {c['d']}, {c['difference_from_winner']:+.4f})"
                for c in competitors
            )
            + "."
        )
    elif competitors:
        lines.append(
            f"{len(competitors) + 1} candidates lie within {ranking['margin']} of the "
            f"leader {ranking['leader']}; the ranking did not discriminate among them."
        )
    lines.append(
        "Weighting declared before any Embedding existed: "
        + ", ".join(
            f"`{name}` {weight}"
            for name, weight in ranking["weights_declared"].items()
        )
        + "."
    )
    if ranking["weights_declared"] != applied:
        lines.append(
            "Weighting actually applied, renormalised over the metrics that survived: "
            + ", ".join(f"`{name}` {weight}" for name, weight in applied.items())
            + "."
        )
    if ranking.get("justification"):
        lines.append(f"Justification as registered: {ranking['justification']}")
    for name, reason in (ranking.get("weights_dropped") or {}).items():
        lines.append(f"- `{name}` was dropped from the score: {reason}")
    if ranking.get("failed_candidates"):
        lines.append(
            "Candidates that produced no Embedding, and are therefore unscored: "
            + ", ".join(ranking["failed_candidates"])
            + "."
        )
    # Verbatim. These are the qualifications `rank` computed -- the close competitors,
    # the scope of the standard errors -- and paraphrasing them here would be the
    # report making a claim the toolbox did not.
    lines += [""] + [f"- {note['text']}" for note in ranking.get("notes") or []]
    lines += ["", _path_block(ranking)]
    return body + "\n" + "\n".join(lines)


def _se(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _path_block(ranking: dict[str, Any]) -> str:
    """The path of winners: who wins as a dimension is priced from nothing upwards."""
    steps = ranking.get("path") or []
    if not steps:
        return ""
    rows = [
        [step["id"], str(step["d"]), f"{step['score']:.4f}",
         f"{step['from_rate']:.4f}",
         "no limit" if step["to_rate"] is None else f"{step['to_rate']:.4f}"]
        for step in steps
    ]
    return (
        "Path of winners: the candidate that wins when each dimension costs a given "
        "amount of score, from nothing upwards.\n\n"
        + _table(["Candidate", "d", "Score", "Wins from a price of", "to"], rows)
        + "\n\n"
        + "\n".join(f"- {sentence}" for sentence in ranking.get("path_sentences") or [])
    )


def _block_comparison(run: RunDir) -> str:
    """A visualization run's section 7: each metric on its own, and the judgment made.

    No weighted total and no winner (section 3.11). The recommendation and any adoption
    are printed as recorded, labelled as what they are: choices made after the results.
    """
    comparison = _read(run, "comparison.json")
    if comparison is None:
        return NOT_PRODUCED
    # After a re-plan round the previous judgments stay on disk; each counts only for
    # the comparison it was made of.
    current = comparison.get("plan_digest")
    recommendation = _read(run, "recommendation.json") or {}
    if recommendation.get("plan_digest") != current:
        recommendation = {}
    adopted = _read(run, "adopted.json")
    if adopted and adopted.get("plan_digest") != current:
        adopted = None
    chosen = recommendation.get("recommended") or []
    order = [c for c in comparison["candidates"] if c in chosen] + [
        c for c in comparison["candidates"] if c not in chosen
    ]
    names = list(comparison["metrics"])
    rows = []
    for candidate in order:
        cells = []
        for name in names:
            metric = comparison["metrics"][name]
            value = metric["values"].get(candidate)
            if value is None:
                cells.append("—")
                continue
            mark = (
                " (best)"
                if metric["best"] == candidate
                else " (within margin)"
                if candidate in metric["within_margin"]
                else ""
            )
            cells.append(f"{value:.4f}{mark}")
        rows.append([candidate, "yes" if candidate in chosen else ""] + cells)
    body = _table(["Candidate", "Recommended"] + [f"`{name}`" for name in names], rows)

    lines = [
        "",
        f"Each metric is compared on its own, and none is summed with another: a "
        f"weighted total would choose the method by itself. \"Within margin\" is within "
        f"{comparison['margin']} of that metric's best.",
        "",
    ]
    if chosen:
        lines.append(
            f"Recommended for a {recommendation.get('focus', 'balanced')} focus: "
            f"**{', '.join(chosen)}**. Recommended by the agent after seeing the results; "
            "this is a judgment, not a measured ranking."
        )
        lines.append(f"The agent's reasons, as recorded: {recommendation.get('rationale', '')}")
    else:
        lines.append("_No recommendation has been recorded._")
    if adopted:
        reason = adopted.get("rationale") or "no reason was given"
        lines.append(
            f"The user adopted **{adopted['candidate']}** after seeing the pictures and "
            f"the recommendation: {reason}"
        )
    elif chosen:
        lines.append("The user did not choose a picture.")
    return body + "\n" + "\n".join(lines)


def _block_limitations(run: RunDir) -> str:
    """The mechanical limitations only.

    A weighting the agent would now choose differently, and what the failures say about
    the data, are judgments and stay prose.
    """
    if run.purpose() == "visualization":
        return _visualization_limitations(run)
    ranking = _read(run, "ranking.json")
    if ranking is None:
        return NOT_PRODUCED

    lines: list[str] = []
    for name, reason in (ranking.get("weights_dropped") or {}).items():
        declared = (ranking.get("weights_declared") or {}).get(name)
        moved = (
            f" The {declared} declared for it moved to the metrics that remained."
            if declared
            else ""
        )
        lines.append(f"- `{name}` was dropped from the score: {reason}.{moved}")

    for note in ranking.get("notes") or []:
        if LIMITATION_KINDS[note["kind"]]:
            lines.append(f"- {note['text']}")

    lines += _scored_rows_line(run)
    lines.append(_reproducibility_line(run, "choose a different winner"))

    if ranking.get("failed_candidates"):
        lines.append(
            "- No Embedding was produced for "
            + ", ".join(ranking["failed_candidates"])
            + ", so they carry no scores here."
        )
    return "\n".join(lines) if lines else "_Nothing mechanical to qualify._"


def _scored_rows_line(run: RunDir) -> list[str]:
    # Every candidate covers every row and is scored on the same rows (section 3.12),
    # so the scored sample is one fact about the comparison, not one per candidate.
    scored = [
        record
        for candidate_id in _candidate_ids(run)
        if (record := _read(run, "metrics", f"{candidate_id}.json"))
    ]
    if scored and scored[0].get("subsampled"):
        return [
            f"- Every candidate was scored on the same {scored[0]['n_used']} of "
            f"{scored[0]['n_total']} rows, drawn once under the run's seed."
        ]
    return []


def _visualization_limitations(run: RunDir) -> str:
    comparison = _read(run, "comparison.json")
    if comparison is None:
        return NOT_PRODUCED
    lines = [
        "- No candidate was ranked. Which pictures are recommended is the agent's "
        "judgment, made after seeing the results and citing the metrics one at a time; "
        "it is not a measured ranking, and pre-registration does not protect it."
    ]
    lines += _scored_rows_line(run)
    lines.append(_reproducibility_line(run, "recommend different pictures"))
    failed = [
        candidate_id
        for candidate_id in _candidate_ids(run)
        if candidate_id not in comparison["candidates"]
    ]
    if failed:
        lines.append(
            "- No Embedding was produced for " + ", ".join(failed)
            + ", so they carry no scores here."
        )
    return "\n".join(lines)


def _reproducibility_line(run: RunDir, outcome: str) -> str:
    """Section 2.2's sentence: reproducible conditional on the registered plan, no more."""
    manifest = _read(run, "run.json") or {}
    return (
        "- This report is reproducible conditional on its registered plan: replaying "
        f"`plan.registered.json` on the same data under seed {manifest.get('seed')} "
        "returns every number in it. The plan itself is not reproducible. Its "
        "candidates were nominated by the agent's judgment, which no seed governs, so "
        "running the analysis again may register a different portfolio and "
        f"{outcome}."
    )


def _block_export(run: RunDir) -> str:
    """Section 10: what `results/data/` holds, read from its manifest."""
    manifest = _read(run, "results", "data", "manifest.json")
    if manifest is None:
        candidate, why = exported_candidate(run)
        return f"_Nothing was exported: {why}._" if candidate is None else NOT_PRODUCED
    rows = manifest.get("rows") or {}
    files = manifest["files"]
    lines = [
        f"Exported: **{manifest['candidate']}**, {manifest['why_this_candidate']}, at "
        f"d = {manifest['d']}.",
        "",
        f"- `data/{files['coordinates']}`: one row per sample -- `sample_id`, whether the "
        f"method was `fitted` on the row or `projected` it ({rows.get('n_fitted', 0):,} "
        f"and {rows.get('n_projected', 0):,}), then `dim_1` to `dim_{manifest['d']}`.",
        "- `data/manifest.json`: the pipeline and its parameter values, the seed "
        f"({manifest['seed']}), d, the features kept, the z-score means and standard "
        "deviations, and the rows fitted and projected.",
    ]
    if files.get("loadings"):
        lines.append(
            f"- `data/{files['loadings']}`: the loadings, one row per feature entering "
            "the method, with the feature names."
        )
    lines.append(
        "- New samples: "
        + (
            f"the method can place them without refitting (`{manifest['new_rows']}`)."
            if manifest["places_new_samples_without_refitting"]
            else "the method cannot place them without refitting the whole pipeline."
        )
    )
    lines.append(
        "- No model objects are saved: a saved model often fails to load under another "
        "library version. The manifest carries what a refit needs."
    )
    return "\n".join(lines)


def _terminal_ops(run: RunDir) -> dict[str, str]:
    """Each registered candidate's last op: the method whose picture it is."""
    plan = _read(run, "plan.registered.json") or {}
    return {
        candidate["id"]: candidate["stages"][-1]["op"]
        for candidate in plan.get("candidates", [])
        if candidate.get("stages")
    }


def _reading_line(op: str) -> str:
    reading = load_registry()[op].reading
    return (
        f"_How to read a {op} picture. Distances: {reading['distances']} Gaps: "
        f"{reading['gaps']} Sizes: {reading['sizes']}_"
    )


_BUILDERS = {
    "profile": _block_profile,
    "preprocessing": _block_preprocessing,
    "methods": _block_methods,
    "hyperparameters": _block_hyperparameters,
    "figures": _block_figures,
    "metrics": _block_metrics,
    "ranking": _block_ranking,
    "limitations": _block_limitations,
    "export": _block_export,
}


def build_blocks(run: RunDir) -> dict[str, str]:
    """A body for every declared block. Never raises for an artefact that is absent.

    Bodies come back in canonical form, so that a caller comparing one against a body
    parsed out of a document is comparing like with like. Without this, every block
    whose generator ends on a blank line would read as changed on every refresh.
    """
    return {block_id: _normalise(_BUILDERS[block_id](run)) for block_id in BLOCK_IDS}


# ----------------------------------------------------------------- the document


SECTIONS: tuple[tuple[str, str | None], ...] = (
    ("1. Dataset profile", "profile"),
    ("2. Preprocessing decisions, and why", "preprocessing"),
    ("3. Methods selected and rejected", "methods"),
    ("4. Hyperparameter choices, and why", "hyperparameters"),
    ("5. Figures", "figures"),
    ("6. Quantitative comparison", "metrics"),
    ("7. Ranking, with the weighting justification", "ranking"),
    ("8. Interpretation", None),
    ("9. Limitations", "limitations"),
    ("10. Exported results", "export"),
)
"""The skeleton, matching `skills/write-report/SKILL.md` heading for heading.

Fixed so that the two generated reports can be read side by side. Section 8 carries no
block id, which is the whole shape of the split.
"""

#: Section 7's heading in a visualization run, which ranks nothing (section 3.11).
VISUALIZATION_SECTION_7 = "7. Comparison and recommendation"

_PROMPT = "_Yours to write. Delete this line._"


def assemble(run: RunDir) -> str:
    """The whole document: the ten headings, the nine blocks, and room to write."""
    bodies = build_blocks(run)
    parts = [f"# Dimension reduction report — {run.id}", ""]
    for heading, block_id in SECTIONS:
        if block_id == "ranking" and run.purpose() == "visualization":
            heading = VISUALIZATION_SECTION_7
        parts += [f"## {heading}", ""]
        if block_id is not None:
            parts += [fence(block_id, bodies[block_id]), ""]
        parts += [_PROMPT, ""]
    return "\n".join(parts)


def stale_blocks(run: RunDir, document: str) -> list[str]:
    """Ids whose generated body no longer matches what the document holds.

    Used by `--refresh` to decide what to rewrite, and by `render` to decide whether
    the document still agrees with the run. One function, so that the two commands
    cannot disagree about what stale means.
    """
    bodies = build_blocks(run)
    present = parse_blocks(document)
    return sorted(
        block_id
        for block_id, block in present.items()
        if block_id in bodies and block.body != bodies[block_id]
    )


def unnamed_competitors(run: RunDir, document: str) -> list[str]:
    """The shown close competitors section 8 does not name, for `render` to refuse.

    A substring test on each literal id (section 3.8). It is gameable -- an id can be
    named and nothing said -- but it turns a silent omission into a deliberate one.
    Applied at render and not at refresh, so drafting is never blocked.
    """
    if run.purpose() == "visualization":
        return []
    start = document.find("## 8. Interpretation")
    end = document.find("## 9.", start)
    section = document[start:end] if start != -1 else ""
    return [c for c in _shown_competitors(run) if c not in section]
