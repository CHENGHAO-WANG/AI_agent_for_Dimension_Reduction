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
from drtools.runs import RunDir

BLOCK_IDS = (
    "profile",
    "preprocessing",
    "methods",
    "hyperparameters",
    "figures",
    "metrics",
    "ranking",
    "limitations",
)
"""The eight generated blocks, in section order.

Section 8, Interpretation, is absent deliberately: nothing in the run grounds it, and a
generated block there would lend the appearance of derivation to the one section that
is entirely the agent's judgment.
"""

_OPEN = re.compile(
    r"<!-- drtools:(?P<id>[a-z_]+) sha256=(?P<digest>[0-9a-f]{16}) -->\r?\n"
)
_DIGEST_CHARS = 16


def _digest(body: str) -> str:
    """Over the body as written, with line endings normalised.

    Normalised because git rewrites them on checkout in this repo, and a block that read
    as hand-edited after a clone would make `--refresh` useless on a fresh checkout.
    """
    return hashlib.sha256(
        body.replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()[:_DIGEST_CHARS]


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
    """One complete fenced region: opening comment with digest, body, closing comment."""
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
    plan = _read(run, "plan.registered.json")
    if plan is None:
        return NOT_PRODUCED

    base = plan.get("base_preprocessing") or []
    if base:
        body = "Base preprocessing, applied to every Candidate and to the reference:\n\n"
        body += "\n".join(
            f"{i}. `{stage['op']}`"
            + (f" {stage['params']}" if stage.get("params") else "")
            for i, stage in enumerate(base, start=1)
        )
    else:
        body = "No base preprocessing: Candidates were scored against the cached matrix."

    decisions = _decision_prose(run, "plan")
    return body + ("\n\n" + decisions if decisions else "")


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
    `param_provenance` marking which is which.
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
            for name, value in (stage.get("params") or {}).items():
                rows.append(
                    [
                        f"`{stage['op']}`",
                        f"`{name}`",
                        f"{value}",
                        provenance.get(name, "unrecorded"),
                    ]
                )
        if rows:
            sections.append(
                f"**{candidate_id}**\n\n"
                + _table(["Stage", "Parameter", "Value", "Source"], rows)
            )
        else:
            sections.append(f"**{candidate_id}** — no parameters to record.")
    return "\n\n".join(sections)


def _block_figures(run: RunDir) -> str:
    drawn = _read(run, "figures", "figures.json")
    if not drawn:
        return NOT_PRODUCED

    parts: list[str] = []
    for name, record in drawn.items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        try:
            relative = Path(record["path"]).relative_to(run.path).as_posix()
        except ValueError:
            # Drawn somewhere else entirely. Naming it beats embedding a path that will
            # not resolve from the report.
            parts.append(f"_{name} was drawn outside this Run and is not embedded._")
            continue
        parts.append(f"**{name}**\n\n![{name}]({relative})")
        if record.get("caveat"):
            parts.append(f"_{record['caveat']}_")
        channel = record.get("identity_channel")
        if channel and channel != "labels":
            parts.append(f"_Identity is carried by {channel} in this figure._")
    return "\n\n".join(parts) if parts else NOT_PRODUCED


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
                f"{record['values'][name]:.4f}" if name in record["values"] else "—"
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
    return body


def _block_ranking(run: RunDir) -> str:
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
        rows.append(
            [str(entry["rank"]), entry["id"], f"{entry['score']:.4f}"] + contributions
        )
    body = _table(
        ["Rank", "Candidate", "Score"] + [f"`{name}`" for name in applied], rows
    )

    lines = ["", f"Winner: **{ranking['winner']}**.", ""]
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
    # Verbatim. These are the qualifications `rank` computed -- a tie inside the noise,
    # a runtime span that dominates the comparison -- and paraphrasing them here would
    # be the report making a claim the toolbox did not.
    lines += [""] + [f"- {note}" for note in ranking.get("notes") or []]
    return body + "\n" + "\n".join(lines)


def _block_limitations(run: RunDir) -> str:
    """The mechanical limitations only.

    A weighting the agent would now choose differently, and what the failures say about
    the data, are judgments and stay prose.
    """
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
        if "tied" in note or "noise" in note:
            lines.append(f"- {note}")

    for candidate_id in _candidate_ids(run):
        record = _read(run, "metrics", f"{candidate_id}.json")
        if record and record.get("subsampled"):
            lines.append(
                f"- {candidate_id} was scored on {record['n_used']} of "
                f"{record['n_total']} rows, so its metrics describe that subsample."
            )

    if ranking.get("failed_candidates"):
        lines.append(
            "- No Embedding was produced for "
            + ", ".join(ranking["failed_candidates"])
            + ", so they carry no scores here."
        )
    return "\n".join(lines) if lines else "_Nothing mechanical to qualify._"


_BUILDERS = {
    "profile": _block_profile,
    "preprocessing": _block_preprocessing,
    "methods": _block_methods,
    "hyperparameters": _block_hyperparameters,
    "figures": _block_figures,
    "metrics": _block_metrics,
    "ranking": _block_ranking,
    "limitations": _block_limitations,
}


def build_blocks(run: RunDir) -> dict[str, str]:
    """A body for every declared block. Never raises for an artefact that is absent."""
    return {block_id: _BUILDERS[block_id](run) for block_id in BLOCK_IDS}


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
)
"""The skeleton, matching `skills/write-report/SKILL.md` heading for heading.

Fixed so that the two generated reports can be read side by side. Section 8 carries no
block id, which is the whole shape of the split.
"""

_PROMPT = "_Yours to write. Delete this line._"


def assemble(run: RunDir) -> str:
    """The whole document: the nine headings, the eight blocks, and room to write."""
    bodies = build_blocks(run)
    parts = [f"# Dimension reduction report — {run.id}", ""]
    for heading, block_id in SECTIONS:
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
