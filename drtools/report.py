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
    body that happens to contain the word `drtools` -- a path in a metrics note, an
    instruction quoted from a refusal -- does not end the block early.
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
