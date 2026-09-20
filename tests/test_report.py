"""The fenced regions the toolbox owns inside a document the agent also writes.

The report has two owners. Every number belongs to the toolbox and sits inside a fence;
everything else is the agent's prose. These tests pin the mechanics that keep the two
apart: finding a block after the prose around it has grown, and telling a block the
toolbox wrote from one the agent has edited.
"""

from __future__ import annotations

import pytest

from drtools.report import (
    BLOCK_IDS,
    edited,
    fence,
    parse_blocks,
    replace_block,
)


def test_a_fence_round_trips_through_the_parser():
    document = f"Some prose.\n\n{fence('ranking', '| a | b |')}\n\nMore prose.\n"

    blocks = parse_blocks(document)

    assert set(blocks) == {"ranking"}
    assert blocks["ranking"].body == "| a | b |"
    assert not edited(blocks["ranking"])


def test_replacing_a_block_preserves_the_prose_around_it_byte_for_byte():
    before = "Prose above, with a number 0.52 the agent wrote.\n\n"
    after = "\n\nProse below.\n"
    document = before + fence("ranking", "old body") + after

    updated = replace_block(document, "ranking", "new body")

    assert updated.startswith(before)
    assert updated.endswith(after)
    assert parse_blocks(updated)["ranking"].body == "new body"


def test_a_hand_edited_block_is_detected():
    document = fence("ranking", "0.52")
    tampered = document.replace("0.52", "0.61")

    assert edited(parse_blocks(tampered)["ranking"])


def test_an_untouched_block_is_not_reported_as_edited():
    """The digest must be over exactly what `fence` writes, or every block reads edited."""
    document = "\n\n".join(fence(block_id, f"body of {block_id}") for block_id in BLOCK_IDS)

    blocks = parse_blocks(document)

    assert set(blocks) == set(BLOCK_IDS)
    assert not any(edited(block) for block in blocks.values())


def test_replacing_a_block_that_is_not_there_raises():
    with pytest.raises(KeyError):
        replace_block("no fences here", "ranking", "body")


def test_a_body_containing_a_fence_like_line_does_not_end_the_block():
    """A metrics note could legitimately contain the word drtools in a path."""
    body = "| note | see <!-- drtools:elsewhere --> in the log |"

    blocks = parse_blocks(fence("metrics", body))

    assert blocks["metrics"].body == body
