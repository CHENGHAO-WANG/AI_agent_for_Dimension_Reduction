"""The fenced regions the toolbox owns inside a document the agent also writes.

The report has two owners. Every number belongs to the toolbox and sits inside a fence;
everything else is the agent's prose. These tests pin the mechanics that keep the two
apart: finding a block after the prose around it has grown, and telling a block the
toolbox wrote from one the agent has edited.
"""

from __future__ import annotations

import json

import pytest

from drtools.cli import main
from drtools.report import (
    BLOCK_IDS,
    NOT_PRODUCED,
    build_blocks,
    edited,
    fence,
    parse_blocks,
    replace_block,
)
from drtools.runs import RunDir


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


# ------------------------------------- the block contents, against a real finished run


def test_every_declared_block_is_generated(finished_run):
    assert set(build_blocks(finished_run)) == set(BLOCK_IDS)


def test_there_is_no_interpretation_block(finished_run):
    """Section 8 is the agent's alone; a generated block there would imply derivation."""
    assert "interpretation" not in build_blocks(finished_run)


def test_the_profile_block_prints_the_shape_the_profile_recorded(finished_run):
    profile = json.loads((finished_run.path / "profile.json").read_text(encoding="utf-8"))

    body = build_blocks(finished_run)["profile"]

    assert str(profile["shape"]["n_samples"]) in body
    assert str(profile["shape"]["n_features"]) in body
    assert profile["dataset_digest"][:12] in body


def test_the_ranking_block_carries_the_weighting_and_every_note(finished_run):
    ranking = json.loads((finished_run.path / "ranking.json").read_text(encoding="utf-8"))

    body = build_blocks(finished_run)["ranking"]

    assert ranking["winner"] in body
    for note in ranking["notes"]:
        assert note in body, "a qualification rank produced must reach the report"


def test_the_metrics_block_prints_the_values_metrics_recorded(finished_run):
    metrics = json.loads(
        (finished_run.path / "metrics" / "pca-2.json").read_text(encoding="utf-8")
    )

    body = build_blocks(finished_run)["metrics"]

    assert f"{metrics['values']['trustworthiness']:.4f}" in body


def test_the_hyperparameter_block_separates_specified_from_default(finished_run):
    """param_provenance is the only record of which values the agent actually chose."""
    body = build_blocks(finished_run)["hyperparameters"]

    assert "n_components" in body
    assert "specified" in body


def test_the_methods_block_carries_the_rejections_with_their_reasons(finished_run):
    """Section 3 is the highest-value part of the document, by the design's own account."""
    plan = json.loads(
        (finished_run.path / "plan.registered.json").read_text(encoding="utf-8")
    )
    rejection = plan["rejected"][0]

    body = build_blocks(finished_run)["methods"]

    assert rejection["method"] in body
    assert rejection["reason"] in body
    assert rejection["evidence"][0] in body


def test_figure_paths_are_relative_to_the_run(finished_run):
    """figures.json stores absolute paths; a report holding one breaks when moved."""
    body = build_blocks(finished_run)["figures"]

    assert "figures/comparison.png" in body.replace("\\", "/")
    assert str(finished_run.path) not in body


def test_a_block_whose_source_is_absent_says_so(tmp_path):
    """Absent must not read as a section the agent has not reached yet."""
    root = tmp_path / "runs"
    assert main(["profile", "--data", "blobs", "--runs-root", str(root),
                 "--run-id", "bare"]) == 0

    blocks = build_blocks(RunDir(root / "bare"))

    assert blocks["ranking"] == NOT_PRODUCED
    assert blocks["figures"] == NOT_PRODUCED
    assert blocks["methods"] == NOT_PRODUCED


# ------------------------------------------------------- emitting the document itself


def test_report_writes_the_nine_sections_with_their_blocks(cli, finished_run):
    result = cli("report", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    document = (finished_run.path / "report.md").read_text(encoding="utf-8")
    for heading in ("## 1. Dataset profile", "## 8. Interpretation", "## 9. Limitations"):
        assert heading in document
    assert set(parse_blocks(document)) == set(BLOCK_IDS)


def test_report_refuses_a_run_that_is_not_ready(cli, tmp_path):
    """status already knows; report asks it rather than asking the question again."""
    root = tmp_path / "runs"
    cli("profile", "--data", "blobs", "--runs-root", root, "--run-id", "early")

    result = cli("report", "--run-dir", root / "early")

    assert result.code == 2
    assert "report" in result.stderr
    assert not (root / "early" / "report.md").exists()


def test_report_refuses_to_overwrite_and_names_refresh(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    (finished_run.path / "report.md").write_text("the agent's prose\n", encoding="utf-8")

    result = cli("report", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "--refresh" in result.stderr
    assert (
        finished_run.path / "report.md"
    ).read_text(encoding="utf-8") == "the agent's prose\n"


# ----------------------------------------------- refreshing without destroying prose


def test_refresh_updates_a_block_and_leaves_the_prose_alone(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    written = "The winner is clear on the neighbourhood metrics."
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "_Yours to write. Delete this line._", written, 1
        ),
        encoding="utf-8",
    )

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    after = path.read_text(encoding="utf-8")
    assert written in after
    assert set(parse_blocks(after)) == set(BLOCK_IDS)


def test_refresh_rewrites_a_block_the_run_has_moved_past(cli, finished_run):
    """The case the whole mechanism exists for: a number changed under the document."""
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    current = parse_blocks(path.read_text(encoding="utf-8"))["ranking"].body
    path.write_text(
        replace_block(path.read_text(encoding="utf-8"), "ranking", "an older ranking"),
        encoding="utf-8",
    )

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    assert "ranking" in result.payload["refreshed"]
    assert parse_blocks(path.read_text(encoding="utf-8"))["ranking"].body == current


def test_refresh_refuses_a_block_the_agent_edited(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    document = path.read_text(encoding="utf-8")
    block = parse_blocks(document)["ranking"]
    tampered = (
        document[: block.start]
        + document[block.start : block.end].replace("| Rank |", "| Rank (mine) |", 1)
        + document[block.end :]
    )
    path.write_text(tampered, encoding="utf-8")

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "ranking" in result.stderr
    assert path.read_text(encoding="utf-8") == tampered, "a refusal writes nothing"


def test_refresh_reports_a_deleted_block_rather_than_reinserting_it(cli, finished_run):
    """The toolbox cannot know where in the prose a deleted fence belonged."""
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    document = path.read_text(encoding="utf-8")
    block = parse_blocks(document)["figures"]
    path.write_text(document[: block.start] + document[block.end :], encoding="utf-8")

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    assert result.payload["missing"] == ["figures"]
    assert "figures" not in parse_blocks(path.read_text(encoding="utf-8"))


def test_refresh_refuses_when_there_is_no_report_yet(cli, finished_run):
    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "drtools report" in result.stderr


def test_a_body_that_ends_in_a_blank_line_still_matches_its_own_digest():
    """The closing comment sits on its own line, so a read always strips what precedes it.

    A generator that ends its body with a blank line -- `_block_ranking` does whenever
    a Run produced no ranking notes -- would otherwise write a block that reads as
    hand-edited the instant it is parsed back, and `--refresh` would refuse a block
    nobody had touched.
    """
    document = fence("ranking", "a table\n\n")

    assert not edited(parse_blocks(document)["ranking"])


def test_an_unchanged_report_refreshes_nothing(cli, finished_run):
    """The comparison must be against a body in the same form the document holds.

    Otherwise a block whose generator ends on a blank line reads as changed on every
    refresh, and `refreshed` stops meaning anything the agent can act on.
    """
    cli("report", "--run-dir", finished_run.path)

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    assert result.payload["refreshed"] == []
    assert set(result.payload["unchanged"]) == set(BLOCK_IDS)
