"""Rendering, and the things that stop it.

pandoc and a PDF engine are both installed in this environment, so their absence is
simulated rather than met. The staleness refusal is the one that matters: it is what
makes rendering last a property of the toolbox rather than an instruction in a skill.
"""

from __future__ import annotations

import shutil

from drtools.report import replace_block


def test_render_produces_a_pdf(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)

    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    pdf = finished_run.path / "report.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1000


def test_render_refuses_a_document_the_run_has_moved_past(cli, finished_run):
    """Two things go stale, and rendering last fixes only one of them.

    The PDF against the Markdown is fixed by ordering. The Markdown against the Run is
    not, and it is the dangerous one: it produces a clean, current-looking PDF of
    numbers the Run has moved past.
    """
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    path.write_text(
        replace_block(path.read_text(encoding="utf-8"), "ranking", "an older ranking"),
        encoding="utf-8",
    )

    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "ranking" in result.stderr
    assert "--refresh" in result.stderr
    assert not (finished_run.path / "report.pdf").exists()


def test_refreshing_makes_the_same_render_succeed(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    path.write_text(
        replace_block(path.read_text(encoding="utf-8"), "ranking", "an older ranking"),
        encoding="utf-8",
    )
    assert cli("render", "--run-dir", finished_run.path).code == 2

    assert cli("report", "--refresh", "--run-dir", finished_run.path).code == 0

    assert cli("render", "--run-dir", finished_run.path).code == 0


def test_render_refuses_a_missing_pandoc_by_name(cli, finished_run, monkeypatch):
    cli("report", "--run-dir", finished_run.path)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "pandoc" in result.stderr


def test_render_refuses_a_missing_pdf_engine_separately(cli, finished_run, monkeypatch):
    """Two installs, two fixes. One message for both sends the agent to the wrong one."""
    cli("report", "--run-dir", finished_run.path)
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None
    )

    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "pdflatex" in result.stderr
    assert "not on PATH" not in result.stderr, "that is the other refusal"


def test_render_refuses_when_there_is_no_report(cli, finished_run):
    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "drtools report" in result.stderr
