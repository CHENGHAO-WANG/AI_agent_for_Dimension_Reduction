"""The pre-flight: its launch command, and the checks applied to a real pre-flight.

The scenario test reads the folder `DRAGENT_PREFLIGHT` names, written by
`python -m preflight.launch`, and is skipped when it is unset, so the default suite
calls no agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from preflight.checks import (SCENARIO_CHECKS, background, chained_slow, finished,
                              foreign_reads, stray_writes)
from preflight.launch import ALLOWED, REPO, build_command, plugin_loaded
from preflight.transcript import Transcript, read

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"


def test_the_command_carries_day_24s_configuration_unchanged():
    cmd = build_command("claude", "/dr-agent:analyze wide_counts --auto --run-id r", None)
    assert cmd[1:3] == ["-p", "/dr-agent:analyze wide_counts --auto --run-id r"]
    assert cmd[cmd.index("--plugin-dir") + 1] == REPO.as_posix()
    assert all(tool in cmd for tool in ALLOWED)
    assert cmd[cmd.index("--disallowedTools") + 1] == "PowerShell"
    assert "--max-turns" not in cmd
    assert build_command("claude", "x", 25)[-2:] == ["--max-turns", "25"]


def test_a_transcript_without_the_plugin_is_recognised():
    assert plugin_loaded(read(FIXTURES / "day21b2.jsonl"))
    assert not plugin_loaded(Transcript(init={"slash_commands": ["analyze"]}))


PREFLIGHT = os.environ.get("DRAGENT_PREFLIGHT")


@pytest.mark.parametrize("scenario", sorted(SCENARIO_CHECKS))
def test_scenario(scenario):
    if not PREFLIGHT:
        pytest.skip("DRAGENT_PREFLIGHT is not set")
    folders = sorted(Path(PREFLIGHT).glob(f"{scenario}-*"))
    if not folders:
        pytest.skip(f"no {scenario} folder under {PREFLIGHT}")
    folder = folders[-1]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    run_dir = folder / "runs" / summary["run_id"]
    transcripts = [read(folder / launch["transcript"]) for launch in summary["launches"]]
    problems = []
    for t in transcripts:
        problems += background(t) + chained_slow(t)
        problems += foreign_reads(t, summary["run_id"], REPO) + stray_writes(t, summary["run_id"])
    problems += finished(run_dir) + SCENARIO_CHECKS[scenario](run_dir, summary, transcripts)
    assert not problems, "\n".join(problems)
