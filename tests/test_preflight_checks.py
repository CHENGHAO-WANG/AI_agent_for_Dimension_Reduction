"""The conduct checks, validated against runs that really broke each rule.

A check that stays green on the failure it exists for checks nothing. So each has a
recorded transcript from days 20 and 21 it must fail on, and one it must pass on. These
run in the default suite: they read fixtures and call no agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preflight.checks import background, chained_slow, foreign_reads, stray_writes
from preflight.transcript import Transcript, Call, read

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
REPO = Path(__file__).resolve().parents[1]
RUN_IDS = {"day20": "day20-pbmc3k", "day20b": "day20-pbmc3k-b", "day21": "day21-pathmnist",
           "day21b": "day21-pathmnist-b", "day21b2": "day21-pathmnist-b"}


def _t(name: str) -> Transcript:
    return read(FIXTURES / f"{name}.jsonl")


def test_background_fails_on_the_run_that_backgrounded_an_embed():
    assert background(_t("day21"))
    assert not background(_t("day21b2"))


def test_chained_slow_fails_on_the_run_that_chained_three_evaluates():
    assert chained_slow(_t("day21b"))
    assert not chained_slow(_t("day21b2"))


def test_foreign_reads_fails_on_the_run_that_read_the_earlier_plan():
    assert foreign_reads(_t("day21b"), RUN_IDS["day21b"], REPO)
    assert not foreign_reads(_t("day21b2"), RUN_IDS["day21b2"], REPO)


def test_stray_writes_fails_on_the_run_that_wrote_into_the_working_directory():
    assert stray_writes(_t("day20"), RUN_IDS["day20"])
    assert not stray_writes(_t("day20b"), RUN_IDS["day20b"])


def test_a_truncated_transcript_still_yields_its_calls(tmp_path):
    good = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "drtools status"}}]}}
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(good) + "\n" + '{"type": "assist', encoding="utf-8")
    assert [c.name for c in read(path).calls] == ["Bash"]


def _with(*calls: Call) -> Transcript:
    return Transcript(calls=list(calls))


@pytest.mark.parametrize("path", [
    "D:\\PythonProject\\AI_agent_for_Dimension_Reduction\\tests\\plans.py",
    "d:/pythonproject/ai_agent_for_dimension_reduction/drtools/cli.py",
    "/d/PythonProject/AI_agent_for_Dimension_Reduction/design/notes.md",
])
def test_a_read_of_the_repository_is_caught_in_every_path_form(path):
    assert foreign_reads(_with(Call("Read", {"file_path": path})), "r1", REPO)


def test_the_own_run_is_not_mistaken_for_another_it_extends():
    own = _with(Call("Read", {"file_path": "D:\\e2e\\runs\\day21-pathmnist-b\\plan.json"}))
    assert not foreign_reads(own, "day21-pathmnist-b", REPO)
    other = _with(Call("Read", {"file_path": "D:\\e2e\\runs\\day21-pathmnist\\plan.json"}))
    assert foreign_reads(other, "day21-pathmnist-b", REPO)


def test_listing_runs_to_find_one_to_resume_is_allowed():
    assert not foreign_reads(_with(Call("Bash", {"command": "ls runs/ 2>/dev/null"})), "r1", REPO)


def test_a_loop_over_a_slow_command_counts_as_chaining():
    loop = Call("Bash", {"command": "for c in a b; do drtools evaluate --id $c; done"})
    assert chained_slow(_with(loop))
    assert not chained_slow(_with(Call("Bash", {"command": "drtools checkpoint && drtools status"})))


from preflight.checks import finished, representation, resume, visualization


def test_finished_names_the_stage_a_bare_run_is_at(tmp_path):
    from drtools.cli import main
    assert main(["profile", "--data", "blobs", "--runs-root", str(tmp_path), "--run-id", "r"]) == 0
    problems = finished(tmp_path / "r")
    assert any("not done" in p for p in problems)
    assert any("report.pdf" in p for p in problems)


def test_representation_reads_the_decision_and_the_plan(finished_run):
    # blobs is not counts and selects nothing, so both checks must speak.
    problems = representation(finished_run.path, {}, [])
    assert any("raw_counts" in p for p in problems)
    assert any("Selected baseline" in p for p in problems)


def test_visualization_refuses_a_ranked_run(finished_run):
    assert any("ranking" in p for p in visualization(finished_run.path, {}, []))


def test_resume_says_it_tested_nothing_when_the_first_launch_finished(finished_run):
    summary = {"launches": [{"next": "done"}, {"next": "done"}]}
    assert any("tested nothing" in p for p in resume(finished_run.path, summary, [Transcript(), Transcript()]))


# ------------------------------------------------ fixes from the day 23 Codex review


def test_a_wildcard_read_across_runs_is_caught():
    assert foreign_reads(_with(Call("Glob", {"pattern": "runs/*/plan.json"})), "r1", REPO)
    assert foreign_reads(_with(Call("Bash", {"command": "cat runs/*/plan.json"})), "r1", REPO)
    assert not foreign_reads(_with(Call("Bash", {"command": "ls runs/"})), "r1", REPO)


def test_only_the_fields_that_name_a_path_are_read():
    search = Call("Grep", {"path": "D:/e/runs/r1/results/report.md", "pattern": "runs/other"})
    described = Call("Bash", {"command": "drtools status --run-dir runs/r1",
                              "description": "compare with runs/other"})
    assert not foreign_reads(_with(search, described), "r1", REPO)


def test_the_word_for_is_not_a_loop():
    single = Call("Bash", {"command": 'echo "waiting for it"; drtools evaluate --run-dir D:/for/runs/r1 --id a'})
    assert not chained_slow(_with(single))


def test_a_damaged_utf8_tail_still_yields_the_calls_before_it(tmp_path):
    good = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "a", "name": "Bash", "input": {"command": "drtools status"}}]}}
    path = tmp_path / "t.jsonl"
    path.write_bytes((json.dumps(good) + "\n").encode() + b'{"type": "\xe4\xb8')
    assert [c.name for c in read(path).calls] == ["Bash"]


def test_resume_refuses_a_stage_run_before_status_in_the_same_call(finished_run):
    first = Transcript()
    second = _with(Call("Bash", {"command": "drtools embed --run-dir runs/r --id a; drtools status --run-dir runs/r"}))
    summary = {"launches": [{"next": "execute"}, {"next": "done"}]}
    assert any("before reading status" in p for p in resume(finished_run.path, summary, [first, second]))
