"""The agent's instructions, checked against the toolbox they drive.

The five skills and `/analyze` are prose the agent reads at runtime, so a command or
flag that has been renamed since they were written is a silent instruction to do
something impossible. Prose cannot be type-checked, but the names in it can.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from drtools.cli import _build_parser

ROOT = Path(__file__).resolve().parents[1]

# The product's own skills, not the build configuration's. `.claude/` holds
# instructions to whoever is *building* dr-agent; these are what dr-agent *is*, and
# they ship from the plugin root so that installing the plugin delivers them.
SKILLS_DIR = ROOT / "skills"
COMMANDS_DIR = ROOT / "commands"
PLUGIN_MANIFEST = ROOT / ".claude-plugin" / "plugin.json"

EXPECTED_SKILLS = {
    "profile-dataset",
    "plan-analysis",
    "execute-plan",
    "evaluate-embeddings",
    "write-report",
}


def _subparsers() -> dict:
    return _build_parser()._subparsers._group_actions[0].choices


def _agent_documents() -> list[Path]:
    return sorted(SKILLS_DIR.rglob("SKILL.md")) + sorted(COMMANDS_DIR.glob("*.md"))


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "document has no YAML frontmatter"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def test_the_chain_has_all_five_skills() -> None:
    """Section 1 names five, and `/analyze` invokes five."""
    assert {d.parent.name for d in SKILLS_DIR.rglob("SKILL.md")} == EXPECTED_SKILLS


def test_the_plugin_manifest_ships_the_skills_and_the_command() -> None:
    """Installing the plugin is the delivery route, so the manifest has to be valid.

    Skills and commands are discovered by convention at the plugin root, so what this
    checks is that the root is a plugin root at all: a manifest Claude Code will parse,
    carrying the `name` it refuses to load without.
    """
    manifest = json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["name"] == "dr-agent"
    assert manifest["version"] == _project_version()
    assert (SKILLS_DIR).is_dir() and (COMMANDS_DIR).is_dir()


def _project_version() -> str:
    """One version, declared in pyproject.toml and repeated nowhere."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml declares no version"
    return match.group(1)


@pytest.mark.parametrize("document", _agent_documents(), ids=lambda p: p.parent.name)
def test_every_command_the_prose_names_exists(document: Path) -> None:
    """A renamed subcommand turns an instruction into a dead end.

    The agent cannot discover that `drtools reconnoitre` is not a command until it runs
    one and reads the usage error, by which point the skill has already sent it there.
    """
    subs = _subparsers()
    text = document.read_text(encoding="utf-8")

    named = {m.group(1) for m in re.finditer(r"`?drtools ([a-z][a-z-]+)", text)}
    unknown = sorted(named - set(subs))
    assert not unknown, f"{document.parent.name} names non-existent command(s): {unknown}"


@pytest.mark.parametrize("document", _agent_documents(), ids=lambda p: p.parent.name)
def test_every_flag_the_prose_names_exists_on_that_command(document: Path) -> None:
    subs = _subparsers()
    text = document.read_text(encoding="utf-8")

    problems = []
    # Scanned per line, so the argument span ends at the line or the closing
    # backtick. Matching only contiguous " --flag" groups stopped at the first flag
    # that took a value, leaving every later flag unchecked -- found by injecting a
    # bogus flag and watching this test pass anyway.
    for line in text.splitlines():
        for match in re.finditer(r"`?drtools ([a-z][a-z-]+)([^`]*)", line):
            command, tail = match.group(1), match.group(2)
            if command not in subs:
                continue  # reported by the command test
            available = {
                option
                for action in subs[command]._actions
                for option in action.option_strings
            }
            for flag in re.findall(r"--[a-z-]+", tail):
                if flag not in available:
                    problems.append(f"drtools {command} has no {flag}")

    assert not problems, f"{document.parent.name}: {sorted(set(problems))}"


@pytest.mark.parametrize("document", _agent_documents(), ids=lambda p: p.parent.name)
def test_frontmatter_is_present_and_addressed_to_a_reader_deciding_whether_to_read(
    document: Path,
) -> None:
    """`description` decides whether the skill is reached at all.

    It states when to use the skill and nothing about how it works: a description that
    summarises the workflow becomes a shortcut the agent takes instead of reading the
    body.
    """
    fields = _frontmatter(document.read_text(encoding="utf-8"))

    if document.name == "SKILL.md":
        assert fields.get("name") == document.parent.name
        assert fields["description"].startswith("Use when")
    else:
        assert fields.get("description")

    assert len(fields["description"]) <= 1024
