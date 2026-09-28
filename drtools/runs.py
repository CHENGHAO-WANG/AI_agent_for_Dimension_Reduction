"""The run directory — the only place state lives.

Nothing the agent learns is carried in conversation context. Each stage reads the
previous stage's artefact off disk and writes its own, so a run survives a lost
context, can be resumed after a crash, and can be opened and checked by someone who
was not there when it happened.

`decisions.jsonl` is the spine of that. Every choice point appends one record naming
what was decided, why, and which measurements the reasoning rests on. The final report
is generated from this log, so a rationale that cites no evidence is visibly a
rationale that cites no evidence.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from drtools import jsonio

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "sklearn",
    "umap",
    "openTSNE",
    "torch",
    "scanpy",
    "matplotlib",
)

# Import name -> distribution name, where the two differ.
DISTRIBUTION_NAMES = {"sklearn": "scikit-learn", "umap": "umap-learn"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "dataset"


class RunDir:
    """A single analysis run, addressed by its directory."""

    def __init__(self, path: Path, *, create: bool = True) -> None:
        """Open a run, creating its subdirectories unless asked not to.

        Creating on construction is right for a command about to write one. It is
        wrong for a read-only command: pointing `status` at a directory that is not a
        run would otherwise scatter empty `embeddings/`, `metrics/` and `figures/`
        into it, so a command that only reports would quietly modify what it reported
        on.
        """
        self.path = Path(path)
        if not create:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        for child in ("embeddings", "metrics", "figures"):
            (self.path / child).mkdir(exist_ok=True)

    @classmethod
    def create(cls, root: Path, dataset: str, run_id: str | None = None) -> RunDir:
        """Open or create a run directory, named for the dataset and the time."""
        if run_id is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = f"{stamp}-{_slug(dataset)}"
        return cls(Path(root) / run_id)

    # ------------------------------------------------------------------ artefacts

    @property
    def id(self) -> str:
        return self.path.name

    @property
    def profile_path(self) -> Path:
        return self.path / "profile.json"

    @property
    def recon_path(self) -> Path:
        return self.path / "recon.json"

    @property
    def plan_path(self) -> Path:
        return self.path / "plan.json"

    @property
    def suggestions_dir(self) -> Path:
        """Every suggestion `suggest-params` and `suggest-base` returned, one file each.

        Persisted so that provenance can say whether a registered value followed the
        suggestion or overrode it; returned only to the agent, it could not.
        """
        return self.path / "suggestions"

    @property
    def manifest_path(self) -> Path:
        return self.path / "run.json"

    @property
    def decisions_path(self) -> Path:
        return self.path / "decisions.jsonl"

    def write_artifact(self, name: str, value: Any) -> Path:
        return jsonio.write(self.path / name, value)

    def read_artifact(self, name: str) -> Any:
        path = self.path / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist; the stage that produces it has not run yet"
            )
        return jsonio.read(path)

    # ------------------------------------------------------------------- manifest

    def write_manifest(self, **extra: Any) -> Path:
        """Record everything needed to reproduce this run, or to explain why it differs.

        Rewriting rather than replacing: fields another stage recorded on the manifest
        are carried over instead of being dropped. `profile` calls this every time it
        runs, and a rebuild-from-scratch quietly deleted the `plan_digest` that
        `validate-plan` had written, so an ordinary `profile → validate-plan → profile`
        sequence left the run wedged. `created` is likewise the moment the run was
        created, not the moment it was last profiled.
        """
        previous = (
            jsonio.read(self.manifest_path) if self.manifest_path.exists() else {}
        )
        manifest = {
            **previous,
            "run_id": self.id,
            "created": previous.get("created") or _timestamp(),
            "command": " ".join(sys.argv),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
            **extra,
        }
        return jsonio.write(self.manifest_path, manifest)

    def update_manifest(self, **fields: Any) -> Path:
        manifest = (
            jsonio.read(self.manifest_path) if self.manifest_path.exists() else {}
        )
        manifest.update(fields)
        manifest["updated"] = _timestamp()
        return jsonio.write(self.manifest_path, manifest)

    # -------------------------------------------------------------- decision log

    def log_decision(
        self,
        stage: str,
        question: str,
        chosen: str,
        rationale: str,
        *,
        evidence: list[str] | None = None,
        options_considered: list[str] | None = None,
        actor: str = "agent",
        **extra: Any,
    ) -> None:
        """Append one decision. `evidence` holds dotted paths into the artefacts.

        An empty `evidence` list is allowed but is a signal in itself: the report
        renders such decisions as unsupported, which is the intended pressure.
        """
        jsonio.append_line(
            self.decisions_path,
            {
                "timestamp": _timestamp(),
                "stage": stage,
                "question": question,
                "options_considered": options_considered or [],
                "chosen": chosen,
                "rationale": rationale,
                "evidence": evidence or [],
                "actor": actor,
                **extra,
            },
        )

    def decisions(self) -> list[dict[str, Any]]:
        return jsonio.read_lines(self.decisions_path)


class _Missing:
    """Sentinel for an evidence key that does not exist in the artefacts."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


def resolve_evidence(paths: list[str], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Look up dotted evidence keys against loaded artefacts.

    Used by the report writer to check that every cited key exists and to quote its
    value. Absent keys resolve to `MISSING` rather than to None, because a measurement
    that is legitimately null — an undefined ratio, an unestimable dimension — is a
    real finding, while a citation pointing at nothing is a broken rationale. Collapsing
    the two would hide exactly the failure this mechanism exists to catch.
    """
    resolved: dict[str, Any] = {}
    for path in paths:
        head, _, tail = path.partition(".")
        node: Any = artifacts.get(head, MISSING)
        for part in filter(None, tail.split(".")):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif isinstance(node, list) and part.lstrip("-").isdigit():
                index = int(part)
                node = node[index] if -len(node) <= index < len(node) else MISSING
            else:
                node = MISSING
            if node is MISSING:
                break
        resolved[path] = node
    return resolved


def unresolved_message(unresolved: list[str], artifacts: dict[str, Any]) -> str:
    """Name the broken key and the keys that do exist beside it.

    An agent told only that something failed will guess again; told what is there, it
    corrects.
    """
    lines = []
    for key in unresolved:
        parent, _, _ = key.rpartition(".")
        if parent:
            neighbour = resolve_evidence([parent], artifacts)[parent]
        else:
            neighbour = artifacts
        if isinstance(neighbour, dict) and neighbour:
            available = ", ".join(sorted(str(k) for k in neighbour))
            # ASCII only. A console on a legacy codepage renders U+2014 as a literal
            # "?", and this message exists to be read and acted on by the agent.
            lines.append(f"  {key} -- {parent or 'the run'} holds: {available}")
        else:
            roots = ", ".join(sorted(artifacts)) or "nothing yet"
            lines.append(f"  {key} -- no such path. This run holds: {roots}")
    return (
        "these evidence keys do not resolve against this run's artefacts:\n"
        + "\n".join(lines)
        + "\nCite a key that exists, or drop it. An empty evidence list is allowed and "
        "renders as unsupported, but a citation pointing at nothing is a broken "
        "rationale rather than a missing measurement."
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    """Distribution versions, via metadata rather than `__version__`.

    Several of these packages now warn on attribute access, and metadata is the
    authoritative answer anyway — it is what a grader would reinstall from.
    """
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        distribution = DISTRIBUTION_NAMES.get(name, name)
        try:
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return versions
