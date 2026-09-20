"""Driving the CLI in-process, so the seams between commands are testable.

Every command is reached through `main(argv)` rather than a subprocess: the exit
code, the JSON on stdout and the message on stderr are what the agent sees, so they
are what the tests assert on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drtools.cli import main


@dataclass(frozen=True)
class CliResult:
    code: int
    payload: dict | None
    stderr: str


@pytest.fixture
def cli(capsys):
    def run(*argv) -> CliResult:
        code = main([str(a) for a in argv])
        captured = capsys.readouterr()
        payload = json.loads(captured.out) if captured.out.strip() else None
        return CliResult(code, payload, captured.err)

    return run


@pytest.fixture
def csv_dataset(tmp_path):
    def make(rows: int = 60, cols: int = 8, seed: int = 0, labels: bool = False) -> Path:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            rng.normal(size=(rows, cols)), columns=[f"f{i}" for i in range(cols)]
        )
        if labels:
            frame["label"] = [f"c{i % 3}" for i in range(rows)]
        path = tmp_path / f"data-{rows}x{cols}-{seed}-{int(labels)}.csv"
        frame.to_csv(path, index=False)
        return path

    return make


@pytest.fixture(autouse=True)
def allow_in_process(monkeypatch):
    """The suite drives candidates in-process for speed; the agent may not.

    Autouse rather than opt-in, because the flag is how nearly every CLI test avoids
    spawning a subprocess per candidate. A test that wants to see the gate refuse
    deletes the variable itself.
    """
    monkeypatch.setenv("DRTOOLS_ALLOW_IN_PROCESS", "1")


PCA_STAGES = [{"op": "pca", "params": {"n_components": 2}}]


def plan_document(**overrides):
    """A small, valid plan over the blobs fixture: one candidate and one rejection."""
    plan = {
        "dataset": "blobs",
        "budget": "fast",
        "base_preprocessing": [{"op": "standardise", "params": {}}],
        "candidates": [
            {
                "id": "pca-2",
                "stages": PCA_STAGES,
                "rationale": "linear baseline, always included",
            }
        ],
        "rejected": [
            {
                "method": "mds",
                "reason": "O(n^2) at this sample count and PCA recovers the same structure",
                "evidence": ["profile.shape.n_samples"],
            }
        ],
        "evaluation": {
            "weights": {"trustworthiness": 0.6, "continuity": 0.4},
            "justification": "neighbourhood faithfulness is the question here",
        },
    }
    plan.update(overrides)
    return plan


@pytest.fixture(scope="session")
def finished_run(tmp_path_factory):
    """One complete run: profile, recon, plan, embed, evaluate, rank, figures.

    Session-scoped because every report test reads it and building it twice doubles
    the suite's runtime. The in-process gate is set here rather than left to the
    autouse `allow_in_process` fixture: that one is function-scoped, and pytest builds
    higher-scoped fixtures first, so it has not run yet when this one embeds.
    """
    from drtools.cli import main
    from drtools.runs import RunDir

    root = tmp_path_factory.mktemp("finished")
    run_dir = root / "r1"
    before = os.environ.get("DRTOOLS_ALLOW_IN_PROCESS")
    os.environ["DRTOOLS_ALLOW_IN_PROCESS"] = "1"
    try:
        assert main(["profile", "--data", "blobs", "--runs-root", str(root),
                     "--run-id", "r1"]) == 0
        assert main(["recon", "--run-dir", str(run_dir)]) == 0
        (run_dir / "plan.json").write_text(json.dumps(plan_document()), encoding="utf-8")
        assert main(["validate-plan", "--run-dir", str(run_dir)]) == 0
        assert main(["prepare-reference", "--run-dir", str(run_dir)]) == 0
        assert main(["embed", "--run-dir", str(run_dir), "--id", "pca-2",
                     "--in-process"]) == 0
        assert main(["evaluate", "--run-dir", str(run_dir), "--id", "pca-2"]) == 0
        assert main(["rank", "--run-dir", str(run_dir)]) == 0
        assert main(["figures", "--run-dir", str(run_dir)]) == 0
    finally:
        if before is None:
            os.environ.pop("DRTOOLS_ALLOW_IN_PROCESS", None)
        else:
            os.environ["DRTOOLS_ALLOW_IN_PROCESS"] = before
    return RunDir(run_dir)


@pytest.fixture(autouse=True)
def _clear_report(request):
    """A session-scoped run must look untouched to every test that uses it.

    `finished_run` is shared, and several tests write report.md or report.pdf into it.
    Without this the second such test sees the first one's document and the refusals
    under test never fire.
    """
    yield
    if "finished_run" in request.fixturenames:
        run = request.getfixturevalue("finished_run")
        for name in ("report.md", "report.pdf"):
            (run.path / name).unlink(missing_ok=True)
