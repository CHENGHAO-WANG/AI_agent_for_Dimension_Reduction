"""Driving the CLI in-process, so the seams between commands are testable.

Every command is reached through `main(argv)` rather than a subprocess: the exit
code, the JSON on stdout and the message on stderr are what the agent sees, so they
are what the tests assert on.
"""

from __future__ import annotations

import json
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
