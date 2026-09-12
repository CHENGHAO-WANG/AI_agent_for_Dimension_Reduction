"""JSON that survives numpy.

Every artefact the agent reads is JSON, and numpy scalars are not JSON-serialisable,
so a single conversion point keeps `json.dump` from failing three stages into a long
run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Convert numpy types to their Python equivalents, recursively."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        # NaN and infinity are valid Python but not valid JSON.
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def dumps(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(jsonable(value), indent=indent, ensure_ascii=False)


def write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value) + "\n", encoding="utf-8")
    return path


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[Any]:
    """Read a JSON Lines file, returning an empty list when it does not exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_line(path: Path, value: Any) -> None:
    """Append one record to a JSON Lines file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False) + "\n")
