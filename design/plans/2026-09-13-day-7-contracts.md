# Day 7 Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make three guarantees the design claims — dataset identity, pre-registration, and comparability — actually hold, refusing rather than warning when they are broken.

**Architecture:** Enforcement lives with the artefact it guards: `cache.py` owns dataset identity, `plan.py` and `validate-plan` own the Plan freeze, `rank.py` owns comparability. One new tiny module, `drtools/digest.py`, provides the shared `content_hash()` primitive. Refusal is `ContractError`, which `main()` already maps to exit code 2. Nothing under `tests/` currently imports `drtools.cli`, so Task 1 builds the harness the other nine tasks test through.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, scikit-learn 1.9.1, pydantic v2, pytest. Interpreter is `.venv/Scripts/python.exe` (Windows, Git Bash).

**Spec:** `design/specs/2026-09-13-day-7-contracts.md`

## Global Constraints

- **Refuse, never warn.** A contract violation raises `ContractError` and writes nothing. No override flag anywhere.
- **A refusal message must name its recovery route.** `cli.py` prints the message and nothing else; an agent with no stated next step loops.
- **Amendment is not implemented.** `rank`'s refusal names it as the route that does not yet exist.
- **Identity is content only** — matrix and label codes. Never the Loader source, never `name`/`source`/`label_names`.
- **Digest on write and in `ensure_cache` only.** Never in `read_cache`: it runs in every Candidate subprocess, in `figures`, and twice in `_reference_for`. Measured cost is 36 ms for a pbmc3k-shaped CSR and 2.0 s for a pathmnist-shaped dense matrix.
- **The agent never chooses the Battery's settings.** `--k` and `--max-samples` leave `evaluate`; `k=` leaves `evaluate_embedding`'s signature rather than becoming an override.
- **Naming:** `dataset_digest` and the Reference's recorded settings are provisional keys. `CONTEXT.md` lists "protocol" as a word to avoid — do not introduce it.
- Run the suite with `.venv/Scripts/python.exe -m pytest -q` (178 tests, ~45 s at the start of this plan).

---

### Task 1: CLI test harness

Every failure this plan fixes lives in `cli.py`, and nothing under `tests/` imports it. This task exists so the other nine have somewhere to hang their regressions.

**Files:**
- Create: `tests/conftest.py`
- Test: `tests/test_cli_harness.py`

**Interfaces:**
- Consumes: nothing
- Produces: pytest fixtures `cli` (callable `cli(*argv) -> CliResult`, where `CliResult` has `.code: int`, `.payload: dict | None`, `.stderr: str`) and `csv_dataset` (callable `csv_dataset(rows=60, cols=8, seed=0, labels=False) -> pathlib.Path`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_harness.py
def test_profile_runs_through_the_harness(cli, csv_dataset, tmp_path):
    result = cli(
        "profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", tmp_path / "runs", "--run-id", "r1",
    )
    assert result.code == 0, result.stderr
    assert result.payload["shape"]["n_samples"] == 60
    assert result.payload["shape"]["n_features"] == 8


def test_harness_reports_a_refusal_without_raising(cli, tmp_path):
    result = cli("evaluate", "--run-dir", tmp_path / "nope", "--id", "x")
    assert result.code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_harness.py -q`
Expected: FAIL — `fixture 'cli' not found`

- [ ] **Step 3: Write the fixtures**

```python
# tests/conftest.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_harness.py -q`
Expected: PASS, 2 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 180 passed

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_cli_harness.py
git commit -m "Add a CLI test harness, which the package has never had"
```

---

### Task 2: `content_hash` with sparse canonicalisation

**Files:**
- Create: `drtools/digest.py`
- Test: `tests/test_digest.py`

**Interfaces:**
- Consumes: nothing
- Produces: `content_hash(X: Matrix, labels: np.ndarray | None) -> str` — 64-char hex sha256

- [ ] **Step 1: Write the failing test**

```python
# tests/test_digest.py
import numpy as np
import scipy.sparse as sp

from drtools.digest import content_hash

BASE = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])


def test_equal_dense_matrices_agree():
    assert content_hash(BASE, None) == content_hash(BASE.copy(), None)


def test_different_data_differs():
    assert content_hash(BASE, None) != content_hash(BASE * 2, None)


def test_unsorted_indices_agree_with_sorted():
    sorted_ = sp.csr_matrix(BASE)
    unsorted = sp.csr_matrix(
        (np.array([2.0, 1.0, 3.0]), np.array([3, 1, 2]), np.array([0, 2, 3])),
        shape=(2, 4),
    )
    assert content_hash(unsorted, None) == content_hash(sorted_, None)


def test_explicit_zero_agrees_with_absent_zero():
    plain = sp.csr_matrix(BASE)
    padded = sp.csr_matrix(
        (np.array([1.0, 0.0, 2.0, 3.0]), np.array([1, 2, 3, 2]), np.array([0, 3, 4])),
        shape=(2, 4),
    )
    assert content_hash(padded, None) == content_hash(plain, None)


def test_index_width_does_not_change_identity():
    narrow = sp.csr_matrix(BASE)
    wide = sp.csr_matrix(BASE)
    wide.indices = wide.indices.astype(np.int64)
    wide.indptr = wide.indptr.astype(np.int64)
    assert content_hash(wide, None) == content_hash(narrow, None)


def test_labels_are_part_of_identity():
    a = np.array([0, 1])
    b = np.array([1, 0])
    assert content_hash(BASE, a) != content_hash(BASE, b)
    assert content_hash(BASE, a) != content_hash(BASE, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_digest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drtools.digest'`

- [ ] **Step 3: Write the implementation**

```python
# drtools/digest.py
"""One content digest for the matrix a run is analysing.

Identity is what the data *is*, not how it happens to be stored. Two sparse matrices
holding the same values must agree even when their indices arrived unsorted, carry an
explicitly stored zero, or are int64 rather than int32 — index width is a storage
detail, and a loader that switches to it has not produced a different dataset. So
everything is canonicalised before it is hashed, on both the write path and the check
path, which is why this lives in one function rather than at each call site.
"""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

from drtools.contract import Matrix


def content_hash(X: Matrix, labels: np.ndarray | None) -> str:
    """A stable digest of the matrix and its label codes."""
    digest = hashlib.sha256()

    if sp.issparse(X):
        matrix = X.tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        matrix.eliminate_zeros()
        digest.update(b"sparse_csr")
        digest.update(np.asarray(matrix.shape, dtype="<i8").tobytes())
        digest.update(str(matrix.data.dtype).encode())
        digest.update(np.ascontiguousarray(matrix.data).tobytes())
        # Fixed width and endianness: int32 and int64 indices describe one matrix.
        digest.update(matrix.indices.astype("<i8").tobytes())
        digest.update(matrix.indptr.astype("<i8").tobytes())
    else:
        dense = np.ascontiguousarray(X)
        digest.update(b"dense")
        digest.update(np.asarray(dense.shape, dtype="<i8").tobytes())
        digest.update(str(dense.dtype).encode())
        # Row blocks rather than one .tobytes(): X is often a memmap, and a whole-array
        # copy of a gigabyte to hash it defeats the point of memory mapping it.
        for start in range(0, dense.shape[0], 4096):
            digest.update(np.ascontiguousarray(dense[start : start + 4096]).tobytes())

    if labels is None:
        digest.update(b"no_labels")
    else:
        codes = np.ascontiguousarray(np.asarray(labels))
        digest.update(b"labels")
        digest.update(np.asarray(codes.shape, dtype="<i8").tobytes())
        digest.update(codes.astype("<i8").tobytes())

    return digest.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_digest.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add drtools/digest.py tests/test_digest.py
git commit -m "Add a canonical content digest for the cached matrix"
```

---

### Task 3: Dataset identity in the cache

**Files:**
- Modify: `drtools/cache.py:33-54` (`write_cache`), `drtools/cache.py:72-77` (`ensure_cache`)
- Test: `tests/test_cache_identity.py`

**Interfaces:**
- Consumes: `content_hash(X, labels) -> str` from Task 2
- Produces: `meta["dataset_digest"]` written by `write_cache`; `ensure_cache` raises `ContractError` on mismatch

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache_identity.py
import numpy as np
import pytest

from drtools.cache import ensure_cache, read_cache, write_cache
from drtools.contract import ContractError
from drtools.runs import RunDir

META = {"name": "t", "source": "t"}


def _run(tmp_path):
    return RunDir(tmp_path / "r1")


def test_write_cache_records_a_digest(tmp_path):
    X = np.ones((4, 3))
    meta = write_cache(_run(tmp_path), X, None, META)
    assert len(meta["dataset_digest"]) == 64


def test_ensure_cache_accepts_identical_data(tmp_path):
    run = _run(tmp_path)
    X = np.ones((4, 3))
    write_cache(run, X, None, META)
    meta = ensure_cache(run, X.copy(), None, META)
    assert meta["cached_shape"] == [4, 3]


def test_ensure_cache_refuses_changed_data(tmp_path):
    run = _run(tmp_path)
    write_cache(run, np.ones((4, 3)), None, META)
    with pytest.raises(ContractError) as error:
        ensure_cache(run, np.ones((4, 2)), None, META)
    assert "new run" in str(error.value)


def test_ensure_cache_refuses_changed_labels(tmp_path):
    run = _run(tmp_path)
    X = np.ones((2, 3))
    write_cache(run, X, np.array([0, 1]), META)
    with pytest.raises(ContractError):
        ensure_cache(run, X, np.array([1, 0]), META)


def test_cached_data_still_reads_back(tmp_path):
    run = _run(tmp_path)
    write_cache(run, np.arange(12.0).reshape(4, 3), np.array([0, 1, 0, 1]), META)
    X, labels, meta = read_cache(run)
    assert X.shape == (4, 3)
    assert labels.tolist() == [0, 1, 0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cache_identity.py -q`
Expected: FAIL — `KeyError: 'dataset_digest'` on the first test

- [ ] **Step 3: Write the implementation**

In `drtools/cache.py`, add the import and the digest, and replace `ensure_cache`:

```python
from drtools.contract import ContractError, Matrix
from drtools.digest import content_hash
```

In `write_cache`, add one key to `descriptor`:

```python
    descriptor = {
        **meta,
        "cached_storage": "sparse_csr" if sparse else "dense",
        "cached_shape": list(X.shape),
        "cached_dtype": str(X.dtype),
        "has_labels": labels is not None,
        "dataset_digest": content_hash(X, labels),
    }
```

Replace `ensure_cache` entirely:

```python
def ensure_cache(
    run: RunDir, X: Matrix, labels: np.ndarray | None, meta: dict[str, Any]
) -> dict[str, Any]:
    """Cache on first contact; on every later contact, verify rather than trust.

    The old behaviour was to return the existing cache untouched whenever one existed,
    which let a repaired loader produce a profile of one matrix and candidates of
    another under a single run id.
    """
    if not is_cached(run):
        return write_cache(run, X, labels, meta)

    cached = jsonio.read(run.path / "data" / "meta.json")
    incoming = content_hash(X, labels)
    if cached.get("dataset_digest") != incoming:
        raise ContractError(
            f"this run was created from a different dataset. The cache holds "
            f"{cached.get('dataset_digest', 'no digest')[:12]} and the data just "
            f"loaded is {incoming[:12]}. A run describes one dataset, so analyse the "
            f"changed data in a new run rather than reusing this one."
        )
    return cached
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cache_identity.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 191 passed. `tests/test_isolation.py` calls `write_cache` at lines 26, 39 and 53 and is unaffected, because the digest is computed inside rather than demanded in `meta`.

- [ ] **Step 6: Commit**

```bash
git add drtools/cache.py tests/test_cache_identity.py
git commit -m "Verify dataset identity instead of trusting an existing cache"
```

---

### Task 4: Commands read the cache, and a refusal writes nothing

**Files:**
- Modify: `drtools/cli.py:232-236` (`--data` no longer required), `drtools/cli.py:270-281` (`_cmd_profile`), `drtools/cli.py:283-300` (`_cmd_recon`), `drtools/cli.py:336-341` (`_cmd_embed`), `drtools/cli.py:681-694` (`_load`, `_open_run`)
- Test: `tests/test_cli_identity.py`

**Interfaces:**
- Consumes: `ensure_cache` refusal from Task 3
- Produces: `_resolve_run(args) -> tuple[RunDir, Matrix, np.ndarray | None, dict]` — opens the Run, loads or reads the cache, and returns the matrix every handler should use

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_identity.py
import json


def test_reprofiling_changed_data_is_refused(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    first = csv_dataset(rows=60, cols=8)
    assert cli("profile", "--data", first, "--runs-root", runs, "--run-id", "r1").code == 0

    second = csv_dataset(rows=60, cols=4)
    result = cli("profile", "--data", second, "--run-dir", runs / "r1")
    assert result.code == 2
    assert "new run" in result.stderr


def test_a_refused_reprofile_leaves_the_manifest_untouched(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    before = (runs / "r1" / "run.json").read_text(encoding="utf-8")

    cli("profile", "--data", csv_dataset(rows=60, cols=4), "--run-dir", runs / "r1")

    assert (runs / "r1" / "run.json").read_text(encoding="utf-8") == before


def test_profile_without_data_reads_the_cache(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")

    result = cli("profile", "--run-dir", runs / "r1")
    assert result.code == 0
    assert result.payload["shape"]["n_features"] == 8


def test_profile_describes_the_cache_not_the_source(cli, csv_dataset, tmp_path):
    """The reproduction: a repaired loader must not leave profile and cache disagreeing."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")

    result = cli("profile", "--data", csv_dataset(rows=60, cols=4), "--run-dir", runs / "r1")
    assert result.code == 2

    meta = json.loads((runs / "r1" / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["cached_shape"] == [60, 8]


def test_a_missing_run_dir_is_refused_rather_than_created(cli, tmp_path):
    result = cli("recon", "--run-dir", tmp_path / "absent")
    assert result.code != 0
    assert not (tmp_path / "absent").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_identity.py -q`
Expected: FAIL — the first test errors with `SystemExit: 2` from argparse, because `--data` is still required.

- [ ] **Step 3: Write the implementation**

Make `--data` optional in `_add_data_arguments`:

```python
    parser.add_argument(
        "--data",
        default=None,
        help="built-in name (see `drtools datasets`) or a path to a data file; "
        "optional when --run-dir names a run that already holds a cached dataset",
    )
```

Add `_resolve_run` next to `_open_run`:

```python
def _resolve_run(args: argparse.Namespace) -> tuple[RunDir, Any, Any, dict[str, Any]]:
    """The run and the matrix every handler should work from.

    A run holds one dataset. When it already has a cache that is the dataset, and
    `--data` becomes a verification rather than a second source of truth: it is loaded,
    digested, and refused if it disagrees. Nothing is written before that check.
    """
    spec = getattr(args, "data", None)

    if args.run_dir:
        path = Path(args.run_dir)
        if not path.exists():
            raise ContractError(
                f"no run at {path}. Omit --run-dir to create one, or correct the path."
            )
        run = RunDir(path)
        if is_cached(run):
            if spec is None:
                X, labels, meta = read_cache(run)
                return run, X, labels, meta
            X, labels, meta = _load(args)
            meta = ensure_cache(run, X, labels, meta)   # refuses before anything writes
            X, labels, _ = read_cache(run)
            return run, X, labels, meta
        if spec is None:
            raise ContractError(
                f"the run at {path} holds no cached dataset yet, so --data is needed "
                "to say which dataset this run is of."
            )
        X, labels, meta = _load(args)
        return run, X, labels, ensure_cache(run, X, labels, meta)

    if spec is None:
        raise ContractError(
            "--data is required when no --run-dir is given: there is no run to read a "
            "cached dataset from."
        )
    X, labels, meta = _load(args)
    run = _open_run(args, meta)
    return run, X, labels, ensure_cache(run, X, labels, meta)
```

Add the imports `_resolve_run` needs at the top of `cli.py`:

```python
from drtools.cache import ensure_cache, is_cached, read_cache
```

Rewrite `_cmd_profile` so nothing is written until the digest check has passed:

```python
def _cmd_profile(args: argparse.Namespace) -> dict[str, Any]:
    run, X, labels, meta = _resolve_run(args)
    # The manifest is written only now: a refused re-profile must not have overwritten
    # the command, spec and timestamp of the run it just declined to touch.
    # `spec` falls back to the cached source, so omitting --data records where the data
    # came from rather than recording null.
    run.write_manifest(
        dataset=meta.get("name"),
        spec=args.data or meta.get("source"),
        seed=args.seed,
    )

    profile = profile_dataset(X, labels, meta)
    profile["run_id"] = run.id
    # Which matrix this describes, so a loader repair that changed nothing is visible as
    # having changed nothing rather than looking like it was never read.
    profile["dataset_digest"] = meta.get("dataset_digest")
    run.write_artifact("profile.json", profile)
    return profile
```

Apply the same `args.data or meta.get("source")` fallback to `_cmd_recon`'s `write_manifest` call.

Rewrite the head of `_cmd_recon` the same way:

```python
def _cmd_recon(args: argparse.Namespace) -> dict[str, Any]:
    run, X, labels, meta = _resolve_run(args)

    if run.profile_path.exists():
        profile = run.read_artifact("profile.json")
    else:
        profile = profile_dataset(X, labels, meta)
        run.write_manifest(dataset=meta.get("name"), spec=args.data, seed=args.seed)
        run.write_artifact("profile.json", profile)
```

And the head of `_cmd_embed`:

```python
def _cmd_embed(args: argparse.Namespace) -> dict[str, Any]:
    stages = _read_stages(args.stages)
    run, X, labels, meta = _resolve_run(args)
```

Delete the now-dead `ensure_cache(run, X, labels, meta)` line that followed it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_identity.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 196 passed

- [ ] **Step 6: Commit**

```bash
git add drtools/cli.py tests/test_cli_identity.py
git commit -m "Read the cached dataset rather than reloading the source"
```

---

### Task 5: The Run seed stops moving

**Files:**
- Modify: `drtools/cli.py` (`_cmd_profile`, `_cmd_recon` — the `write_manifest` calls from Task 4)
- Test: `tests/test_cli_seed.py`

**Interfaces:**
- Consumes: `_resolve_run` from Task 4
- Produces: `run_seed(run: RunDir, requested: int) -> int` in `cli.py` — the Run's immutable seed, refusing a different one

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_seed.py
import json


def _seed(run_dir):
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["seed"]


def test_reprofiling_cannot_move_the_seed(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    data = csv_dataset(rows=60, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1", "--seed", 0)
    assert _seed(runs / "r1") == 0

    result = cli("profile", "--data", data, "--run-dir", runs / "r1", "--seed", 7)

    assert result.code == 2
    assert "seed" in result.stderr
    assert _seed(runs / "r1") == 0


def test_reprofiling_with_the_same_seed_is_fine(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    data = csv_dataset(rows=60, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1", "--seed", 3)
    result = cli("profile", "--data", data, "--run-dir", runs / "r1", "--seed", 3)
    assert result.code == 0
    assert _seed(runs / "r1") == 3


def test_omitting_the_seed_keeps_the_recorded_one(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    data = csv_dataset(rows=60, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1", "--seed", 5)
    result = cli("profile", "--data", data, "--run-dir", runs / "r1")
    assert result.code == 0
    assert _seed(runs / "r1") == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_seed.py -q`
Expected: FAIL — the first test gets `code == 0` and `_seed == 7`

- [ ] **Step 3: Write the implementation**

Add to `cli.py`, next to `_resolve_run`:

```python
def run_seed(run: RunDir, requested: int | None) -> int:
    """The run's seed, which is fixed when the run is created.

    Every metric subsample is drawn from this. Letting a re-profile move it would let
    two candidates be measured on different rows without the registered plan changing,
    which is the comparability guarantee defeated by a route that looks compliant.
    """
    if not run.manifest_path.exists():
        return 0 if requested is None else requested

    recorded = jsonio.read(run.manifest_path).get("seed")
    if recorded is None:
        return 0 if requested is None else requested
    if requested is not None and requested != recorded:
        raise ContractError(
            f"this run was created with seed {recorded} and every measurement in it "
            f"is drawn from that seed, so it cannot be re-run with seed {requested}. "
            "Use the recorded seed, or start a new run for the new one."
        )
    return recorded
```

`--seed` must be able to say "not given", so change its default in `_add_run_arguments`:

```python
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed; fixed when the run is created and immutable afterwards",
    )
```

Then in `_cmd_profile`, resolve the seed before writing anything:

```python
def _cmd_profile(args: argparse.Namespace) -> dict[str, Any]:
    run, X, labels, meta = _resolve_run(args)
    seed = run_seed(run, args.seed)
    run.write_manifest(
        dataset=meta.get("name"), spec=args.data or meta.get("source"), seed=seed
    )
```

And the same in `_cmd_recon`'s `write_manifest` call. Everywhere else that reads `args.seed` to drive a pipeline — `_cmd_embed`, `_cmd_prepare_reference` — replace it with `run_seed(run, args.seed)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_seed.py -q`
Expected: PASS, 3 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 199 passed

- [ ] **Step 6: Commit**

```bash
git add drtools/cli.py tests/test_cli_seed.py
git commit -m "Fix the run seed at creation so measurements stay comparable"
```

---

### Task 6: The published neighbourhood rule and its guards

**Files:**
- Modify: `drtools/metrics.py:39` (`DEFAULT_K`), `drtools/metrics.py:137-205` (`evaluate_embedding`)
- Modify: `tests/test_metrics_rank_plan.py:52,64,84,94,105` (drop the `k=` argument)
- Test: `tests/test_metrics_rule.py`

**Interfaces:**
- Consumes: nothing
- Produces: `neighbourhood_size(n: int) -> int` in `metrics.py`; `evaluate_embedding(reference, embedding, labels=None, *, seed=0, max_samples=METRIC_SAMPLE_CAP, runtime_s=None)` — note `k` is gone from the signature

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics_rule.py
import numpy as np
import pytest

from drtools.metrics import evaluate_embedding, neighbourhood_size

LOCAL = ("trustworthiness", "continuity", "shepard_correlation")


def _pair(n, d_ref=5, d_emb=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d_ref)), rng.normal(size=(n, d_emb))


@pytest.mark.parametrize("n", [3, 4, 5, 6, 19, 20, 30, 31, 32, 1000])
def test_the_rule_satisfies_sklearns_constraint(n):
    assert neighbourhood_size(n) < n / 2


def test_the_rule_saturates_at_fifteen():
    assert neighbourhood_size(1000) == 15
    assert neighbourhood_size(31) == 15
    assert neighbourhood_size(30) == 14


@pytest.mark.parametrize("n", [20, 31, 200])
def test_local_metrics_are_produced_above_the_floor(n):
    reference, embedding = _pair(n)
    values = evaluate_embedding(reference, embedding)["values"]
    assert all(values[name] is not None for name in LOCAL)


@pytest.mark.parametrize("n", [2, 3, 4, 19])
def test_local_metrics_are_absent_below_the_floor(n):
    reference, embedding = _pair(n)
    values = evaluate_embedding(reference, embedding)["values"]
    assert all(values[name] is None for name in LOCAL)


def test_a_small_dataset_scores_instead_of_raising():
    """The reproduction: k=15 against 20 rows used to raise ValueError."""
    reference, embedding = _pair(20)
    result = evaluate_embedding(reference, embedding)
    assert result["values"]["trustworthiness"] is not None


def test_silhouette_is_absent_when_every_row_is_its_own_class():
    reference, embedding = _pair(4)
    values = evaluate_embedding(reference, embedding, np.arange(4))["values"]
    assert values["silhouette"] is None


def test_silhouette_is_present_when_classes_are_valid():
    reference, embedding = _pair(60)
    values = evaluate_embedding(reference, embedding, np.arange(60) % 3)["values"]
    assert values["silhouette"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics_rule.py -q`
Expected: FAIL — `ImportError: cannot import name 'neighbourhood_size'`

- [ ] **Step 3: Write the implementation**

Replace `DEFAULT_K = 15` in `metrics.py` with the rule and the floor:

```python
#: Largest neighbourhood the battery ever uses.
MAX_K = 15

#: Below this many rows the local metrics are reported as unavailable rather than
#: computed. The rule would still satisfy scikit-learn here, but trustworthiness over
#: one or two neighbours is noise, and `rank` would weight it at face value.
LOCAL_METRIC_FLOOR = 20


def neighbourhood_size(n: int) -> int:
    """The battery's neighbourhood, chosen by rule rather than by the agent.

    scikit-learn requires `k < n / 2` for trustworthiness. The published rule is the
    largest k that satisfies it, capped at MAX_K: n=30 gives 14 against a limit of
    15.0, n=31 gives 15 against 15.5.
    """
    return max(1, min(MAX_K, math.ceil(n / 2) - 1))
```

Add `import math` at the top of `metrics.py`.

In `evaluate_embedding`, remove `k: int = DEFAULT_K` from the signature and replace the clamp at line 173 and everything that depended on it:

```python
    n_used = reference.shape[0]
    k = neighbourhood_size(n_used)
    values: dict[str, float | None] = {}

    if n_used < LOCAL_METRIC_FLOOR:
        values["trustworthiness"] = None
        values["continuity"] = None
        values["shepard_correlation"] = None
    else:
        values["trustworthiness"] = float(
            trustworthiness(reference, embedding, n_neighbors=k)
        )
        # Continuity is trustworthiness with the two spaces exchanged: intrusions in one
        # direction are extrusions in the other.
        values["continuity"] = float(
            trustworthiness(embedding, reference, n_neighbors=k)
        )
        values["shepard_correlation"] = _shepard(reference, embedding)

    reference_values: dict[str, float | None] = {}
    n_classes = 0 if labels is None else int(np.unique(labels).size)

    # scikit-learn wants 1 < n_classes < n_used for silhouette, which is a separate
    # constraint from the neighbourhood: n=4 with 4 classes raises whatever k is.
    if labels is not None and 1 < n_classes < n_used:
        values["silhouette"] = float(silhouette_score(embedding, labels))
        reference_values["silhouette"] = float(silhouette_score(reference, labels))
    else:
        values["silhouette"] = None

    # The kNN agreement needs k + 1 rows to have k neighbours besides the point itself.
    if labels is not None and n_classes > 1 and k + 1 <= n_used:
        values["knn_label_preservation"] = _label_agreement(embedding, labels, k)
        reference_values["knn_label_preservation"] = _label_agreement(
            reference, labels, k
        )
    else:
        values["knn_label_preservation"] = None

    values["runtime_s"] = runtime_s
```

Record the settings the result was produced under, so `rank` can check them later — add to the returned dict alongside `values` and `reference_values`:

```python
        "settings": {"k": k, "max_samples": max_samples, "seed": seed},
```

Then remove the `k=` argument from the five call sites in `tests/test_metrics_rank_plan.py` (lines 52, 64, 84, 94, 105). Every assertion in those four scenarios holds at the rule's k.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics_rule.py tests/test_metrics_rank_plan.py -q`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 212 passed

- [ ] **Step 6: Commit**

```bash
git add drtools/metrics.py tests/test_metrics_rule.py tests/test_metrics_rank_plan.py
git commit -m "Derive the battery's neighbourhood by rule, and guard the label metrics"
```

---

### Task 7: Registration, and the Reference that follows from it

Registration and the Reference are one task because neither is testable without the
other: `prepare-reference` reads the registered Plan for its Base preprocessing, and
nothing writes `plan.registered.json` until `validate-plan` registers.

**Files:**
- Modify: `drtools/cli.py:168-186` (`prepare-reference` and `evaluate` parsers), `drtools/cli.py:370-400` (`_cmd_prepare_reference`), `drtools/cli.py:410-433` (`_cmd_evaluate`), `drtools/cli.py:470-497` (`_cmd_validate_plan`), `drtools/cli.py:436-468` (`_cmd_rank`)
- Test: `tests/test_cli_registration.py`, `tests/test_cli_reference.py`

**Interfaces:**
- Consumes: `neighbourhood_size` from Task 6, `run_seed` from Task 5
- Produces: `_registered_plan(run: RunDir) -> Plan`; `_require_run(args) -> RunDir`; `_plan_digest(plan: dict) -> str`; `plan.registered.json` and `plan_digest` in the manifest; a `register_plan` decision record carrying `plan_digest`, `weights` and `candidates`; `reference.json` gains `settings: {k, max_samples, seed}` and `n_rows`; `evaluate` takes only `--run-dir` and `--id`

- [ ] **Step 1: Write the failing tests**

Registration first, in `tests/test_cli_registration.py`:

```python
# tests/test_cli_registration.py
import json

PCA = [{"op": "pca", "params": {"n_components": 2}}]
TSNE = [{"op": "tsne", "params": {"n_components": 2}}]


def _plan(weights):
    return {
        "dataset": "d",
        "candidates": [{"id": "a", "stages": PCA}, {"id": "b", "stages": TSNE}],
        "evaluation": {"weights": weights, "justification": "declared up front"},
    }


def _prepared(cli, csv_dataset, tmp_path, weights):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    (runs / "r1" / "plan.json").write_text(json.dumps(_plan(weights)), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    return runs / "r1"


def test_validate_plan_writes_a_frozen_copy(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    assert (run / "plan.registered.json").exists()
    assert json.loads((run / "run.json").read_text(encoding="utf-8"))["plan_digest"]


def test_registration_is_recorded_with_its_weights(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    registrations = [r for r in records if r["stage"] == "register_plan"]
    assert len(registrations) == 1
    assert registrations[0]["weights"] == {"trustworthiness": 1.0}
    assert sorted(registrations[0]["candidates"]) == ["a", "b"]


def test_rank_refuses_a_plan_rewritten_after_the_fact(cli, csv_dataset, tmp_path):
    """The reproduction: editing the weights after metrics exist used to flip the winner."""
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    for candidate in ("a", "b"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
        cli("evaluate", "--run-dir", run, "--id", candidate)

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["evaluation"]["weights"] = {"runtime_s": 1.0}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("rank", "--run-dir", run)

    assert result.code == 2
    assert "amendment" in result.stderr.lower()
    assert not (run / "ranking.json").exists()


def test_rank_stamps_the_digest_it_ranked_under(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    for candidate in ("a", "b"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
        cli("evaluate", "--run-dir", run, "--id", candidate)

    result = cli("rank", "--run-dir", run)

    assert result.code == 0
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert result.payload["plan_digest"] == manifest["plan_digest"]


def test_no_rank_record_claims_pre_registration_it_cannot_know(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    for candidate in ("a", "b"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
        cli("evaluate", "--run-dir", run, "--id", candidate)
    cli("rank", "--run-dir", run)

    log = (run / "decisions.jsonl").read_text(encoding="utf-8")
    assert "before any embedding was computed" not in log
```

Then the Reference, in `tests/test_cli_reference.py`:

```python
# tests/test_cli_reference.py
import json


def _plan(base, candidates):
    return {
        "dataset": "d",
        "base_preprocessing": base,
        "candidates": candidates,
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "declared"},
    }


PCA = [{"op": "pca", "params": {"n_components": 2}}]


def test_prepare_reference_takes_base_stages_from_the_plan(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = _plan([{"op": "standardise", "params": {}}], [{"id": "a", "stages": PCA}])
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")

    result = cli("prepare-reference", "--run-dir", runs / "r1")

    assert result.code == 0
    assert result.payload["stages"] == [{"op": "standardise", "params": {}}]


def test_reference_records_the_settings_it_fixes(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1", "--seed", 4)
    plan = _plan([{"op": "standardise", "params": {}}], [{"id": "a", "stages": PCA}])
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("prepare-reference", "--run-dir", runs / "r1")

    recorded = json.loads((runs / "r1" / "data" / "reference.json").read_text(encoding="utf-8"))
    assert recorded["settings"] == {"k": 15, "max_samples": 2000, "seed": 4}
    assert recorded["n_rows"] == 60


def test_evaluate_refuses_when_the_plan_declares_a_base_and_none_was_prepared(
    cli, csv_dataset, tmp_path
):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = _plan([{"op": "standardise", "params": {}}], [{"id": "a", "stages": PCA}])
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("embed", "--run-dir", runs / "r1", "--id", "a", "--in-process")

    result = cli("evaluate", "--run-dir", runs / "r1", "--id", "a")

    assert result.code == 2
    assert "prepare-reference" in result.stderr


def test_evaluate_needs_no_reference_when_the_plan_declares_no_base(
    cli, csv_dataset, tmp_path
):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    (runs / "r1" / "plan.json").write_text(
        json.dumps(_plan([], [{"id": "a", "stages": PCA}])), encoding="utf-8"
    )
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("embed", "--run-dir", runs / "r1", "--id", "a", "--in-process")

    result = cli("evaluate", "--run-dir", runs / "r1", "--id", "a")
    assert result.code == 0


def test_evaluate_rejects_a_neighbourhood_argument(cli, tmp_path):
    result = cli("evaluate", "--run-dir", tmp_path, "--id", "a", "--k", "5")
    assert result.code != 0
```

- [ ] **Step 2: Run both files to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_registration.py tests/test_cli_reference.py -q`
Expected: FAIL — no `plan.registered.json` is written, and `prepare-reference` still demands `--stages`

- [ ] **Step 3: Register the plan at validation**

Add the registration helpers to `cli.py`:

```python
def _plan_digest(plan: dict[str, Any]) -> str:
    """A digest over the plan as registered, so divergence is detectable."""
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _registered_plan(run: RunDir) -> Plan:
    path = run.path / "plan.registered.json"
    if not path.exists():
        raise ContractError(
            "this run has no registered plan. Run `drtools validate-plan` to register "
            "one before embedding or ranking: the weighting has to be fixed before any "
            "embedding exists for pre-registration to mean anything."
        )
    return Plan.model_validate(jsonio.read(path))
```

Add `import hashlib` and `import json` to `cli.py` if absent.

Extend `_cmd_validate_plan` — after the existing `jsonio.write(... "plan_validation.json", report)` and the error-logging loop, register when there are no errors:

```python
    errors = [f for f in report["findings"] if f["severity"] == "error"]
    if errors:
        return report

    registered = Plan.model_validate(plan)
    digest = _plan_digest(registered.model_dump(mode="json"))
    jsonio.write(run.path / "plan.registered.json", registered.model_dump(mode="json"))
    run.update_manifest(plan_digest=digest)
    run.log_decision(
        stage="register_plan",
        question="What will this run compare, and how will the results be judged?",
        chosen=f"registered {len(registered.candidates)} candidates",
        rationale=registered.evaluation.justification
        or "the weighting was declared before this record was written",
        evidence=["profile.shape.n_samples"],
        plan_digest=digest,
        weights=dict(registered.evaluation.weights),
        candidates=[c.id for c in registered.candidates],
    )
    report["plan_digest"] = digest
    return report
```

Rewrite `_cmd_rank` to read the registration rather than the live file, and delete the false fallback rationale:

```python
def _cmd_rank(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    plan = _registered_plan(run)

    live = run.path / "plan.json"
    if live.exists():
        current = Plan.model_validate(run.read_artifact("plan.json"))
        if _plan_digest(current.model_dump(mode="json")) != _plan_digest(
            plan.model_dump(mode="json")
        ):
            raise ContractError(
                "plan.json no longer matches the plan this run registered, so ranking "
                "it would score results under a weighting chosen after they existed. "
                "Restore the registered plan, or record an amendment — which this "
                "toolbox does not yet implement, so a changed weighting means a new run."
            )

    metrics_by_id: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for candidate in plan.candidates:
        record_path = run.path / "embeddings" / f"{candidate.id}.json"
        metrics_path = run.path / "metrics" / f"{candidate.id}.json"
        outcome = jsonio.read(record_path).get("status") if record_path.exists() else None
        if outcome == "ok" and metrics_path.exists():
            metrics_by_id[candidate.id] = jsonio.read(metrics_path)
        elif record_path.exists():
            failures[candidate.id] = jsonio.read(record_path).get("failure", {})

    ranking = rank_candidates(
        metrics_by_id,
        plan.evaluation.weights,
        justification=plan.evaluation.justification,
        failures=failures,
    )
    ranking["plan_digest"] = jsonio.read(run.manifest_path)["plan_digest"]
    run.write_artifact("ranking.json", ranking)
    run.log_decision(
        stage="rank",
        question="Which candidate best serves the question this analysis is answering?",
        chosen=ranking["winner"],
        rationale=plan.evaluation.justification
        or "scored under the weighting registered for this run",
        evidence=[f"metrics.{candidate}" for candidate in metrics_by_id],
        options_considered=sorted(metrics_by_id),
        weights_applied=ranking["weights_applied"],
        plan_digest=ranking["plan_digest"],
    )
    return ranking
```

The `outcome == "ok"` condition is the belt-and-braces half of Task 9: a Candidate whose current Outcome is not `ok` is never scored, whatever `metrics/` still holds.

- [ ] **Step 4: Take the Reference from the registered plan**

In the parser, drop `--stages` from `prepare-reference` and `--k`/`--max-samples` from `evaluate`:

```python
    reference = subparsers.add_parser(
        "prepare-reference",
        help="run the shared base preprocessing once and cache it as the common "
        "representation every candidate is measured against",
    )
    _add_run_arguments(reference)
    reference.set_defaults(handler=_cmd_prepare_reference)

    evaluate = subparsers.add_parser(
        "evaluate", help="score one candidate's embedding against the reference"
    )
    _add_run_arguments(evaluate)
    evaluate.add_argument("--id", required=True, help="candidate to score")
    evaluate.set_defaults(handler=_cmd_evaluate)
```

Rewrite `_cmd_prepare_reference` to read the registered Plan and record the settings:

```python
def _cmd_prepare_reference(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    plan = _registered_plan(run)
    stages = [stage.model_dump() for stage in plan.base_preprocessing]
    X, labels, _ = read_cache(run)
    seed = run_seed(run, args.seed)

    directory = run.path / "data"
    if stages:
        result = run_pipeline(
            X, labels, stages, seed=seed, require_terminal_reduction=False
        )
        reference = result.embedding
        np.save(directory / "reference.npy", np.asarray(reference))
        stage_records = result.as_dict()["stages"]
    else:
        # No base preprocessing means the reference *is* the cache. Writing a copy would
        # add a second artefact to keep honest, and np.asarray on a sparse matrix writes
        # an unloadable 0-d object array.
        reference = X
        stage_records = []

    n_rows = int(reference.shape[0])
    record = {
        "stages": stages,
        "shape": list(np.shape(reference)),
        "n_rows": n_rows,
        "is_cache": not stages,
        "settings": {
            "k": neighbourhood_size(min(n_rows, METRIC_SAMPLE_CAP)),
            "max_samples": METRIC_SAMPLE_CAP,
            "seed": seed,
        },
        "stage_records": stage_records,
    }
    jsonio.write(directory / "reference.json", record)
    return record
```

Add the imports it needs:

```python
from drtools.metrics import METRIC_SAMPLE_CAP, evaluate_embedding, neighbourhood_size
```

In `_cmd_evaluate`, refuse a missing Reference the Plan asked for, and take the settings from the record:

```python
def _cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run = _require_run(args)
    plan = _registered_plan(run)
    reference_record = run.path / "data" / "reference.json"

    if plan.base_preprocessing and not reference_record.exists():
        raise ContractError(
            "the registered plan declares base preprocessing, so candidates must be "
            "scored against its output rather than against the raw cache. Run "
            "`drtools prepare-reference` before evaluating."
        )

    if reference_record.exists():
        settings = jsonio.read(reference_record)["settings"]
    else:
        # No base preprocessing: the reference is the cache, and the same rule applies
        # to its row count.
        rows = min(jsonio.read(run.path / "data" / "meta.json")["cached_shape"][0],
                   METRIC_SAMPLE_CAP)
        settings = {
            "k": neighbourhood_size(rows),
            "max_samples": METRIC_SAMPLE_CAP,
            "seed": run_seed(run, None),
        }

    candidate = jsonio.read(run.path / "embeddings" / f"{args.id}.json")
    embedding = np.load(run.path / "embeddings" / f"{args.id}.npy")
    reference, labels = _reference_for(run, args.id)

    metrics = evaluate_embedding(
        reference,
        embedding,
        labels,
        seed=settings["seed"],
        max_samples=settings["max_samples"],
        runtime_s=candidate.get("total_duration_s"),
    )
    metrics["id"] = args.id
    metrics["reference"] = (
        "base preprocessing output"
        if (run.path / "data" / "reference.npy").exists()
        else "loaded dataset as cached"
    )
    jsonio.write(run.path / "metrics" / f"{args.id}.json", metrics)
    return metrics
```

Add `_require_run`, used by every command that cannot create one:

```python
def _require_run(args: argparse.Namespace) -> RunDir:
    """The run a command must be given, rather than one it may create."""
    if not args.run_dir:
        raise ContractError("--run-dir is required: this command reads an existing run.")
    path = Path(args.run_dir)
    if not path.exists():
        raise ContractError(f"no run at {path}.")
    return RunDir(path)
```

- [ ] **Step 5: Run both files to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_registration.py tests/test_cli_reference.py -q`
Expected: PASS, 10 passed

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 222 passed

- [ ] **Step 7: Commit**

```bash
git add drtools/cli.py tests/test_cli_registration.py tests/test_cli_reference.py
git commit -m "Register the plan at validation and derive the reference from it"
```

---


### Task 8: `embed` binds to a registered Candidate

**Files:**
- Modify: `drtools/cli.py:130-166` (`embed` parser), `drtools/cli.py:336-369` (`_cmd_embed`)
- Test: `tests/test_cli_embed_binding.py`

**Interfaces:**
- Consumes: `_registered_plan` from Task 7
- Produces: an `embed` decision record per attempt carrying `candidate`, `outcome` and `plan_digest`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_embed_binding.py
import json

PCA = [{"op": "pca", "params": {"n_components": 2}}]


def _prepared(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = {
        "dataset": "d",
        "candidates": [{"id": "a", "stages": PCA}],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    return runs / "r1"


def test_embedding_without_a_registered_plan_is_refused(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    result = cli("embed", "--run-dir", runs / "r1", "--id", "a", "--in-process")
    assert result.code == 2
    assert "validate-plan" in result.stderr


def test_an_unregistered_candidate_id_is_refused(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path)
    result = cli("embed", "--run-dir", run, "--id", "ghost", "--in-process")
    assert result.code == 2
    assert "ghost" in result.stderr


def test_stages_come_from_the_registered_plan(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path)
    result = cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    assert result.code == 0
    assert result.payload["output_shape"] == [60, 2]


def test_every_attempt_is_recorded_with_its_outcome(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path)
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    attempts = [r for r in records if r["stage"] == "embed"]
    assert len(attempts) == 1
    assert attempts[0]["candidate"] == "a"
    assert attempts[0]["outcome"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_embed_binding.py -q`
Expected: FAIL — argparse still requires `--stages`

- [ ] **Step 3: Write the implementation**

Remove `--stages` from the `embed` parser (the whole `embed.add_argument("--stages", ...)` block).

Rewrite `_cmd_embed`:

```python
def _cmd_embed(args: argparse.Namespace) -> dict[str, Any]:
    run, X, labels, _ = _resolve_run(args)
    plan = _registered_plan(run)

    candidate = next((c for c in plan.candidates if c.id == args.id), None)
    if candidate is None:
        registered = ", ".join(c.id for c in plan.candidates)
        raise ContractError(
            f"{args.id} is not a candidate in the registered plan. The plan registered "
            f"[{registered}]; add the candidate by re-registering, or embed one of those."
        )

    stages = plan.stages_for(candidate)
    seed = run_seed(run, args.seed)

    if not args.in_process:
        # A candidate that exhausts memory or never converges cannot be caught in
        # process, so by default it runs somewhere that can be killed.
        outcome = run_candidate(run, args.id, stages, seed=seed, timeout_s=args.timeout)
    else:
        result = run_pipeline(X, labels, stages, seed=seed)
        embeddings = run.path / "embeddings"
        np.save(embeddings / f"{args.id}.npy", result.embedding)
        if result.labels is not None:
            np.save(embeddings / f"{args.id}.labels.npy", result.labels)
        if result.context.sample_index is not None:
            np.save(embeddings / f"{args.id}.index.npy", result.context.sample_index)
        outcome = {"id": args.id, "status": "ok", **result.as_dict()}
        jsonio.write(embeddings / f"{args.id}.json", outcome)

    run.log_decision(
        stage="embed",
        question=f"What did candidate {args.id} produce?",
        chosen=args.id,
        rationale=candidate.rationale or "the candidate as registered",
        evidence=["plan.registered.candidates"],
        candidate=args.id,
        outcome=outcome.get("status"),
        plan_digest=jsonio.read(run.manifest_path)["plan_digest"],
    )
    return outcome
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_embed_binding.py -q`
Expected: PASS, 4 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 226 passed

- [ ] **Step 6: Commit**

```bash
git add drtools/cli.py tests/test_cli_embed_binding.py
git commit -m "Bind embed to the registered plan and record every attempt"
```

---

### Task 9: The retry loop, and invalidation before every attempt

**Files:**
- Modify: `drtools/cli.py` (`_cmd_embed` from Task 8, `_cmd_validate_plan` from Task 7)
- Test: `tests/test_cli_retry.py`

**Interfaces:**
- Consumes: everything above
- Produces: `_invalidate_candidate(run: RunDir, candidate_id: str) -> None`; re-registration rules keyed on recorded Outcome

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_retry.py
import json

import pytest

PCA = [{"op": "pca", "params": {"n_components": 2}}]
TSNE = [{"op": "tsne", "params": {"n_components": 2}}]


def _plan(candidates, weights={"trustworthiness": 1.0}):
    return {
        "dataset": "d",
        "candidates": candidates,
        "evaluation": {"weights": weights, "justification": "up front"},
    }


def _prepared(cli, csv_dataset, tmp_path, candidates):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    (runs / "r1" / "plan.json").write_text(json.dumps(_plan(candidates)), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    return runs / "r1"


def test_re_embedding_a_successful_candidate_is_refused(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    result = cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    assert result.code == 2
    assert "already" in result.stderr


def test_rank_ignores_metrics_for_a_candidate_that_did_not_succeed(
    cli, csv_dataset, tmp_path
):
    """The reproduction, in the only shape it can still take.

    A candidate that succeeded can no longer be re-embedded at all, so the original
    route to stale scores is closed upstream. This asserts the belt-and-braces half:
    even with a metrics file sitting there, a candidate whose outcome is not ok is not
    ranked. That is what stopped `rank` reporting a timed-out candidate as a winner.
    """
    run = _prepared(cli, csv_dataset, tmp_path,
                    [{"id": "a", "stages": PCA}, {"id": "b", "stages": TSNE}])
    for candidate in ("a", "b"):
        cli("embed", "--run-dir", run, "--id", candidate, "--in-process")
        cli("evaluate", "--run-dir", run, "--id", candidate)

    # Exactly what a timed-out retry used to leave behind: a stale metrics file beside
    # an outcome that is no longer ok.
    record = json.loads((run / "embeddings" / "b.json").read_text(encoding="utf-8"))
    record["status"] = "timeout"
    (run / "embeddings" / "b.json").write_text(json.dumps(record), encoding="utf-8")
    assert (run / "metrics" / "b.json").exists()

    ranked = cli("rank", "--run-dir", run)

    assert [r["id"] for r in ranked.payload["ranking"]] == ["a"]
    assert "b" in ranked.payload["failed_candidates"]


def test_a_retry_clears_what_the_previous_attempt_left(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path,
                    [{"id": "a", "stages": [{"op": "subsample", "params": {"n_samples": 30}},
                                            *PCA]}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    assert (run / "embeddings" / "a.index.npy").exists()

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = PCA
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    # a succeeded, so revising it is refused; a failed attempt is the retry path.
    record = json.loads((run / "embeddings" / "a.json").read_text(encoding="utf-8"))
    record["status"] = "failed"
    (run / "embeddings" / "a.json").write_text(json.dumps(record), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0
    assert cli("embed", "--run-dir", run, "--id", "a", "--in-process").code == 0

    # The subsample index from the first attempt must not survive to subset a reference
    # the second attempt never subsampled.
    assert not (run / "embeddings" / "a.index.npy").exists()


@pytest.mark.parametrize("outcome", ["failed", "timeout", "crashed"])
def test_any_unsuccessful_outcome_may_be_revised(cli, csv_dataset, tmp_path, outcome):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    record = json.loads((run / "embeddings" / "a.json").read_text(encoding="utf-8"))
    record["status"] = outcome
    (run / "embeddings" / "a.json").write_text(json.dumps(record), encoding="utf-8")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = [{"op": "pca", "params": {"n_components": 3}}]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0


def test_a_failed_candidate_can_be_revised_and_retried(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--timeout", "0.01")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = [{"op": "pca", "params": {"n_components": 3}}]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0
    assert cli("embed", "--run-dir", run, "--id", "a", "--in-process").code == 0


def test_a_successful_candidates_stages_cannot_be_revised(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"][0]["stages"] = [{"op": "pca", "params": {"n_components": 3}}]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 2
    assert "succeeded" in result.stderr


def test_the_weighting_can_never_be_revised(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["evaluation"]["weights"] = {"runtime_s": 1.0}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 2
    assert "amendment" in result.stderr.lower()


def test_adding_a_candidate_is_allowed_after_embedding(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, [{"id": "a", "stages": PCA}])
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"].append({"id": "b", "stages": TSNE})
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_retry.py -q`
Expected: FAIL — re-embedding a successful Candidate returns 0

- [ ] **Step 3: Write the implementation**

Add the invalidation helper and the outcome lookup to `cli.py`:

```python
def _recorded_outcome(run: RunDir, candidate_id: str) -> str | None:
    """How this candidate last ended according to the decision log.

    The freeze anchors here rather than on files, because files can be deleted: an
    agent that removes `embeddings/` would otherwise be free to re-register a different
    weighting and re-run. An attempt that began but recorded no outcome — the process
    died, the machine rebooted — reads as `crashed`, which is revisable, so an
    interrupted run is recoverable rather than wedged.
    """
    attempts = [
        record
        for record in run.decisions()
        if record.get("stage") == "embed" and record.get("candidate") == candidate_id
    ]
    if not attempts:
        return None
    return attempts[-1].get("outcome") or "crashed"


def _invalidate_candidate(run: RunDir, candidate_id: str) -> None:
    """Clear everything derived from a previous attempt at this candidate.

    The toolbox does this rather than the agent, and does it before *every* attempt
    rather than only when stages change: the commonest retry of all is the same stages
    with a bigger budget, and leaving the old metrics in place lets a candidate that has
    just timed out be ranked on the scores of the run before it.
    """
    for path in (run.path / "embeddings").glob(f"{candidate_id}.*"):
        path.unlink()
    metrics = run.path / "metrics" / f"{candidate_id}.json"
    metrics.unlink(missing_ok=True)
```

In `_cmd_embed`, between resolving the Candidate and running it:

```python
    outcome_so_far = _recorded_outcome(run, args.id)
    if outcome_so_far == "ok":
        raise ContractError(
            f"candidate {args.id} has already produced an embedding. Its stages are "
            "fixed once it has succeeded, and replacing the result would leave the "
            "embedding disagreeing with the record of how it was produced. Register a "
            "new candidate id to try something different."
        )
    _invalidate_candidate(run, args.id)
```

In `_cmd_validate_plan`, before writing the frozen copy, check the proposed Plan against the registered one:

```python
    existing = run.path / "plan.registered.json"
    if existing.exists():
        previous = Plan.model_validate(jsonio.read(existing))
        _check_revision(run, previous, registered)
```

And add the rule itself:

```python
def _check_revision(run: RunDir, previous: Plan, proposed: Plan) -> None:
    """What a re-registration may change once candidates have run.

    The weighting never moves. A candidate that has succeeded is finished. Everything
    else — adding a candidate, revising one that failed, timed out or crashed — is the
    one diagnose-and-retry the design promises, and is the clearest evidence of agency
    the run can produce.
    """
    if dict(previous.evaluation.weights) != dict(proposed.evaluation.weights):
        raise ContractError(
            "the pre-registered weighting cannot change once it is registered. Moving "
            "it after results exist is what pre-registration prevents; the sanctioned "
            "route is an amendment, which this toolbox does not yet implement."
        )
    if previous.base_preprocessing != proposed.base_preprocessing:
        raise ContractError(
            "the base preprocessing cannot change once registered: every candidate is "
            "scored against its output, so moving it would invalidate every metric "
            "already computed. Start a new run for a different base."
        )

    proposed_by_id = {c.id: c for c in proposed.candidates}
    for candidate in previous.candidates:
        if candidate.id not in proposed_by_id:
            raise ContractError(
                f"candidate {candidate.id} cannot be removed from a registered plan: a "
                "candidate that ran and lost is part of the record."
            )
        if proposed_by_id[candidate.id].stages == candidate.stages:
            continue
        if _recorded_outcome(run, candidate.id) == "ok":
            raise ContractError(
                f"candidate {candidate.id} has already succeeded, so its stages cannot "
                "be revised. Register a new candidate id for the variant."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cli_retry.py -q`
Expected: PASS, 10 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 236 passed

- [ ] **Step 6: Commit**

```bash
git add drtools/cli.py tests/test_cli_retry.py
git commit -m "Invalidate before every attempt and allow the diagnose-and-retry loop"
```

---

## After the plan

The spec defers to day 9: `reference_digest`, per-record reference digests, and the mixed-digest check in `rank`. Do not add them here — they collide with two queued day-9 defects in the same lines of `_reference_for`.

`recon` still carries its own `--k` and `--max-samples` with a separate `DEFAULT_K` in `recon.py:36`, so the agent can still tune the evidence that justifies its own Plan. That is a real gap and it is day 9's, not day 7's; widening this work to cover it is how a one-day slip becomes two.

Once this lands, `domain-modeling` names the two concepts `CONTEXT.md` lacks — the dataset's identity within a Run, and the settings the Battery is computed under — and `design/notes.md` gets the day 7 decision log entry.
