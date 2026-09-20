# Report contract implementation plan

> **For agentic workers:** one implementer works this plan in order. Later tasks
> reference earlier ones rather than repeating their code. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** `drtools report` emits the nine-section report with every number already in
it, `drtools report --refresh` regenerates those numbers without touching prose, and
`drtools render` produces the PDF only from a Markdown that still agrees with the Run.

**Architecture:** A new `drtools/report.py` owns the fenced blocks — generating their
content from the Run's artefacts, parsing them out of a document, and detecting hand
edits through a digest carried in the fence. A new `drtools/render.py` owns the pandoc
call and its two environment refusals. `drtools/cli.py` gains two handlers and two
subparsers, matching the shape of the existing ones.

**Tech Stack:** Python 3.12, argparse, hashlib, pydantic (already in use), pytest.
pandoc 3.9 and MiKTeX are installed in this environment.

**Spec:** `design/specs/2026-09-20-report-contract.md`

## Global Constraints

- **Refuse, never warn.** A violation is a `ContractError` that writes nothing. No
  override flag on any refusal in this plan.
- **Every refusal names the route out** and does not name a bypass.
- **The agent never retypes a number.** Anything numeric in the report comes from a
  block; anything outside a block is prose.
- **Line endings are CRLF** in this working tree for `.py` and `.md`. Write with
  `newline=""` or via the editing tools; do not let a rewrite convert the file.
- **Tests drive the CLI in-process** through the `cli` fixture in `tests/conftest.py`,
  which returns `CliResult(code, payload, stderr)`. Refusals assert on `code == 2` and
  a substring of `stderr`.
- **`DRTOOLS_ALLOW_IN_PROCESS`** is set autouse by `tests/conftest.py`, so tests may
  pass `--in-process` to `embed`.

---

## File structure

| File | Responsibility |
|---|---|
| `drtools/report.py` (create) | Block ids, fence format, digest, parse/replace, block generation from artefacts, skeleton assembly, staleness comparison |
| `drtools/render.py` (create) | Locating pandoc and a PDF engine; the subprocess call; the two environment refusals |
| `drtools/cli.py` (modify) | `_cmd_report`, `_cmd_render`, two subparsers |
| `skills/write-report/SKILL.md` (modify) | Name the two commands in the prose the agent reads |
| `design/notes.md` (modify) | Day 10 decision log entry |
| `tests/test_report.py` (create) | Fence mechanics, block content, `report`, `--refresh` |
| `tests/test_render.py` (create) | Staleness refusal and the two environment refusals |

### Block inventory (ids are the contract between tasks)

| id | Section | Source |
|---|---|---|
| `profile` | 1. Dataset profile | `profile.json`, `data/meta.json` |
| `preprocessing` | 2. Preprocessing decisions, and why | `plan.registered.json`, `decisions.jsonl` |
| `methods` | 3. Methods selected and rejected | `plan.registered.json`, `decisions.jsonl` |
| `hyperparameters` | 4. Hyperparameter choices, and why | `embeddings/<id>.json` |
| `figures` | 5. Figures | `figures/figures.json` |
| `metrics` | 6. Quantitative comparison | `metrics/*.json`, `ranking.json` |
| `ranking` | 7. Ranking, with the weighting justification | `ranking.json` |
| `limitations` | 9. Limitations | `ranking.json`, `metrics/*.json` |

Section 8, Interpretation, has no block. This is deliberate and is tested.

### Artefact shapes, verified against a real Run

Read these before writing Task 2; they were confirmed by building a complete Run on
the `blobs` fixture, not inferred from the code.

```
profile.json      name, spec, source, modality, value_kind_declared,
                  shape{n_samples,n_features,aspect_ratio,dtype,storage,memory_mb},
                  values{sparsity,min,max,is_nonnegative,is_integer_valued,
                         n_distinct_sampled,suspected_kind},
                  features{...}, samples{...},
                  labels{present,kind,n_classes,class_counts,names,balance_ratio,
                         smallest_class_size},
                  observations[], run_id, dataset_digest

data/meta.json    name, source, label_kind, label_names[], spec, cached_storage,
                  cached_shape[2], cached_dtype, has_labels, dataset_digest,
                  adapter{path,sha256,size}   # present only when an adapter ran

ranking.json      ranking[{id, score, rank,
                           contributions{<metric>:{raw,normalised,weight,contribution}}}],
                  winner, weights_declared{}, weights_applied{}, weights_dropped{},
                  justification, failed_candidates[], notes[], plan_digest

metrics/<id>.json values{trustworthiness,continuity,shepard_correlation,silhouette,
                         knn_label_preservation,runtime_s},
                  reference_values{}, n_used, n_total, subsampled, seed,
                  settings{k,max_samples,seed}, notes[], id, reference

embeddings/<id>.json  id, status, output_shape[2], total_duration_s,
                      n_samples_subsampled,
                      stages[{op, params{}, param_provenance{<param>:"specified"
                              |"registry_default"}, input_shape, output_shape,
                              duration_s, notes{}}]
                      # on failure: status != "ok", failure{op,error_type,message}

figures/figures.json  <figure name>:{path, ...}   # path is ABSOLUTE
                      comparison{panels{<id>:{render_mode,n_points,n_classes,
                                              identity_channel,points_outside_view}},path}
                      embedding_<id>{render_mode,n_points,n_classes,identity_channel,
                                     points_outside_view,path}
                      class_facet{}, metrics{metrics[],path}, scree{path},
                      recon_thumbnail{drawn,source,caveat,...,path}, shepard{path}

decisions.jsonl   {timestamp, stage, question, options_considered[], chosen,
                   rationale, evidence[], actor,
                   evidence_resolved{}   # written by log-decision only
                   ...}
```

Two consequences, both corrections to the spec made while reading these:

1. **Hyperparameters come from `embeddings/<id>.json`, not the Plan.** The Plan holds
   what was asked for; the embedding record holds what ran, with registry defaults
   filled in and `param_provenance` marking which is which. The report says what ran.
2. **Figure paths in `figures.json` are absolute.** The block emits paths relative to
   the Run directory, or the Markdown breaks the moment the Run is moved and pandoc
   cannot resolve the images.

And one simplification: the spec's "one exception, a Run where every Candidate failed"
needs no code. `status._next_stage` already returns `"report"` when nothing succeeded
and the re-plan round is spent, and `"plan"` while the round is unspent — which is the
right answer in both cases. `report` checks one condition and has no special case.

---

## Task 1: The fence format

Pure text handling. No artefacts, no CLI. This is the piece every later task builds on,
so it is worth having exactly right before anything reads a Run.

**Files:**
- Create: `drtools/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces:
  - `BLOCK_IDS: tuple[str, ...]` — the eight ids in section order.
  - `fence(block_id: str, body: str) -> str` — the complete fenced text, opening
    comment with digest, body, closing comment.
  - `parse_blocks(document: str) -> dict[str, ParsedBlock]` — every fence found, keyed
    by id.
  - `ParsedBlock` — a frozen dataclass with `id: str`, `body: str`, `digest: str`
    (the digest recorded in the fence), `start: int`, `end: int` (character offsets of
    the whole fenced region, opening comment through closing comment).
  - `edited(block: ParsedBlock) -> bool` — True when the body no longer hashes to the
    digest the fence records.
  - `replace_block(document: str, block_id: str, body: str) -> str` — returns the
    document with that block's fenced region replaced; raises `KeyError` if absent.

- [ ] **Step 1: Write the failing tests**

```python
"""The fenced regions the toolbox owns inside a document the agent also writes."""

from __future__ import annotations

import pytest

from drtools.report import (
    BLOCK_IDS,
    edited,
    fence,
    parse_blocks,
    replace_block,
)


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
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: collection error — `No module named 'drtools.report'`.

- [ ] **Step 3: Write `drtools/report.py` as far as the fence**

```python
"""The report: what the toolbox writes into it, and what it must never touch.

The document has two owners. Every number belongs to the toolbox and lives inside a
fenced block; everything else is the agent's prose. The split exists so that the agent
never retypes a value -- a figure in the report cannot disagree with the artefact it
came from, because there was no opportunity to transcribe it.

The fence carries a digest of the body the toolbox last wrote. That is what lets a
refresh tell "this block is mine to regenerate" from "the agent edited this", and
refuse the second rather than silently destroying the edit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

BLOCK_IDS = (
    "profile",
    "preprocessing",
    "methods",
    "hyperparameters",
    "figures",
    "metrics",
    "ranking",
    "limitations",
)
"""The eight generated blocks, in section order.

Section 8, Interpretation, is absent deliberately: nothing in the run grounds it, and a
generated block there would lend the appearance of derivation to the one section that
is entirely the agent's judgment.
"""

_OPEN = re.compile(
    r"<!-- drtools:(?P<id>[a-z_]+) sha256=(?P<digest>[0-9a-f]{16}) -->\r?\n"
)
_DIGEST_CHARS = 16


def _digest(body: str) -> str:
    """Over the body as written, with line endings normalised.

    Normalised because git rewrites them on checkout in this repo, and a block that
    read as hand-edited after a clone would make `--refresh` useless on a fresh
    checkout.
    """
    return hashlib.sha256(
        body.replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()[:_DIGEST_CHARS]


@dataclass(frozen=True)
class ParsedBlock:
    id: str
    body: str
    digest: str
    start: int
    end: int


def fence(block_id: str, body: str) -> str:
    """One complete fenced region: opening comment with digest, body, closing comment."""
    return (
        f"<!-- drtools:{block_id} sha256={_digest(body)} -->\n"
        f"{body}\n"
        f"<!-- /drtools:{block_id} -->"
    )


def parse_blocks(document: str) -> dict[str, ParsedBlock]:
    """Every fenced region in the document, keyed by id.

    The closing comment is matched by id rather than by a generic pattern, so a body
    that happens to contain the word `drtools` does not end the block early.
    """
    blocks: dict[str, ParsedBlock] = {}
    for match in _OPEN.finditer(document):
        block_id = match.group("id")
        closing = f"<!-- /drtools:{block_id} -->"
        close_at = document.find(closing, match.end())
        if close_at == -1:
            continue
        body = document[match.end() : close_at].rstrip("\r\n")
        blocks[block_id] = ParsedBlock(
            id=block_id,
            body=body,
            digest=match.group("digest"),
            start=match.start(),
            end=close_at + len(closing),
        )
    return blocks


def edited(block: ParsedBlock) -> bool:
    """Whether the body no longer matches the digest the toolbox wrote beside it."""
    return _digest(block.body) != block.digest


def replace_block(document: str, block_id: str, body: str) -> str:
    """Swap one block's fenced region, leaving every other byte of the document alone."""
    blocks = parse_blocks(document)
    if block_id not in blocks:
        raise KeyError(block_id)
    block = blocks[block_id]
    return document[: block.start] + fence(block_id, body) + document[block.end :]
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add drtools/report.py tests/test_report.py
git commit -m "Add the fenced block format the report is built from"
```

---

## Task 2: The block generators

Every block's body, from the Run's artefacts. Still no CLI: this task is about the
content being correct and traceable.

**Files:**
- Modify: `drtools/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `BLOCK_IDS` from Task 1.
- Produces:
  - `build_blocks(run: RunDir) -> dict[str, str]` — a body for every id in
    `BLOCK_IDS`. Never raises for an absent source; see `NOT_PRODUCED`.
  - `NOT_PRODUCED: str` — the line a block prints when its source does not exist, so
    that "the Run did not produce this" stays distinct from "the agent has not reached
    this section".

- [ ] **Step 1: Write the failing tests**

A shared fixture builds one complete Run. It is slow, so it is session-scoped and every
content test reads it. **It goes in `tests/conftest.py`, not in the test module**,
because `tests/test_render.py` needs the same Run in Task 5 and building it twice
doubles the suite's runtime. `_plan_document` goes there with it.

```python
import json

from drtools.report import NOT_PRODUCED, build_blocks
from drtools.runs import RunDir

PCA = [{"op": "pca", "params": {"n_components": 2}}]


def _plan_document(**overrides):
    plan = {
        "dataset": "blobs",
        "budget": "fast",
        "base_preprocessing": [{"op": "standardise", "params": {}}],
        "candidates": [
            {"id": "pca-2", "stages": PCA, "rationale": "linear baseline, always included"}
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
    """One complete Run: profile, recon, plan, embed, evaluate, rank, figures."""
    from drtools.cli import main

    root = tmp_path_factory.mktemp("runs")
    run_dir = root / "r1"
    assert main(["profile", "--data", "blobs", "--runs-root", str(root), "--run-id", "r1"]) == 0
    assert main(["recon", "--run-dir", str(run_dir)]) == 0
    (run_dir / "plan.json").write_text(json.dumps(_plan_document()), encoding="utf-8")
    assert main(["validate-plan", "--run-dir", str(run_dir)]) == 0
    assert main(["prepare-reference", "--run-dir", str(run_dir)]) == 0
    assert main(["embed", "--run-dir", str(run_dir), "--id", "pca-2", "--in-process"]) == 0
    assert main(["evaluate", "--run-dir", str(run_dir), "--id", "pca-2"]) == 0
    assert main(["rank", "--run-dir", str(run_dir)]) == 0
    assert main(["figures", "--run-dir", str(run_dir)]) == 0
    return RunDir(run_dir)


def test_every_declared_block_is_generated(finished_run):
    blocks = build_blocks(finished_run)
    assert set(blocks) == set(BLOCK_IDS)


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


def test_figure_paths_are_relative_to_the_run(finished_run):
    """figures.json stores absolute paths; a report holding one breaks when moved."""
    body = build_blocks(finished_run)["figures"]

    assert "figures/comparison.png" in body.replace("\\", "/")
    assert str(finished_run.path) not in body


def test_a_block_whose_source_is_absent_says_so(tmp_path):
    """Absent must not read as a section the agent has not reached yet."""
    from drtools.cli import main

    root = tmp_path / "runs"
    assert main(["profile", "--data", "blobs", "--runs-root", str(root), "--run-id", "bare"]) == 0

    blocks = build_blocks(RunDir(root / "bare"))

    assert blocks["ranking"] == NOT_PRODUCED
    assert blocks["figures"] == NOT_PRODUCED
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: `ImportError: cannot import name 'build_blocks'`.

- [ ] **Step 3: Implement `build_blocks`**

One private function per block id, each returning a Markdown body. `ranking` is the
most intricate and is written out here as the pattern the other seven follow — read an
artefact, guard its absence, print at the artefact's own precision, and carry every
qualification the artefact already holds:

```python
NOT_PRODUCED = "_This Run did not produce this._"


def _block_ranking(run: RunDir) -> str:
    path = run.path / "ranking.json"
    if not path.exists():
        return NOT_PRODUCED
    ranking = jsonio.read(path)

    lines = ["| rank | candidate | score | " + " | ".join(ranking["weights_applied"]) + " |"]
    lines.append("|---" * (3 + len(ranking["weights_applied"])) + "|")
    for entry in ranking["ranking"]:
        contributions = [
            f"{entry['contributions'][metric]['contribution']:.4f}"
            if metric in entry["contributions"]
            else "—"
            for metric in ranking["weights_applied"]
        ]
        lines.append(
            f"| {entry['rank']} | {entry['id']} | {entry['score']:.4f} | "
            + " | ".join(contributions)
            + " |"
        )

    lines += ["", f"Winner: **{ranking['winner']}**.", ""]
    lines.append("Weighting declared before any Embedding existed: " + ", ".join(
        f"{metric} {weight}" for metric, weight in ranking["weights_declared"].items()
    ) + ".")
    if ranking["weights_declared"] != ranking["weights_applied"]:
        lines.append("Weighting actually applied: " + ", ".join(
            f"{metric} {weight}" for metric, weight in ranking["weights_applied"].items()
        ) + ".")
    for metric, reason in ranking["weights_dropped"].items():
        lines.append(f"- `{metric}` was dropped from the score: {reason}")
    if ranking["failed_candidates"]:
        lines.append(
            "Candidates that produced no Embedding, and are therefore unscored: "
            + ", ".join(ranking["failed_candidates"]) + "."
        )
    # Verbatim. These are the qualifications `rank` computed -- a tie inside the noise,
    # a runtime span that dominates the comparison -- and paraphrasing them here would
    # be the report making a claim the toolbox did not.
    lines += [""] + [f"- {note}" for note in ranking["notes"]]
    return "\n".join(lines)
```

The remaining seven follow the same shape. Their contents:

- Read artefacts through `jsonio.read`, guarding existence with `Path.exists()`; a
  missing source returns `NOT_PRODUCED` rather than raising.
- Numbers print at the precision the artefact holds, with metric values at `:.4f` and
  scores at `:.4f`. Never round a value into a claim the artefact does not support.
- Figure paths: `Path(record["path"]).relative_to(run.path).as_posix()`, emitted as
  `![caption](figures/name.png)`. A figure record whose path is not under the Run is
  skipped and named in the block, since it cannot be embedded.
- `profile`: a two-column table of shape, storage, sparsity, label kind and class
  count, plus `dataset_digest[:12]` and, when `data/meta.json` has an `adapter` key,
  the adapter path and its `sha256[:12]`.
- `preprocessing`: the registered `base_preprocessing` stages as a list, then every
  `decisions.jsonl` record with `stage == "plan"`, each printing `chosen`, `rationale`
  and its `evidence_resolved` pairs.
- `methods`: a table of registered Candidates with their stage ops and `rationale`,
  then a table of `plan.registered.json["rejected"]` with `method`, `reason` and
  `evidence`. The Rejections table prints even when empty, with a line saying no
  method was rejected — an empty section here would read as an omission, and section 3
  is the one the design calls the highest-value part of the document.
- `hyperparameters`: per Candidate, from `embeddings/<id>.json`, every stage's `params`
  with the `param_provenance` value beside each. Candidates with no embedding record
  are listed as not run.
- `figures`: one image per record in `figures/figures.json`, with the caveat text where
  the record carries one (`recon_thumbnail.caveat`), and the `identity_channel` where
  it is not `"labels"`.
- `metrics`: a table of `metrics/<id>.json` `values` per Candidate, with a footnote row
  marking any metric in `ranking.json["weights_dropped"]` as excluded from the score.
- `limitations`: the mechanical ones only — each dropped metric and the weight that
  moved with it, each note in `ranking.json["notes"]` that concerns a tie, and for each
  Candidate whose `metrics/<id>.json` has `subsampled: true`, the `n_used` of `n_total`
  the metrics therefore describe.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add drtools/report.py tests/test_report.py
git commit -m "Generate every block's body from the artefacts that hold its numbers"
```

---

## Task 3: `drtools report`

The first emit: the nine-section skeleton with the eight blocks in place.

**Files:**
- Modify: `drtools/report.py`, `drtools/cli.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `build_blocks`, `fence` from Tasks 1–2.
- Produces:
  - `SECTIONS: tuple[tuple[str, str | None], ...]` — `(heading, block_id or None)` in
    order, the nine headings matching `skills/write-report/SKILL.md`.
  - `assemble(run: RunDir) -> str` — the whole document.
  - `_cmd_report(args)` in `cli.py`, returning
    `{"path": str, "blocks": [...], "written": True}`.

- [ ] **Step 1: Write the failing tests**

```python
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
    assert (finished_run.path / "report.md").read_text(encoding="utf-8") == "the agent's prose\n"
```

**Two notes for the implementer.**

`finished_run` is session-scoped and shared with `tests/test_render.py`, so a test that
writes `report.md` leaves it there for the next one. Add this to `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def _clear_report(request):
    """A shared Run must look untouched to every test that uses it."""
    yield
    if "finished_run" in request.fixturenames:
        for name in ("report.md", "report.pdf"):
            (request.getfixturevalue("finished_run").path / name).unlink(missing_ok=True)
```

`_cmd_report` below calls `_refresh_report`, which Task 4 defines. Python resolves the
name at call time and nothing in Task 3 passes `--refresh`, so the tests pass; leave
the call in place rather than stubbing it.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q -k report_`
Expected: `invalid choice: 'report'` from argparse, surfacing as exit 3.

- [ ] **Step 3: Implement `assemble` and the handler**

In `drtools/report.py`:

```python
SECTIONS: tuple[tuple[str, str | None], ...] = (
    ("1. Dataset profile", "profile"),
    ("2. Preprocessing decisions, and why", "preprocessing"),
    ("3. Methods selected and rejected", "methods"),
    ("4. Hyperparameter choices, and why", "hyperparameters"),
    ("5. Figures", "figures"),
    ("6. Quantitative comparison", "metrics"),
    ("7. Ranking, with the weighting justification", "ranking"),
    ("8. Interpretation", None),
    ("9. Limitations", "limitations"),
)
"""The skeleton, matching `skills/write-report/SKILL.md` heading for heading.

Fixed so that the two generated reports can be read side by side. Section 8 carries no
block id, which is the whole shape of the split.
"""

_PROMPT = "_Yours to write. Delete this line._"


def assemble(run: RunDir) -> str:
    bodies = build_blocks(run)
    parts = [f"# Dimension reduction report — {run.id}", ""]
    for heading, block_id in SECTIONS:
        parts += [f"## {heading}", ""]
        if block_id is not None:
            parts += [fence(block_id, bodies[block_id]), ""]
        parts += [_PROMPT, ""]
    return "\n".join(parts)
```

In `drtools/cli.py`, beside the other handlers:

```python
def _cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    """Emit the report skeleton, or refresh the blocks in one that exists.

    The readiness condition is `status`'s, not a second copy of it. Two implementations
    of one question drift at the edges, and the failure that produces is a run where
    `status` says the next stage is `report` and `report` says the run is not ready,
    with nothing to tell the agent which is right.
    """
    run = _require_run(args)
    path = run.path / "report.md"

    if args.refresh:
        return _refresh_report(run, path)   # Task 4

    stage = run_status(run)["next"]
    if stage != "report":
        raise ContractError(
            f"this run's next stage is {stage}, not report, so a report now would "
            "describe an analysis that has not finished. Run `drtools status "
            f"--run-dir {run.path}` and complete that stage first."
        )
    if path.exists():
        raise ContractError(
            f"{path} already exists, and writing it again would discard the prose "
            "around the generated blocks. Use `--refresh` to bring the blocks up to "
            "date and leave everything else alone."
        )

    path.write_text(assemble(run), encoding="utf-8")
    return {"path": str(path), "blocks": list(BLOCK_IDS), "written": True}
```

Subparser, following the shape of `figures`:

```python
    report = subparsers.add_parser(
        "report",
        help="emit the report skeleton with every number already in it",
    )
    _add_run_arguments(report)
    report.add_argument(
        "--refresh",
        action="store_true",
        help="rewrite the generated blocks of an existing report.md in place, "
        "leaving the prose around them untouched",
    )
    report.set_defaults(handler=_cmd_report)
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add drtools/report.py drtools/cli.py tests/test_report.py
git commit -m "Emit the nine-section report with every number already in it"
```

---

## Task 4: `drtools report --refresh`

**Files:**
- Modify: `drtools/report.py`, `drtools/cli.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `parse_blocks`, `edited`, `replace_block`, `build_blocks` from Tasks 1–2.
- Produces:
  - `stale_blocks(run: RunDir, document: str) -> list[str]` — ids whose generated body
    differs from what the document holds. Task 5 uses this.
  - `_refresh_report(run, path)` in `cli.py`, returning
    `{"path": str, "refreshed": [...], "unchanged": [...], "missing": [...]}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_refresh_updates_a_block_and_leaves_the_prose_alone(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    document = path.read_text(encoding="utf-8").replace(
        "_Yours to write. Delete this line._",
        "The winner is clear on the neighbourhood metrics.",
        1,
    )
    path.write_text(document, encoding="utf-8")

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    after = path.read_text(encoding="utf-8")
    assert "The winner is clear on the neighbourhood metrics." in after
    assert set(parse_blocks(after)) == set(BLOCK_IDS)


def test_refresh_rewrites_a_block_the_run_has_moved_past(cli, finished_run, tmp_path):
    """The case the whole mechanism exists for: a number changed under the document."""
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    stale = parse_blocks(path.read_text(encoding="utf-8"))["ranking"].body
    # Rank again under the same registered weighting; the file is rewritten and the
    # block in the document is now a copy of an older ranking.json.
    path.write_text(
        replace_block(path.read_text(encoding="utf-8"), "ranking", "an older ranking"),
        encoding="utf-8",
    )

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert "ranking" in result.payload["refreshed"]
    assert parse_blocks(path.read_text(encoding="utf-8"))["ranking"].body == stale


def test_refresh_refuses_a_block_the_agent_edited(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    path = finished_run.path / "report.md"
    document = path.read_text(encoding="utf-8")
    blocks = parse_blocks(document)
    tampered = (
        document[: blocks["ranking"].start]
        + document[blocks["ranking"].start : blocks["ranking"].end].replace(
            "| rank |", "| rank (my ordering) |", 1
        )
        + document[blocks["ranking"].end :]
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
    blocks = parse_blocks(document)
    without = document[: blocks["figures"].start] + document[blocks["figures"].end :]
    path.write_text(without, encoding="utf-8")

    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    assert result.payload["missing"] == ["figures"]
    assert "figures" not in parse_blocks(path.read_text(encoding="utf-8"))


def test_refresh_refuses_when_there_is_no_report_yet(cli, finished_run):
    result = cli("report", "--refresh", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "drtools report" in result.stderr
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q -k refresh`
Expected: `AttributeError` or `NameError` on `_refresh_report`.

- [ ] **Step 3: Implement the refresh**

In `drtools/report.py`:

```python
def stale_blocks(run: RunDir, document: str) -> list[str]:
    """Ids whose generated body no longer matches what the document holds.

    Used by `--refresh` to decide what to rewrite, and by `render` to decide whether
    the document still agrees with the run. One function, so the two commands cannot
    disagree about what stale means.
    """
    bodies = build_blocks(run)
    present = parse_blocks(document)
    return sorted(
        block_id
        for block_id, block in present.items()
        if block_id in bodies and block.body != bodies[block_id]
    )
```

In `drtools/cli.py`:

```python
def _refresh_report(run: RunDir, path: Path) -> dict[str, Any]:
    """Rewrite the generated blocks of an existing report, and nothing else.

    A block the agent has edited is refused rather than rewritten. Regenerating it
    would destroy that edit as a side effect of an unrelated request, which is the
    defect found in `embed` on day 9 -- and the whole reason the fence carries a digest.
    """
    if not path.exists():
        raise ContractError(
            f"there is no report at {path} to refresh. Run `drtools report --run-dir "
            f"{run.path}` to write one."
        )

    document = path.read_text(encoding="utf-8")
    blocks = parse_blocks(document)
    hand_edited = sorted(block_id for block_id, block in blocks.items() if edited(block))
    if hand_edited:
        raise ContractError(
            f"block(s) {hand_edited} have been edited by hand since the toolbox wrote "
            "them, and refreshing would discard those edits. The blocks hold the run's "
            "numbers and the toolbox owns them; prose belongs outside the fence. Move "
            "the edit outside the block, and the refresh will go through."
        )

    bodies = build_blocks(run)
    refreshed, unchanged = [], []
    for block_id in BLOCK_IDS:
        if block_id not in blocks:
            continue
        if blocks[block_id].body == bodies[block_id]:
            unchanged.append(block_id)
            continue
        document = replace_block(document, block_id, bodies[block_id])
        refreshed.append(block_id)

    path.write_text(document, encoding="utf-8")
    return {
        "path": str(path),
        "refreshed": refreshed,
        "unchanged": unchanged,
        "missing": [b for b in BLOCK_IDS if b not in blocks],
    }
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add drtools/report.py drtools/cli.py tests/test_report.py
git commit -m "Refresh the blocks in place, and refuse one the agent edited"
```

---

## Task 5: `drtools render`

**Files:**
- Create: `drtools/render.py`, `tests/test_render.py`
- Modify: `drtools/cli.py`

**Interfaces:**
- Consumes: `stale_blocks` from Task 4.
- Produces:
  - `render_pdf(source: Path, destination: Path) -> dict[str, Any]` in `render.py`,
    raising `ContractError` for a missing pandoc or a missing PDF engine.
  - `_cmd_render(args)` in `cli.py`, returning `{"path": str, "pages": None}`.

- [ ] **Step 1: Write the failing tests**

`monkeypatch` is how absence is simulated; both tools are installed in this
environment, so the refusals cannot be reached any other way.

```python
"""Rendering, and the three things that stop it."""

import shutil

import pytest

from drtools.report import parse_blocks, replace_block


def test_render_produces_a_pdf(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)

    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 0, result.stderr
    pdf = finished_run.path / "report.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1000


def test_render_refuses_a_document_the_run_has_moved_past(cli, finished_run):
    """Rendering last is a property of the toolbox, not an instruction in a skill."""
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
    """Two installs, two fixes; one message for both would send the agent to the wrong one."""
    cli("report", "--run-dir", finished_run.path)
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/pandoc" if name == "pandoc" else None
    )

    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "pdflatex" in result.stderr or "PDF engine" in result.stderr
    assert "install pandoc" not in result.stderr.lower()


def test_render_refuses_when_there_is_no_report(cli, finished_run):
    result = cli("render", "--run-dir", finished_run.path)

    assert result.code == 2
    assert "drtools report" in result.stderr
```

`finished_run` and `_clear_report` are already in `tests/conftest.py` from Tasks 2 and
3, so this module needs no fixtures of its own. Run the whole suite afterwards, not
just the new file — moving nothing means nothing to break, but `test_skills.py` now
sees two new commands.

- [ ] **Step 2: Run the tests and watch them fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_render.py -q`
Expected: `invalid choice: 'render'`.

- [ ] **Step 3: Implement `drtools/render.py` and the handler**

```python
"""Markdown to PDF, and the two installs that have to be there.

pandoc and a PDF engine are separate programs with separate fixes, and pandoc's own
error for the second is about a missing engine rather than a missing program. Naming
them separately is the difference between an agent installing the right thing and
installing pandoc again.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from drtools.contract import ContractError

PDF_ENGINES = ("pdflatex", "xelatex", "lualatex", "tectonic", "typst", "weasyprint")
"""Engines pandoc can drive, in the order it prefers them."""


def render_pdf(source: Path, destination: Path) -> dict[str, Any]:
    if shutil.which("pandoc") is None:
        raise ContractError(
            "pandoc is not on PATH, and it is what turns the report into a PDF. "
            "Install it from https://pandoc.org/installing.html. The Markdown at "
            f"{source} is the source of truth and is already complete; the PDF is a "
            "rendering of it."
        )

    engine = next((name for name in PDF_ENGINES if shutil.which(name)), None)
    if engine is None:
        raise ContractError(
            "pandoc is installed but no PDF engine is. pandoc renders Markdown; a "
            f"separate program makes the PDF. Install one of {', '.join(PDF_ENGINES)} "
            "-- a TeX distribution such as MiKTeX or TeX Live provides pdflatex."
        )

    completed = subprocess.run(
        ["pandoc", str(source), "-o", str(destination), f"--pdf-engine={engine}"],
        capture_output=True,
        text=True,
        cwd=str(source.parent),
    )
    if completed.returncode != 0:
        raise ContractError(
            f"pandoc could not render {source.name} with {engine}: "
            f"{completed.stderr.strip()[:500]}"
        )
    return {"path": str(destination), "engine": engine}
```

The handler runs the staleness check before touching pandoc, so an out-of-date document
is refused for the reason that matters rather than for an environment reason:

```python
def _cmd_render(args: argparse.Namespace) -> dict[str, Any]:
    """Render the report, and refuse one whose numbers the run has moved past.

    Rendering does not refresh on the agent's behalf. Silently changing the document's
    numbers while producing the PDF would let the two artefacts a reader compares
    differ, and the Markdown is the declared source of truth. So it refuses and names
    the route, which makes rendering last a property of the toolbox rather than an
    instruction in a skill.
    """
    run = _require_run(args)
    source = run.path / "report.md"
    if not source.exists():
        raise ContractError(
            f"there is no report at {source} to render. Run `drtools report --run-dir "
            f"{run.path}` to write one."
        )

    stale = stale_blocks(run, source.read_text(encoding="utf-8"))
    if stale:
        raise ContractError(
            f"block(s) {stale} no longer match this run, so the PDF would carry "
            "numbers the run has moved past. Run `drtools report --refresh --run-dir "
            f"{run.path}` and render again."
        )

    return render_pdf(source, run.path / "report.pdf")
```

Subparser:

```python
    render = subparsers.add_parser(
        "render", help="render report.md to report.pdf, refusing a stale one"
    )
    _add_run_arguments(render)
    render.set_defaults(handler=_cmd_render)
```

- [ ] **Step 4: Run the whole suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: every test passes, including `tests/test_skills.py`, which scans the skills
for `drtools <command> --flag` and will now see `report` and `render` as valid.

- [ ] **Step 5: Commit**

```bash
git add drtools/render.py drtools/cli.py tests/test_render.py
git commit -m "Render the report, and refuse a document the run has moved past"
```

---

## Task 6: The skill and the record

The toolbox is finished; the agent still does not know the commands exist.

**Files:**
- Modify: `skills/write-report/SKILL.md`, `design/notes.md`
- Test: `tests/test_skills.py` (no change needed; it scans automatically)

- [ ] **Step 1: Update the skill**

Replace the closing line of the "Fixed sections" section — currently "Markdown in the
Run directory is the source of truth; rendering to PDF is a separate step." — with a
section naming both commands, the order they run in, and what each refuses:

```markdown
## How the document is produced

```
drtools report --run-dir runs/<id>
```

writes `report.md` with the nine sections and, inside fenced blocks, every number this
Run produced. **You never type a number into the report.** The blocks are the toolbox's;
everything outside them is yours.

Write your prose around the blocks. If the Run changes — another Candidate evaluated, a
re-rank — bring the numbers up to date with:

```
drtools report --refresh --run-dir runs/<id>
```

which rewrites only the blocks and leaves every word you wrote. It refuses if you have
edited inside a block, because regenerating it would discard your edit; move the edit
outside the fence and it goes through.

Last:

```
drtools render --run-dir runs/<id>
```

It refuses a `report.md` whose blocks no longer match the Run, so the PDF can only be
made from a document that still agrees with the analysis. Refresh, then render.

Section 8, Interpretation, has no block. Nothing in the Run grounds it — it is yours
alone.
```

- [ ] **Step 2: Run the skills test**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_skills.py -q`
Expected: passes. If it fails on a flag name, the skill and the parser disagree — fix
the skill, not the test.

- [ ] **Step 3: Append the day 10 decision log entry**

Append to `design/notes.md` under the decision log, keyed `- **Day 10** —`, covering:
the classification and why (the notes fixed the requirement, not the mechanism);
the decision and the three rejected alternatives, summarised from the spec;
the two spec corrections found while reading real artefacts (hyperparameters come from
the embedding record, not the Plan; figure paths are absolute and must be relativised);
that the all-failed exception needed no code because `status` already returns `report`
in that case; and the final test count.

- [ ] **Step 4: Run the whole suite and commit**

```bash
./.venv/Scripts/python.exe -m pytest -q
git add skills/write-report/SKILL.md design/notes.md
git commit -m "Name the report commands where the agent reads them"
```

---

## What this plan does not cover

The second half of day 10 — the first end-to-end run on PBMC3k — is not in this plan.
It depends on this work and on one decision the spec deliberately leaves open: whether
PBMC3k gets derived Leiden reference labels. That decision is taken when the run
starts, not here.
