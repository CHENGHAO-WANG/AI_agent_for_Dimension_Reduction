"""The suggestion crossing the boundary between measuring and analysing.

A Probe representation exists so measurements describe the data rather than an artefact
of scale, is chosen by a fixed published rule, and is discarded once the measuring is
done — it never produces an Embedding. Base preprocessing is part of the analysis, is
chosen by the agent, and its output survives as the Reference. Day 6 decided
deliberately that these are two things and that collapsing them would be the wrong fix.

But both answer "what transform makes distances on this data meaningful", so the rule
that settles one is the honest default for the other. One implementation of that rule,
not a prose copy in the planning skill that drifts from it.
"""

from __future__ import annotations

import json


def _run(cli, tmp_path, data, run_id="r1"):
    runs = tmp_path / "runs"
    assert cli("profile", "--data", data, "--runs-root", runs, "--run-id", run_id).code == 0
    assert cli("recon", "--run-dir", runs / run_id).code == 0
    return runs / run_id


def test_the_suggestion_matches_recons_rule_rather_than_restating_it(
    cli, tmp_path
):
    run = _run(cli, tmp_path, "sparse_counts")
    recon = json.loads((run / "recon.json").read_text(encoding="utf-8"))

    result = cli("suggest-base", "--run-dir", run)

    assert result.code == 0, result.stderr
    assert [stage["op"] for stage in result.payload["stages"]] == (
        recon["probe_representation"]["transform"]
    )


def test_every_suggested_op_is_a_registry_op(cli, tmp_path):
    """A suggestion naming an op the registry does not declare is a plan that cannot
    be registered."""
    from drtools.registry import load_registry

    ops = set(load_registry().ops)
    for dataset in ("sparse_counts", "swiss_roll", "blobs"):
        run = _run(cli, tmp_path, dataset, run_id=f"r-{dataset}")
        result = cli("suggest-base", "--run-dir", run)
        assert result.code == 0, result.stderr
        assert {stage["op"] for stage in result.payload["stages"]} <= ops


def test_the_suggestion_carries_evidence_that_resolves(cli, tmp_path):
    """Its own keys must survive log-decision, or the agent cannot cite the reason it
    was given for adopting it."""
    run = _run(cli, tmp_path, "sparse_counts")
    suggestion = cli("suggest-base", "--run-dir", run).payload

    document = tmp_path / "d.json"
    document.write_text(json.dumps({
        "stage": "plan",
        "question": "What base preprocessing should every candidate share?",
        "chosen": "as suggested",
        "rationale": suggestion["rationale"],
        "evidence": suggestion["evidence"],
    }), encoding="utf-8")

    logged = cli("log-decision", "--run-dir", run, "--json", f"@{document}")
    assert logged.code == 0, logged.stderr


def test_it_says_it_is_a_suggestion_not_the_probe_representation(cli, tmp_path):
    run = _run(cli, tmp_path, "sparse_counts")
    note = cli("suggest-base", "--run-dir", run).payload["note"]
    assert "suggestion" in note.lower()


def test_counts_are_suggested_normalisation_and_log1p(cli, tmp_path):
    run = _run(cli, tmp_path, "sparse_counts")
    result = cli("suggest-base", "--run-dir", run)
    assert [stage["op"] for stage in result.payload["stages"]] == [
        "normalise_total",
        "log1p",
    ]


def test_data_already_on_a_comparable_scale_suggests_no_stages(cli, csv_dataset, tmp_path):
    run = _run(cli, tmp_path, csv_dataset(rows=60, cols=8))
    result = cli("suggest-base", "--run-dir", run)
    assert result.payload["stages"] == []
    assert "as given" in result.payload["rationale"]


def test_a_suggested_plan_registers(cli, tmp_path):
    """End to end: what suggest-base emits must be usable as base_preprocessing.

    A suggestion the validator then refuses would be worse than no suggestion, because
    the agent would have to discover that by being refused.
    """
    run = _run(cli, tmp_path, "sparse_counts")
    suggestion = cli("suggest-base", "--run-dir", run).payload

    plan = {
        "dataset": "sparse_counts",
        "base_preprocessing": suggestion["stages"],
        "candidates": [
            {"id": "pca2", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0
    assert cli("prepare-reference", "--run-dir", run).code == 0


def test_suggest_base_needs_no_recon(cli, csv_dataset, tmp_path):
    """The rule reads the profile. Recon applies the same rule; it does not own it."""
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")

    result = cli("suggest-base", "--run-dir", runs / "r1")

    assert result.code == 0, result.stderr
    assert "stages" in result.payload
