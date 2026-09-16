"""The Budget binds the commands that spend it, or it is decoration.

`Plan.budget` has existed since day 5, typed and defaulting, and nothing in the toolbox
ever read it. A declared resource nothing consumes is not a resource the agent
allocates, and a report describing a run as planned under a budget would be describing
something that never constrained anything.

A default timeout is advisory: any caller may pass a larger one and the run silently
exceeds the budget it declared. A cap refuses.
"""

from __future__ import annotations

import json

from drtools.isolation import budget_timeout

PCA = [{"op": "pca", "params": {"n_components": 2}}]


def _registered(cli, csv_dataset, tmp_path, budget="standard"):
    """A run with a registered two-candidate plan at the given Budget."""
    runs = tmp_path / "runs"
    cli(
        "profile",
        "--data",
        csv_dataset(rows=60, cols=8),
        "--runs-root",
        runs,
        "--run-id",
        "r1",
    )
    plan = {
        "dataset": "d",
        "budget": budget,
        "candidates": [
            {"id": "pca2", "stages": PCA},
            {"id": "pca3", "stages": [{"op": "pca", "params": {"n_components": 3}}]},
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", runs / "r1").code == 0
    return runs / "r1"


def test_the_budget_orders_the_caps():
    assert (
        budget_timeout("fast") < budget_timeout("standard") < budget_timeout("thorough")
    )


def test_status_reports_the_registered_budget(cli, csv_dataset, tmp_path):
    run = _registered(cli, csv_dataset, tmp_path, budget="fast")
    assert cli("status", "--run-dir", run).payload["budget"] == "fast"


def test_embed_refuses_a_timeout_above_the_budgets_cap(cli, csv_dataset, tmp_path):
    run = _registered(cli, csv_dataset, tmp_path, budget="fast")
    cap = budget_timeout("fast")

    result = cli(
        "embed", "--run-dir", run, "--id", "pca2", "--timeout", cap * 2, "--in-process"
    )

    assert result.code == 2
    assert "fast" in result.stderr
    assert f"{cap:g}" in result.stderr


def test_a_refused_timeout_does_not_count_as_an_attempt(cli, csv_dataset, tmp_path):
    """A refusal is not a try. The candidate must still have its full allowance."""
    run = _registered(cli, csv_dataset, tmp_path, budget="fast")
    cli(
        "embed",
        "--run-dir",
        run,
        "--id",
        "pca2",
        "--timeout",
        budget_timeout("fast") * 2,
        "--in-process",
    )

    status = cli("status", "--run-dir", run).payload
    assert status["candidates"]["pca2"]["attempts"] == 0
    assert status["candidates"]["pca2"]["outcome"] is None


def test_embed_accepts_a_timeout_at_the_cap(cli, csv_dataset, tmp_path):
    run = _registered(cli, csv_dataset, tmp_path, budget="standard")

    result = cli(
        "embed",
        "--run-dir",
        run,
        "--id",
        "pca2",
        "--timeout",
        budget_timeout("standard"),
        "--in-process",
    )

    assert result.code == 0, result.stderr


def test_the_budget_cannot_move_once_registered(cli, csv_dataset, tmp_path):
    """Raising it after a candidate has timed out would describe a discipline the run
    did not keep."""
    run = _registered(cli, csv_dataset, tmp_path, budget="fast")
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["budget"] = "thorough"
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "budget" in result.stderr
    assert "new run" in result.stderr


def test_an_unchanged_budget_still_re_registers(cli, csv_dataset, tmp_path):
    """The freeze must bound the budget, not block the retry loop that depends on
    re-registration."""
    run = _registered(cli, csv_dataset, tmp_path, budget="fast")
    assert cli("validate-plan", "--run-dir", run).code == 0
