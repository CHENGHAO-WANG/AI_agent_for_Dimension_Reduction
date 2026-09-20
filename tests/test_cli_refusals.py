"""Ordinary command sequences, and what the toolbox does when one of them goes wrong.

Everything here is about the shape of the failure rather than about a contract: an
autonomous agent loops on whatever it is told, so a traceback with no exit code it can
read, or an `[Errno 2]` naming a path instead of a step, is a wedge. Each of these used
to be one of those.
"""

import json

PCA = [{"op": "pca", "params": {"n_components": 2}}]


def _plan(**overrides):
    plan = {
        "dataset": "d",
        "candidates": [{"id": "a", "stages": PCA}],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    plan.update(overrides)
    return plan


def _profiled(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    data = csv_dataset(rows=60, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1")
    return runs / "r1", data


# ------------------------------------------------- re-profiling must not wedge embed


def test_re_profiling_after_registering_leaves_embed_working(cli, csv_dataset, tmp_path):
    """profile -> validate-plan -> profile -> embed.

    Nothing unusual: the agent re-measures a run it has already planned. `profile`
    rebuilt the manifest from scratch, dropping the `plan_digest` `validate-plan` had
    written, and `embed` read that key straight out of the manifest — so the sequence
    ended in an uncaught KeyError, exit 1, no message and no route, and every later
    `embed` in the run stayed wedged until `validate-plan` was run again.
    """
    run, data = _profiled(cli, csv_dataset, tmp_path)
    (run / "plan.json").write_text(json.dumps(_plan()), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", run).code == 0

    reprofile = cli("profile", "--data", data, "--run-dir", run)
    assert reprofile.code == 0, reprofile.stderr

    result = cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    assert result.code == 0, result.stderr
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    attempt = [r for r in records if r["stage"] == "embed"][-1]
    registration = [r for r in records if r["stage"] == "register_plan"][-1]
    # And the digest it stamped is the registered plan's, not whatever the manifest
    # happened to be carrying.
    assert attempt["plan_digest"] == registration["plan_digest"]


def test_re_profiling_preserves_the_registered_digest_on_the_manifest(
    cli, csv_dataset, tmp_path
):
    run, data = _profiled(cli, csv_dataset, tmp_path)
    (run / "plan.json").write_text(json.dumps(_plan()), encoding="utf-8")
    cli("validate-plan", "--run-dir", run)
    before = json.loads((run / "run.json").read_text(encoding="utf-8"))["plan_digest"]

    cli("profile", "--data", data, "--run-dir", run)

    after = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert after["plan_digest"] == before


# -------------------------------------------------- a mistyped path is not a new run


def test_validate_plan_refuses_a_run_directory_that_does_not_exist(cli, tmp_path):
    result = cli("validate-plan", "--run-dir", tmp_path / "typo")
    assert result.code == 2
    assert "no run at" in result.stderr
    assert not (tmp_path / "typo").exists()


def test_suggest_params_refuses_a_run_directory_that_does_not_exist(cli, tmp_path):
    result = cli("suggest-params", "--run-dir", tmp_path / "typo", "--op", "pca")
    assert result.code == 2
    assert "no run at" in result.stderr
    assert not (tmp_path / "typo").exists()


def test_figures_refuses_a_run_directory_that_does_not_exist(cli, tmp_path):
    result = cli("figures", "--run-dir", tmp_path / "typo")
    assert result.code == 2
    assert "no run at" in result.stderr
    assert not (tmp_path / "typo").exists()


def test_a_typo_does_not_turn_rank_into_the_wrong_diagnosis(cli, tmp_path):
    """The downstream damage: an empty tree made `rank` report a missing plan.

    Which sends the agent to `validate-plan` for a run that was never created, instead
    of to the typo in the path it just typed.
    """
    cli("validate-plan", "--run-dir", tmp_path / "typo")
    result = cli("rank", "--run-dir", tmp_path / "typo")
    assert "no run at" in result.stderr
    assert "no registered plan" not in result.stderr


# -------------------------------------------------- refusals that were OS errors


def test_evaluating_a_candidate_that_was_never_embedded_names_the_missing_step(
    cli, csv_dataset, tmp_path
):
    run, _ = _profiled(cli, csv_dataset, tmp_path)
    (run / "plan.json").write_text(
        json.dumps(_plan(candidates=[{"id": "a", "stages": PCA},
                                     {"id": "b", "stages": PCA}])),
        encoding="utf-8",
    )
    cli("validate-plan", "--run-dir", run)
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")

    result = cli("evaluate", "--run-dir", run, "--id", "b")

    assert result.code == 2
    assert "contract error:" in result.stderr
    assert "embed" in result.stderr
    assert "Errno" not in result.stderr


def test_figures_on_a_run_with_no_cache_names_profile(cli, tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)

    result = cli("figures", "--run-dir", run)

    assert result.code == 2
    assert "profile" in result.stderr
    assert "Errno" not in result.stderr


# ----------------------------------------------------------- a malformed plan.json


def test_a_malformed_plan_is_refused_with_the_field_that_is_wrong(
    cli, csv_dataset, tmp_path
):
    """plan.json is the artefact the agent edits most, so a typo in it must be readable.

    Pydantic's ValidationError is neither a ContractError nor a FileNotFoundError, so it
    propagated straight out of `main` as a traceback.
    """
    run, _ = _profiled(cli, csv_dataset, tmp_path)
    (run / "plan.json").write_text(
        json.dumps({"dataset": "d", "candidates": [{"id": "a", "stages": PCA}],
                    "evaluation": {"weight": {"trustworthiness": 1.0}}}),
        encoding="utf-8",
    )

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "contract error:" in result.stderr
    assert "evaluation" in result.stderr
    assert not (run / "plan.registered.json").exists()


def test_a_candidate_with_no_stages_is_refused_rather_than_raising(
    cli, csv_dataset, tmp_path
):
    run, _ = _profiled(cli, csv_dataset, tmp_path)
    (run / "plan.json").write_text(
        json.dumps(_plan(candidates=[{"id": "a", "stages": []}])), encoding="utf-8"
    )

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "at least one stage" in result.stderr


# ------------------------------------------ malformed JSON passed on the command line


def test_a_malformed_plan_argument_is_refused_rather_than_raising(
    cli, csv_dataset, tmp_path
):
    """`--plan '{bad'` used to leave a JSONDecodeError traceback and exit 1.

    `_plan_from` already closed this for pydantic's ValidationError, which is raised
    once the document has parsed. A document that never parses is raised a step
    earlier, by `json.loads`, and reached `main` uncaught -- so the agent read a
    traceback and an exit code that is none of the four the toolbox documents.
    """
    run, _ = _profiled(cli, csv_dataset, tmp_path)

    result = cli("validate-plan", "--run-dir", run, "--plan", "{not json")

    assert result.code == 2
    assert "contract error:" in result.stderr
    assert "--plan" in result.stderr


def test_a_malformed_decision_argument_is_refused_rather_than_raising(
    cli, csv_dataset, tmp_path
):
    """The same parse, on the agent's only write route into the decision log."""
    run, _ = _profiled(cli, csv_dataset, tmp_path)

    result = cli("log-decision", "--run-dir", run, "--json", "{not json")

    assert result.code == 2
    assert "contract error:" in result.stderr
    assert "--json" in result.stderr


def test_a_malformed_json_file_names_the_file_rather_than_the_flag(
    cli, csv_dataset, tmp_path
):
    """`@path` is a second source, and the message has to say which one was read."""
    run, _ = _profiled(cli, csv_dataset, tmp_path)
    document = tmp_path / "decision.json"
    document.write_text("{not json", encoding="utf-8")

    result = cli("log-decision", "--run-dir", run, "--json", f"@{document}")

    assert result.code == 2
    assert "contract error:" in result.stderr
    assert str(document) in result.stderr
