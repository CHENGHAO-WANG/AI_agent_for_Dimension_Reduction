import json

PCA = [{"op": "pca", "params": {"n_components": 2}}]
# perplexity is kept well below n/3 = 20 for the 60-row dataset these tests profile;
# the default of 30 trips validate-plan's perplexity_too_large check and the plan
# would never reach registration, which is incidental to what this file tests.
TSNE = [{"op": "tsne", "params": {"n_components": 2, "perplexity": 5}}]


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


def _embed(cli, run, candidate):
    cli("embed", "--run-dir", run, "--id", candidate, "--in-process")


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
        _embed(cli, run, candidate)
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
        _embed(cli, run, candidate)
        cli("evaluate", "--run-dir", run, "--id", candidate)

    result = cli("rank", "--run-dir", run)

    assert result.code == 0
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert result.payload["plan_digest"] == manifest["plan_digest"]


def test_no_rank_record_claims_pre_registration_it_cannot_know(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    for candidate in ("a", "b"):
        _embed(cli, run, candidate)
        cli("evaluate", "--run-dir", run, "--id", candidate)
    cli("rank", "--run-dir", run)

    log = (run / "decisions.jsonl").read_text(encoding="utf-8")
    assert "before any embedding was computed" not in log


def test_rereg_after_tampering_refuses(cli, csv_dataset, tmp_path):
    """The full reproduction: register, embed, evaluate, rank — then rewrite the
    weights and re-register, rather than only rewriting and ranking. A run whose
    winner was already computed under the registered weighting must not be able to
    launder a changed weighting back into the record through validate-plan.
    """
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    for candidate in ("a", "b"):
        _embed(cli, run, candidate)
        cli("evaluate", "--run-dir", run, "--id", candidate)
    first_rank = cli("rank", "--run-dir", run)
    assert first_rank.code == 0

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["evaluation"]["weights"] = {"runtime_s": 1.0}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "amendment" in result.stderr.lower()
    registered = json.loads((run / "plan.registered.json").read_text(encoding="utf-8"))
    assert registered["evaluation"]["weights"] == {"trustworthiness": 1.0}
    manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert manifest["plan_digest"] == first_rank.payload["plan_digest"]
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([r for r in records if r["stage"] == "register_plan"]) == 1

    # And ranking still refuses, exactly as before the tampered re-registration
    # attempt — the attempt changed nothing about what is on record.
    second_rank = cli("rank", "--run-dir", run)
    assert second_rank.code == 2


def test_rereg_with_different_base_preprocessing_refuses(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = {
        "dataset": "d",
        "base_preprocessing": [{"op": "standardise", "params": {}}],
        "candidates": [{"id": "a", "stages": PCA}],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "declared"},
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")

    plan["base_preprocessing"] = []
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    result = cli("validate-plan", "--run-dir", runs / "r1")

    assert result.code == 2
    assert "base preprocessing" in result.stderr


def test_rereg_dropping_a_candidate_refuses(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})  # registers a, b
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"] = [c for c in plan["candidates"] if c["id"] != "b"]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "'b'" in result.stderr


def test_reregistering_an_identical_plan_stays_legal(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    result = cli("validate-plan", "--run-dir", run)
    assert result.code == 0
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([r for r in records if r["stage"] == "register_plan"]) == 2


def test_adding_a_new_candidate_to_the_registration_stays_legal(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["candidates"].append({"id": "c", "stages": PCA})
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 0
    registered = json.loads((run / "plan.registered.json").read_text(encoding="utf-8"))
    assert sorted(c["id"] for c in registered["candidates"]) == ["a", "b", "c"]


def test_copying_the_live_plan_over_the_registration_does_not_launder_it(
    cli, csv_dataset, tmp_path
):
    """The file copy that defeated pre-registration.

    `rank` refused when plan.json diverged from plan.registered.json and told the agent
    to "restore the registered plan" — which an autonomous agent can read as "make the
    registration match". One `cp plan.json plan.registered.json` then ranked cleanly
    under a weighting no register_plan record ever authorised. The decision log is the
    anchor the design claimed and did not check: `rank` now verifies the registration
    against it, so overwriting the file only moves the refusal.
    """
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    for candidate in ("a", "b"):
        _embed(cli, run, candidate)
        cli("evaluate", "--run-dir", run, "--id", candidate)

    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["evaluation"]["weights"] = {"runtime_s": 1.0}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("rank", "--run-dir", run).code == 2

    # The bypass: make the registration agree with the edited plan.
    (run / "plan.registered.json").write_text(
        (run / "plan.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    result = cli("rank", "--run-dir", run)

    assert result.code == 2
    assert "decision log" in result.stderr
    assert not (run / "ranking.json").exists()
    log = (run / "decisions.jsonl").read_text(encoding="utf-8")
    assert "runtime_s" not in log


def test_the_divergence_message_does_not_offer_overwriting_the_registration(
    cli, csv_dataset, tmp_path
):
    """The message is the agent's only instruction, so it must not name the bypass."""
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["evaluation"]["weights"] = {"runtime_s": 1.0}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("rank", "--run-dir", run)

    assert result.code == 2
    assert "copy plan.registered.json back over plan.json" in result.stderr
    assert "start a new run" in result.stderr


def test_rank_refuses_a_registration_the_log_never_witnessed(cli, csv_dataset, tmp_path):
    """A registration with no record at all, not merely one that disagrees."""
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    _embed(cli, run, "a")
    cli("evaluate", "--run-dir", run, "--id", "a")
    records = [
        line
        for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["stage"] != "register_plan"
    ]
    (run / "decisions.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")

    result = cli("rank", "--run-dir", run)

    assert result.code == 2
    assert "no plan registration" in result.stderr
    assert not (run / "ranking.json").exists()


# ------------------------------------------- a ceiling the plan declared for itself


def _prepared_with_ceiling(cli, csv_dataset, tmp_path, ceiling):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    plan = _plan({"trustworthiness": 1.0})  # candidates a and b
    plan["max_candidates"] = ceiling
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert cli("validate-plan", "--run-dir", runs / "r1").code == 0
    return runs / "r1"


def test_a_declared_ceiling_is_recorded_where_it_cannot_be_rewritten(
    cli, csv_dataset, tmp_path
):
    """plan.registered.json is a file and is overwritten by the next registration.

    The log is the only place a declaration made before the results existed survives
    one made after them. Without it, a ceiling of 2 raised to 5 leaves no trace that 2
    was ever claimed, so the self-restraint is unfalsifiable rather than checkable.
    """
    run = _prepared_with_ceiling(cli, csv_dataset, tmp_path, 2)

    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    registration = [r for r in records if r["stage"] == "register_plan"][-1]

    assert registration["max_candidates"] == 2


def test_no_declared_ceiling_records_none_rather_than_omitting_the_key(
    cli, csv_dataset, tmp_path
):
    """Absent and unstated must be distinguishable in the record."""
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})

    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    registration = [r for r in records if r["stage"] == "register_plan"][-1]

    assert "max_candidates" in registration
    assert registration["max_candidates"] is None


def test_raising_a_declared_ceiling_after_registering_refuses(
    cli, csv_dataset, tmp_path
):
    """Declared before any result existed, so it is pre-registration like the weighting.

    Re-registration froze the budget, the weighting and the base, and left this open,
    which let a run declare restraint at 2 and then register 5 candidates under it.
    """
    run = _prepared_with_ceiling(cli, csv_dataset, tmp_path, 2)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["max_candidates"] = 5
    plan["candidates"].append({"id": "c", "stages": PCA})
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "max_candidates" in result.stderr
    registered = json.loads((run / "plan.registered.json").read_text(encoding="utf-8"))
    assert registered["max_candidates"] == 2
    assert sorted(c["id"] for c in registered["candidates"]) == ["a", "b"]


def test_tightening_a_declared_ceiling_stays_legal(cli, csv_dataset, tmp_path):
    """More restraint than was claimed is not a loosening, and nothing rests on it."""
    run = _prepared_with_ceiling(cli, csv_dataset, tmp_path, 5)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["max_candidates"] = 3
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 0
    registered = json.loads((run / "plan.registered.json").read_text(encoding="utf-8"))
    assert registered["max_candidates"] == 3


def test_declaring_a_ceiling_where_none_was_declared_stays_legal(
    cli, csv_dataset, tmp_path
):
    """Going from the budget's ceiling to a tighter one of your own is a tightening."""
    run = _prepared(cli, csv_dataset, tmp_path, {"trustworthiness": 1.0})
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    plan["max_candidates"] = 3
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    assert cli("validate-plan", "--run-dir", run).code == 0


def test_dropping_a_declared_ceiling_back_to_the_budgets_refuses(
    cli, csv_dataset, tmp_path
):
    """Omitting the field is the loosening that looks like saying nothing."""
    run = _prepared_with_ceiling(cli, csv_dataset, tmp_path, 2)
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    del plan["max_candidates"]
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    result = cli("validate-plan", "--run-dir", run)

    assert result.code == 2
    assert "max_candidates" in result.stderr
