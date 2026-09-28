import json
from plans import complete, reconnoitre

PCA = [{"op": "pca", "params": {}}]


def _prepared(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    reconnoitre(cli, runs / "r1")
    plan = {
        "dataset": "d",
        "candidates": [{"id": "a", "stages": PCA}],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "up front"},
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(complete(plan)), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    return runs / "r1"


def test_embedding_without_a_registered_plan_is_refused(cli, csv_dataset, tmp_path):
    runs = tmp_path / "runs"
    cli("profile", "--data", csv_dataset(rows=60, cols=8),
        "--runs-root", runs, "--run-id", "r1")
    reconnoitre(cli, runs / "r1")
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
    # d is chosen by tuning inside the Attempt (day 15), and the record says which.
    chosen = result.payload["tuning"]["chosen"]["d"]
    assert result.payload["output_shape"] == [60, chosen]
    assert [s["op"] for s in result.payload["stages"]][-1] == "pca"


def test_every_attempt_is_recorded_with_its_outcome(cli, csv_dataset, tmp_path):
    run = _prepared(cli, csv_dataset, tmp_path)
    cli("embed", "--run-dir", run, "--id", "a", "--in-process")
    records = [json.loads(line) for line in
               (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    attempts = [r for r in records if r["stage"] == "embed"]
    assert len(attempts) == 1
    assert attempts[0]["candidate"] == "a"
    assert attempts[0]["outcome"] == "ok"
