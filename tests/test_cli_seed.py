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


def test_evaluating_without_seed_is_still_reproducible(cli, csv_dataset, tmp_path):
    """`evaluate` no longer takes --k/--max-samples: the neighbourhood and the sample
    cap are both fixed by rule. Above METRIC_SAMPLE_CAP is what makes that a real
    check rather than a vacuous one — the metric subsample is then a strict subset of
    the rows, drawn under the run's recorded seed, so two evaluations only agree if
    that seed was actually held fixed rather than redrawn each time.
    """
    runs = tmp_path / "runs"
    data = csv_dataset(rows=2500, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1", "--seed", 0)
    plan = {
        "dataset": "d",
        "candidates": [
            {"id": "c1", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "pinned"},
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("embed", "--run-dir", runs / "r1", "--id", "c1", "--in-process")

    first = cli("evaluate", "--run-dir", runs / "r1", "--id", "c1")
    second = cli("evaluate", "--run-dir", runs / "r1", "--id", "c1")

    assert first.code == 0
    assert second.code == 0
    assert first.payload["values"] == second.payload["values"]


def test_evaluating_with_a_conflicting_seed_is_refused(cli, csv_dataset, tmp_path):
    """`evaluate` takes its metric settings from reference.json, not from --seed, but
    a --seed that disagrees with the run's recorded one must still be refused rather
    than silently discarded — the same discipline every other command follows.
    """
    runs = tmp_path / "runs"
    data = csv_dataset(rows=60, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1", "--seed", 0)
    plan = {
        "dataset": "d",
        "candidates": [
            {"id": "c1", "stages": [{"op": "pca", "params": {"n_components": 2}}]}
        ],
        "evaluation": {"weights": {"trustworthiness": 1.0}, "justification": "pinned"},
    }
    (runs / "r1" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    cli("validate-plan", "--run-dir", runs / "r1")
    cli("embed", "--run-dir", runs / "r1", "--id", "c1", "--in-process")

    result = cli("evaluate", "--run-dir", runs / "r1", "--id", "c1", "--seed", 7)

    assert result.code == 2
    assert "seed" in result.stderr
    assert not (runs / "r1" / "metrics" / "c1.json").exists()
