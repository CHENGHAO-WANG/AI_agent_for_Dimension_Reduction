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
    runs = tmp_path / "runs"
    data = csv_dataset(rows=60, cols=8)
    cli("profile", "--data", data, "--runs-root", runs, "--run-id", "r1", "--seed", 0)
    cli("embed", "--run-dir", runs / "r1", "--id", "c1",
        "--stages", '[{"op":"pca","params":{"n_components":2}}]', "--in-process")

    first = cli(
        "evaluate", "--run-dir", runs / "r1", "--id", "c1", "--k", 3, "--max-samples", 20
    )
    second = cli(
        "evaluate", "--run-dir", runs / "r1", "--id", "c1", "--k", 3, "--max-samples", 20
    )

    assert first.code == 0
    assert second.code == 0
    assert first.payload["values"] == second.payload["values"]
