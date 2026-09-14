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
