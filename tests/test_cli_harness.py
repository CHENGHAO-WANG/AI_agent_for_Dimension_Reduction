"""Pins the harness itself: the CLI is reachable in-process, a success prints its
payload as JSON on stdout with exit code 0, and a refusal reports a non-zero code
on stderr rather than raising out of `main`.
"""


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
