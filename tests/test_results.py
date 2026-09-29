"""Day 19: the `results/` folder, the winner's export, and the report's new blocks.

The report, its figures and the export live together in `results/`, so the folder can
be shared alone (section 5). The export holds the winner only: its coordinates with
sample identifiers and which rows were fitted, a manifest a refit can work from, and
loadings where the method has them. The report gains the d curves, the standard errors,
the path of winners, coverage and run times, the close competitors' diagnostics, the
reproducibility line and a section on the export; and `render` refuses a section 8 that
leaves a close competitor unnamed (section 3.8).
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from drtools import jsonio
from drtools.export import exported_candidate
from drtools.report import SECTIONS, build_blocks, unnamed_competitors
from drtools.runs import RunDir


def test_the_report_and_its_figures_live_in_results(cli, finished_run):
    assert (finished_run.figures_dir / "comparison.png").exists()
    assert not (finished_run.path / "figures").exists()

    result = cli("report", "--run-dir", finished_run.path)

    assert result.code == 0 and finished_run.report_path.exists()
    assert finished_run.report_path.parent == finished_run.results_dir


def test_the_winner_is_exported_with_identifiers_rows_and_loadings(cli, finished_run):
    cli("report", "--run-dir", finished_run.path)
    data = finished_run.results_dir / "data"
    embedding = np.load(finished_run.path / "embeddings" / "pca-2.npy")

    table = pd.read_csv(data / "pca-2.csv")
    manifest = jsonio.read(data / "manifest.json")
    loadings = pd.read_csv(data / "pca-2.loadings.csv")

    assert list(table.columns[:2]) == ["sample_id", "fitted"]
    assert len(table) == embedding.shape[0] and set(table["fitted"]) == {"fitted"}
    assert np.allclose(table.filter(like="dim_").to_numpy(), embedding)
    assert manifest["candidate"] == "pca-2" and manifest["d"] == embedding.shape[1]
    assert manifest["pipeline"][-1]["op"] == "pca"
    assert manifest["places_new_samples_without_refitting"] is True
    # The base z-scores every feature, so its means and deviations are carried by name.
    zscore = manifest["zscore"][0]
    assert len(zscore["features"]) == len(zscore["mean"]) == len(zscore["std"])
    assert list(loadings.columns) == ["feature"] + [f"dim_{j + 1}" for j in range(manifest["d"])]
    assert list(loadings["feature"]) == manifest["features_kept"]


def test_a_visualization_run_exports_only_a_current_adoption(tmp_path):
    run = RunDir(tmp_path / "r")
    jsonio.write(run.path / "checkpoint.json", {"purpose": "visualization"})
    jsonio.write(run.path / "comparison.json", {"plan_digest": "now"})

    assert exported_candidate(run)[0] is None
    jsonio.write(run.path / "adopted.json", {"candidate": "umap", "plan_digest": "before"})
    assert exported_candidate(run)[0] is None
    jsonio.write(run.path / "adopted.json", {"candidate": "umap", "plan_digest": "now"})
    assert exported_candidate(run)[0] == "umap"


def test_the_report_has_ten_sections_and_the_new_blocks(finished_run):
    blocks = build_blocks(finished_run)

    assert SECTIONS[-1] == ("10. Exported results", "export")
    assert "Paired SE" in blocks["ranking"] and "Path of winners" in blocks["ranking"]
    assert "Rows projected" in blocks["metrics"]
    assert "Tuned: d =" in blocks["hyperparameters"]
    assert "reproducible conditional on its registered plan" in blocks["limitations"]


@pytest.fixture
def contested(finished_run, tmp_path):
    """The finished run copied, with a second candidate the margin cannot separate."""
    path = tmp_path / "r1"
    shutil.copytree(finished_run.path, path)
    for suffix in (".npy", ".json", ".labels.npy", ".params.npz"):
        source = path / "embeddings" / f"pca-2{suffix}"
        if source.exists():
            shutil.copy(source, path / "embeddings" / f"rival{suffix}")
    for plot in path.glob("plots/pca-2_*.npy"):
        shutil.copy(plot, plot.with_name(plot.name.replace("pca-2", "rival")))
    record = jsonio.read(path / "embeddings" / "rival.json")
    jsonio.write(path / "embeddings" / "rival.json", {**record, "id": "rival"})
    shutil.copy(path / "metrics" / "pca-2.json", path / "metrics" / "rival.json")
    ranking = jsonio.read(path / "ranking.json")
    winner = ranking["ranking"][0]
    ranking["ranking"].append({**winner, "id": "rival", "rank": 2, "close_competitor": True})
    ranking["close_competitors"] = [{"id": "rival", "d": 2, "difference_from_winner": -0.001}]
    jsonio.write(path / "ranking.json", ranking)
    return RunDir(path)


def test_a_close_competitor_gets_the_diagnostic_figures(cli, contested):
    drawn = cli("figures", "--run-dir", contested.path).payload

    assert {"class_facet", "shepard", "class_facet_rival", "shepard_rival"} <= set(drawn)
    assert "close competitor" in build_blocks(contested)["figures"]


def test_render_refuses_a_section_8_that_leaves_a_competitor_unnamed(cli, contested):
    cli("report", "--run-dir", contested.path)
    document = contested.report_path.read_text(encoding="utf-8")
    assert unnamed_competitors(contested, document) == ["rival"]

    refused = cli("render", "--run-dir", contested.path)
    assert refused.code == 2 and "rival" in refused.stderr

    named = document.replace(
        "## 8. Interpretation\n", "## 8. Interpretation\n\nrival keeps the same geometry.\n"
    )
    assert unnamed_competitors(contested, named) == []


def test_the_fitted_arrays_are_mapped_to_the_cached_columns(finished_run):
    arrays = np.load(finished_run.path / "embeddings" / "pca-2.params.npz")
    keys = set(arrays.files)
    last = len(jsonio.read(finished_run.path / "embeddings" / "pca-2.json")["stages"]) - 1

    assert f"{last}.components" in keys and "features_kept" in keys
    assert np.array_equal(arrays[f"{last}.features"], arrays["features_kept"])
    assert json.dumps(sorted(keys))  # every key is a plain string
