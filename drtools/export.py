"""The export in `results/data/`: the winner only (section 5).

A representation run exports the winner's representation at its chosen d; a
visualization run exports the picture the user adopted, and nothing when none was.
Every candidate's `.npy` stays in `embeddings/` regardless.

Three files. A CSV of the coordinates, first column the sample identifier, then whether
the method was fitted on the row or projected it (section 3.12); CSV because `.npy`
needs an extra package in R. A manifest carrying what a refit needs: the pipeline and
its values, the seed, d, the features kept, the z-score means and standard deviations,
whether the method places new samples without refitting, and the rows fitted and
projected. And loadings, for a method whose last stage kept its components -- PCA and
sparse PCA -- with the feature names attached. No saved model objects: a pickled model
often fails to load under another library version, so it would look reusable when it
may not be.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import pandas as pd

from drtools import jsonio
from drtools.registry import load_registry
from drtools.runs import RunDir


def exported_candidate(run: RunDir) -> tuple[str | None, str]:
    """The candidate to export, or None, with the reason either way."""
    if run.purpose() == "visualization":
        comparison = _read(run, "comparison.json") or {}
        adopted = _read(run, "adopted.json")
        if adopted and adopted.get("plan_digest") == comparison.get("plan_digest"):
            return adopted["candidate"], "the picture the user adopted"
        return None, "the user adopted no picture, so there is nothing to export"
    ranking = _read(run, "ranking.json")
    if ranking and ranking.get("winner"):
        return ranking["winner"], "the winner of the ranking"
    return None, "the run has no ranking, so there is no winner to export"


def write_export(run: RunDir) -> dict[str, Any]:
    """Write `results/data/` afresh for the current winner or adopted picture."""
    data = run.results_dir / "data"
    shutil.rmtree(data, ignore_errors=True)
    candidate, why = exported_candidate(run)
    if candidate is None:
        return {"candidate": None, "reason": why}
    data.mkdir(parents=True)

    embeddings = run.path / "embeddings"
    record = jsonio.read(embeddings / f"{candidate}.json")
    coordinates = np.load(embeddings / f"{candidate}.npy")
    n_rows, d = coordinates.shape
    fitted_path = embeddings / f"{candidate}.fitted.npy"
    fitted = np.zeros(n_rows, dtype=bool)
    fitted[np.load(fitted_path) if fitted_path.exists() else slice(None)] = True
    params_path = embeddings / f"{candidate}.params.npz"
    arrays = dict(np.load(params_path)) if params_path.exists() else {}

    ids = jsonio.read(run.path / "data" / "sample_ids.json")
    meta = jsonio.read(run.path / "data" / "meta.json")
    names = meta.get("feature_names") or [
        f"feature_{j}" for j in range(meta["cached_shape"][1])
    ]
    names = np.asarray(names, dtype=object)

    table = pd.DataFrame(coordinates, columns=[f"dim_{j + 1}" for j in range(d)])
    table.insert(0, "fitted", np.where(fitted, "fitted", "projected"))
    table.insert(0, "sample_id", ids)
    table.to_csv(data / f"{candidate}.csv", index=False)

    stages = record["stages"]
    last = stages[-1]["op"]
    zscore = [
        {
            "stage": position,
            "features": names[arrays[f"{position}.features"]].tolist(),
            "mean": arrays[f"{position}.mean"].tolist(),
            "std": arrays[f"{position}.std"].tolist(),
        }
        for position in range(len(stages))
        if f"{position}.mean" in arrays
    ]
    last_position = len(stages) - 1
    loadings_key = f"{last_position}.components"
    if loadings_key in arrays:
        components = arrays[loadings_key]
        features_key = f"{last_position}.features"
        # After an earlier reduction the method acts on that reduction's components,
        # which are named for it: `pca_1` and on.
        index = (
            names[arrays[features_key]]
            if features_key in arrays
            else [f"{stages[last_position - 1]['op']}_{j + 1}" for j in range(components.shape[1])]
        )
        loadings = pd.DataFrame(
            components.T,
            index=index,
            columns=[f"dim_{j + 1}" for j in range(components.shape[0])],
        )
        loadings.index.name = "feature"
        loadings.to_csv(data / f"{candidate}.loadings.csv")

    new_rows = load_registry()[last].raw.get("new_rows", "none")
    manifest = {
        "candidate": candidate,
        "why_this_candidate": why,
        "purpose": run.purpose(),
        "pipeline": [{"op": s["op"], "params": s["params"]} for s in stages],
        "seed": jsonio.read(run.manifest_path).get("seed"),
        "d": int(d),
        "features_kept": names[arrays["features_kept"]].tolist()
        if "features_kept" in arrays
        else None,
        "zscore": zscore,
        "places_new_samples_without_refitting": new_rows != "none",
        "new_rows": new_rows,
        "rows": record.get("rows"),
        "files": {
            "coordinates": f"{candidate}.csv",
            "loadings": f"{candidate}.loadings.csv" if loadings_key in arrays else None,
        },
        "no_model_objects": "a saved model often fails to load under another library "
        "version; refit from this pipeline, seed and data instead",
    }
    jsonio.write(data / "manifest.json", manifest)
    return {"candidate": candidate, "reason": why, "path": str(data)}


def _read(run: RunDir, name: str) -> Any | None:
    path = run.path / name
    return jsonio.read(path) if path.exists() else None
