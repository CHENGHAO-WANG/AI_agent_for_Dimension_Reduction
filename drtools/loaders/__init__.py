"""Resolving a dataset specification into a contract-checked dataset.

A spec is one of three things, tried in order:

1. a built-in name — a synthetic fixture (`swiss_roll`) or a named dataset (`pbmc3k`);
2. a path whose suffix is recognised — `.npy`, `.npz`, `.csv`, `.tsv`, `.h5ad`;
3. a path plus an adapter module, which is the agent's escape hatch.

Whatever the route, the result passes through `check_dataset` before it is returned.
An agent-written adapter is therefore no more trusted than a built-in loader.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp

from drtools.contract import ContractError, Dataset, check_dataset
from drtools.synthetic import GENERATORS, generate

FileLoader = Callable[..., Dataset]

LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "target", "y", "cell_type")


def data_dir() -> Path:
    """Where downloaded datasets are cached. Git-ignored; override for a shared cache."""
    root = os.environ.get("DRTOOLS_DATA_DIR")
    path = Path(root) if root else Path.cwd() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load(spec: str, *, adapter: str | Path | None = None, **kwargs: Any) -> Dataset:
    """Load `spec` and verify it against the loader contract before returning it."""
    if adapter is not None:
        X, labels, meta = _load_via_adapter(spec, adapter, **kwargs)
        origin = f"adapter {Path(adapter).name}"
    elif spec in GENERATORS:
        X, labels, meta = generate(spec, **kwargs)
        origin = f"synthetic loader {spec!r}"
    elif spec in NAMED_DATASETS:
        X, labels, meta = NAMED_DATASETS[spec](**kwargs)
        origin = f"dataset loader {spec!r}"
    else:
        X, labels, meta = _load_file(Path(spec), **kwargs)
        origin = f"file loader for {Path(spec).suffix or 'unknown suffix'}"

    check_dataset(X, labels, meta, origin=origin)
    meta.setdefault("spec", spec)
    return X, labels, meta


# --------------------------------------------------------------------------- files


def _load_file(path: Path, **kwargs: Any) -> Dataset:
    if not path.exists():
        raise ContractError(
            f"no such dataset: {path}. Built-in names are "
            f"{', '.join(sorted(set(GENERATORS) | set(NAMED_DATASETS)))}."
        )
    try:
        loader = FILE_LOADERS[path.suffix.lower()]
    except KeyError:
        raise ContractError(
            f"no built-in loader for {path.suffix!r}. Supported: "
            f"{', '.join(sorted(FILE_LOADERS))}. Write an adapter exposing "
            "load(path) -> (X, labels, meta) and pass it with --adapter."
        ) from None
    return loader(path, **kwargs)


def _load_npy(path: Path, **_: Any) -> Dataset:
    X = np.load(path)
    return _as_float(X), None, {"name": path.stem, "source": str(path)}


def _load_npz(path: Path, **_: Any) -> Dataset:
    """Expects an `X` array, optionally a label array under a conventional name."""
    archive = np.load(path, allow_pickle=False)
    if "X" not in archive:
        raise ContractError(
            f"{path.name}: expected an array named 'X', found "
            f"{sorted(archive.files)}"
        )
    labels = None
    for key in LABEL_COLUMN_CANDIDATES:
        if key in archive:
            labels = archive[key].ravel().astype(np.int64)
            break
    return _as_float(archive["X"]), labels, {"name": path.stem, "source": str(path)}


def _load_table(path: Path, *, label_column: str | None = None, **_: Any) -> Dataset:
    import pandas as pd

    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=separator)

    column = label_column or next(
        (c for c in frame.columns if str(c).lower() in LABEL_COLUMN_CANDIDATES), None
    )
    labels, label_names = None, None
    if column is not None:
        codes, uniques = pd.factorize(frame[column], sort=True)
        labels = codes.astype(np.int64)
        label_names = [str(u) for u in uniques]
        frame = frame.drop(columns=[column])

    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < frame.shape[1]:
        dropped = sorted(set(frame.columns) - set(numeric.columns))
        raise ContractError(
            f"{path.name}: non-numeric column(s) {dropped} cannot be embedded. Encode "
            "them, drop them, or name one with --label-column."
        )
    if numeric.isna().any().any():
        n_missing = int(numeric.isna().sum().sum())
        raise ContractError(
            f"{path.name}: {n_missing} missing values. Datasets with missing values "
            "are out of scope for this toolbox, and an adapter that filled them in "
            "would put fabricated numbers into audited results with nothing recorded. "
            "Supply a complete table, or resolve the missingness upstream where the "
            "choice can be written down."
        )

    meta = {
        "name": path.stem,
        "source": str(path),
        "feature_names": [str(c) for c in numeric.columns],
    }
    if label_names is not None:
        meta["label_names"] = label_names
        meta["label_kind"] = "ground_truth"
        meta["label_column"] = str(column)
    return _as_float(numeric.to_numpy()), labels, meta


def _load_h5ad(path: Path, *, label_column: str | None = None, **_: Any) -> Dataset:
    import anndata as ad

    adata = ad.read_h5ad(path)
    X = adata.X
    X = _as_float(X.tocsr() if sp.issparse(X) else np.asarray(X))

    column = label_column or next(
        (c for c in adata.obs.columns if str(c).lower() in LABEL_COLUMN_CANDIDATES),
        None,
    )
    labels, meta = None, {
        "name": path.stem,
        "source": str(path),
        "feature_names": [str(v) for v in adata.var_names],
    }
    if column is not None:
        import pandas as pd

        codes, uniques = pd.factorize(adata.obs[column], sort=True)
        labels = codes.astype(np.int64)
        meta["label_names"] = [str(u) for u in uniques]
        meta["label_kind"] = "ground_truth"
        meta["label_column"] = str(column)
    return X, labels, meta


FILE_LOADERS: dict[str, FileLoader] = {
    ".npy": _load_npy,
    ".npz": _load_npz,
    ".csv": _load_table,
    ".tsv": _load_table,
    ".txt": _load_table,
    ".h5ad": _load_h5ad,
}


# ------------------------------------------------------------------------ adapters


def _load_via_adapter(spec: str, adapter: str | Path, **kwargs: Any) -> Dataset:
    """Run an agent-written adapter module exposing `load(path, **kwargs)`.

    The result is contract-checked by the caller exactly like a built-in loader's.
    """
    adapter_path = Path(adapter)
    if not adapter_path.exists():
        raise ContractError(f"adapter module not found: {adapter_path}")

    module_spec = importlib.util.spec_from_file_location(
        f"drtools_adapter_{adapter_path.stem}", adapter_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ContractError(f"{adapter_path.name} is not an importable Python module")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    if not hasattr(module, "load"):
        raise ContractError(
            f"{adapter_path.name} defines no load(); an adapter must expose "
            "load(path, **kwargs) -> (X, labels, meta)"
        )
    result = module.load(spec, **kwargs)
    if not (isinstance(result, tuple) and len(result) == 3):
        raise ContractError(
            f"{adapter_path.name}: load() must return a 3-tuple (X, labels, meta), "
            f"got {type(result).__name__}"
        )
    X, labels, meta = result
    if isinstance(meta, dict):
        # Assigned, never defaulted: this is the toolbox recording which code it
        # executed, not the adapter describing itself. `setdefault` let the adapter
        # pre-empt the one field that says where the matrix came from.
        meta["adapter"] = _adapter_provenance(adapter_path)
    return X, labels, meta


def _adapter_provenance(adapter_path: Path) -> dict[str, Any]:
    """What ran, and what was in it.

    The path alone dates badly: a run recording only `my_adapter.py` describes a
    matrix produced by whatever that file holds when someone later opens it. The
    digest pins the source that actually ran.

    Deliberately separate from the dataset digest, which covers the matrix and label
    codes and excludes adapter source so that tidying an adapter does not invalidate
    a run. Identity asks what the data is; provenance asks what produced it, and the
    two are free to move independently.
    """
    source = adapter_path.read_bytes()
    return {
        "path": str(adapter_path),
        "sha256": hashlib.sha256(source).hexdigest(),
        "bytes": len(source),
    }


# ------------------------------------------------------------------ named datasets


def _as_float(X: Any) -> Any:
    """Cast to float32 unless already float, keeping sparse matrices sparse."""
    if sp.issparse(X):
        return X.astype(np.float32) if X.dtype not in (np.float32, np.float64) else X
    X = np.asarray(X)
    return X.astype(np.float32) if X.dtype not in (np.float32, np.float64) else X


def load_pbmc3k(**_: Any) -> Dataset:
    """2,700 PBMCs from 10x Genomics, as raw sparse UMI counts.

    Returned unnormalised and unfiltered on purpose: normalisation and gene selection
    are decisions the agent must make and justify, not things the loader does quietly.
    Unlabelled — deriving reference labels is likewise a decision left downstream.
    """
    import scanpy as sc

    adata = sc.datasets.pbmc3k()
    X = adata.X
    X = _as_float(X.tocsr() if sp.issparse(X) else np.asarray(X))
    meta = {
        "name": "pbmc3k",
        "source": "scanpy.datasets.pbmc3k",
        "modality": "scRNA-seq",
        "value_kind": "raw UMI counts",
        "label_kind": None,
        "feature_names": [str(v) for v in adata.var_names],
        "citation": "Zheng et al. 2017, 10x Genomics",
    }
    return X, None, meta


def load_pathmnist(*, split: str = "train", **_: Any) -> Dataset:
    """Colorectal histology tiles from MedMNIST, flattened to raw pixel vectors.

    28x28x3 -> 2,352 features, scaled to [0, 1]. Left dense and unreduced so that the
    agent confronts the actual dimensionality and sample count and has to plan around
    them.
    """
    import medmnist
    from medmnist import INFO

    dataset = medmnist.PathMNIST(split=split, download=True, root=str(data_dir()))
    images = np.asarray(dataset.imgs, dtype=np.float32) / 255.0
    X = images.reshape(images.shape[0], -1)
    labels = np.asarray(dataset.labels).ravel().astype(np.int64)

    names = INFO["pathmnist"]["label"]
    meta = {
        "name": f"pathmnist_{split}",
        "source": f"medmnist.PathMNIST(split={split!r})",
        "modality": "histology images",
        "value_kind": "pixel intensities in [0, 1]",
        "label_kind": "ground_truth",
        "label_names": [names[str(i)] for i in range(len(names))],
        "image_shape": list(images.shape[1:]),
        "citation": "Yang et al. 2023, MedMNIST v2",
    }
    return X, labels, meta


NAMED_DATASETS: dict[str, Callable[..., Dataset]] = {
    "pbmc3k": load_pbmc3k,
    "pathmnist": load_pathmnist,
}


def available() -> dict[str, list[str]]:
    """Everything `load` will accept without an adapter. Used by the CLI and skills."""
    return {
        "synthetic": sorted(GENERATORS),
        "named": sorted(NAMED_DATASETS),
        "file_suffixes": sorted(FILE_LOADERS),
    }
