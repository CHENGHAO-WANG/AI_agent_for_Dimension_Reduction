import numpy as np
import pytest

from drtools import jsonio
from drtools.cache import ensure_cache, read_cache, read_sample_ids, write_cache
from drtools.contract import ContractError
from drtools.runs import RunDir

META = {"name": "t", "source": "t"}


def _run(tmp_path):
    return RunDir(tmp_path / "r1")


def test_write_cache_records_a_digest(tmp_path):
    X = np.ones((4, 3))
    meta = write_cache(_run(tmp_path), X, None, META)
    assert len(meta["dataset_digest"]) == 64


def test_ensure_cache_accepts_identical_data(tmp_path):
    run = _run(tmp_path)
    X = np.ones((4, 3))
    write_cache(run, X, None, META)
    meta = ensure_cache(run, X.copy(), None, META)
    assert meta["cached_shape"] == [4, 3]


def test_ensure_cache_refuses_changed_data(tmp_path):
    run = _run(tmp_path)
    write_cache(run, np.ones((4, 3)), None, META)
    with pytest.raises(ContractError) as error:
        ensure_cache(run, np.ones((4, 2)), None, META)
    assert "new run" in str(error.value)


def test_ensure_cache_refuses_changed_labels(tmp_path):
    run = _run(tmp_path)
    X = np.ones((2, 3))
    write_cache(run, X, np.array([0, 1]), META)
    with pytest.raises(ContractError):
        ensure_cache(run, X, np.array([1, 0]), META)


def test_cached_data_still_reads_back(tmp_path):
    run = _run(tmp_path)
    write_cache(run, np.arange(12.0).reshape(4, 3), np.array([0, 1, 0, 1]), META)
    X, labels, meta = read_cache(run)
    assert X.shape == (4, 3)
    assert labels.tolist() == [0, 1, 0, 1]


def test_identifiers_survive_the_cache_in_their_own_file(tmp_path):
    run = _run(tmp_path)
    ids = ["cell-a", "cell-b", "cell-c", "cell-d"]
    meta = write_cache(run, np.ones((4, 3)), None, {**META, "sample_ids": ids,
                                                   "sample_ids_source": "loader"})
    assert read_sample_ids(run) == ids
    assert "sample_ids" not in meta
    assert meta["sample_ids_source"] == "loader" and meta["n_sample_ids"] == 4


def test_ensure_cache_refuses_renamed_rows(tmp_path):
    run = _run(tmp_path)
    X = np.ones((2, 3))
    write_cache(run, X, None, {**META, "sample_ids": ["a", "b"]})
    with pytest.raises(ContractError, match="different dataset"):
        ensure_cache(run, X, None, {**META, "sample_ids": ["b", "a"]})


def test_a_cache_from_before_identifiers_is_told_so(tmp_path):
    """Review focus 3: not 'a different dataset' -- the rule changed, not the data."""
    run = _run(tmp_path)
    X = np.ones((4, 3))
    write_cache(run, X, None, META)
    path = run.path / "data" / "meta.json"
    legacy = jsonio.read(path)
    legacy.pop("sample_ids_source")
    jsonio.write(path, legacy)
    with pytest.raises(ContractError, match="before sample identifiers"):
        ensure_cache(run, X, None, META)
