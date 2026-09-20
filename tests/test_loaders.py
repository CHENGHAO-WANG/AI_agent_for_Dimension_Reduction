"""Dataset resolution, including the agent's adapter escape hatch.

The failure messages are part of the interface: when the agent hits an unsupported
format or writes a bad adapter, what it reads on stderr is what it has to repair from.
"""

from __future__ import annotations

import numpy as np
import pytest

from drtools.contract import ContractError
from drtools.loaders import available, load

ADAPTER_SOURCE = """
import numpy as np

def load(path, **kwargs):
    return (
        np.arange(30, dtype=np.float64).reshape(10, 3),
        np.arange(10) % 2,
        {"name": "adapted", "source": str(path)},
    )
"""


def test_lists_everything_loadable_without_an_adapter() -> None:
    catalogue = available()
    assert "swiss_roll" in catalogue["synthetic"]
    assert "pbmc3k" in catalogue["named"]
    assert ".csv" in catalogue["file_suffixes"]


def test_loads_a_synthetic_fixture_by_name() -> None:
    X, labels, meta = load("swiss_roll", n_samples=100)
    assert X.shape == (100, 3)
    assert labels.shape == (100,)
    assert meta["spec"] == "swiss_roll"


def test_round_trips_an_npz_with_labels(tmp_path) -> None:
    path = tmp_path / "data.npz"
    np.savez(path, X=np.eye(8, 4), labels=np.arange(8) % 3)

    X, labels, meta = load(str(path))

    assert X.shape == (8, 4)
    assert labels is not None and labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
    assert meta["name"] == "data"


def test_reads_a_csv_and_factorises_its_label_column(tmp_path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("a,b,label\n1,2,cat\n3,4,dog\n5,6,cat\n", encoding="utf-8")

    X, labels, meta = load(str(path))

    assert X.shape == (3, 2)
    assert labels.tolist() == [0, 1, 0]
    assert meta["label_names"] == ["cat", "dog"]
    assert meta["feature_names"] == ["a", "b"]


def test_refuses_a_dataset_with_missing_values(tmp_path) -> None:
    """Missing values are out of scope, and the refusal must say so.

    It used to say imputation was "a preprocessing decision for the agent to make
    explicitly" -- a route that does not exist. No registry op imputes, and all ten
    reductions refuse NaN outright, so the only path that sentence left open was an
    adapter filling the holes in silently, after which audited metrics treat
    fabricated numbers as observations.
    """
    path = tmp_path / "holes.csv"
    path.write_text("a,b\n1,2\n3,\n", encoding="utf-8")

    with pytest.raises(ContractError) as error:
        load(str(path))

    message = str(error.value)
    assert "missing values" in message
    assert "out of scope" in message
    assert "preprocessing decision" not in message


def test_names_the_offending_column_when_it_is_not_numeric(tmp_path) -> None:
    path = tmp_path / "text.csv"
    path.write_text("a,note\n1,hello\n2,world\n", encoding="utf-8")

    with pytest.raises(ContractError, match="note"):
        load(str(path))


def test_unsupported_suffix_points_at_the_adapter_route(tmp_path) -> None:
    path = tmp_path / "mystery.parquet"
    path.write_bytes(b"not really parquet")

    with pytest.raises(ContractError, match="--adapter"):
        load(str(path))


def test_missing_file_lists_the_built_in_names() -> None:
    with pytest.raises(ContractError, match="swiss_roll"):
        load("definitely/not/here.csv")


def test_an_agent_written_adapter_is_accepted_when_it_honours_the_contract(
    tmp_path,
) -> None:
    adapter = tmp_path / "my_adapter.py"
    adapter.write_text(ADAPTER_SOURCE, encoding="utf-8")

    X, labels, meta = load("whatever.weird", adapter=str(adapter))

    assert X.shape == (10, 3)
    assert meta["name"] == "adapted"
    assert meta["adapter"].endswith("my_adapter.py")


def test_an_adapter_that_breaks_the_contract_is_still_rejected(tmp_path) -> None:
    """Agent-written code gets no more trust than a built-in loader."""
    adapter = tmp_path / "bad_adapter.py"
    adapter.write_text(
        "import numpy as np\n"
        "def load(path, **kwargs):\n"
        "    return np.zeros((5, 2)), np.arange(3), {'name': 'x', 'source': 'y'}\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="misaligned"):
        load("whatever", adapter=str(adapter))


def test_an_adapter_without_a_load_function_says_what_is_required(tmp_path) -> None:
    adapter = tmp_path / "empty_adapter.py"
    adapter.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ContractError, match=r"load\(path"):
        load("whatever", adapter=str(adapter))
