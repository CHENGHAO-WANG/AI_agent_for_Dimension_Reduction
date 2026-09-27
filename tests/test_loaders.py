"""Dataset resolution, including the agent's adapter escape hatch.

The failure messages are part of the interface: when the agent hits an unsupported
format or writes a bad adapter, what it reads on stderr is what it has to repair from.
"""

from __future__ import annotations

import numpy as np
import hashlib

import pandas as pd
import pytest

from drtools.contract import ContractError
from drtools.digest import content_hash
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
    assert meta["adapter"]["path"].endswith("my_adapter.py")


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


def test_an_adapter_records_the_digest_of_the_code_that_ran(tmp_path) -> None:
    """The path says which file; only the digest says what was in it.

    An adapter is the one piece of agent-written code in the analysis, and a run that
    records only its path describes a matrix produced by whatever that file happens to
    contain when someone later looks.
    """
    adapter = tmp_path / "my_adapter.py"
    adapter.write_text(ADAPTER_SOURCE, encoding="utf-8")

    _, _, meta = load("whatever.weird", adapter=str(adapter))

    expected = hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert meta["adapter"]["sha256"] == expected
    assert meta["adapter"]["bytes"] == len(adapter.read_bytes())


def test_an_adapter_cannot_supply_its_own_provenance(tmp_path) -> None:
    """This record is the toolbox saying what it executed, not the adapter claiming it.

    `setdefault` let an adapter pre-empt the field, so the one record of which code
    produced the matrix could be written by that code.
    """
    adapter = tmp_path / "liar.py"
    adapter.write_text(
        ADAPTER_SOURCE.replace(
            '"source": str(path)',
            '"source": str(path), "adapter": "something/else.py"',
        ),
        encoding="utf-8",
    )

    _, _, meta = load("whatever.weird", adapter=str(adapter))

    assert meta["adapter"]["path"].endswith("liar.py")


def test_editing_an_adapter_moves_provenance_but_not_dataset_identity(tmp_path) -> None:
    """Provenance and identity answer different questions, and must move separately.

    Day 7 kept adapter source out of the dataset digest on purpose, so an adapter can
    be tidied without invalidating a run. The corollary is that the digest cannot then
    be the record of which code ran, so provenance has to carry that itself.
    """
    first = tmp_path / "a.py"
    first.write_text(ADAPTER_SOURCE, encoding="utf-8")
    second = tmp_path / "b.py"
    second.write_text(ADAPTER_SOURCE + "# a tidying comment", encoding="utf-8")

    X1, y1, meta1 = load("whatever", adapter=str(first))
    X2, y2, meta2 = load("whatever", adapter=str(second))

    assert content_hash(X1, y1) == content_hash(X2, y2), "same matrix, same identity"
    assert meta1["adapter"]["sha256"] != meta2["adapter"]["sha256"]


def test_a_csv_names_its_rows_from_the_id_column(tmp_path) -> None:
    path = tmp_path / "named.csv"
    pd.DataFrame({"sample": ["x", "y", "z"], "f0": [1.0, 2.0, 3.0],
                  "f1": [0.0, 1.0, 0.5]}).to_csv(path, index=False)
    X, _, meta = load(str(path), id_column="sample")
    assert X.shape == (3, 2)
    assert meta["sample_ids"] == ["x", "y", "z"]
    assert meta["sample_ids_source"] == "loader"


def test_an_unknown_id_column_is_refused_naming_the_columns(tmp_path) -> None:
    path = tmp_path / "named.csv"
    pd.DataFrame({"f0": [1.0, 2.0], "f1": [0.0, 1.0]}).to_csv(path, index=False)
    with pytest.raises(ContractError, match="no column named 'sample'"):
        load(str(path), id_column="sample")


def test_an_id_column_for_a_format_without_columns_is_refused() -> None:
    """Review focus 4: silently ignoring it would leave the rows numbered unasked."""
    with pytest.raises(ContractError, match="has none"):
        load("blobs", id_column="sample")


def test_rows_a_loader_cannot_name_are_numbered_and_say_so() -> None:
    X, _, meta = load("blobs", n_samples=30)
    assert meta["sample_ids"] == [str(i) for i in range(30)]
    assert meta["sample_ids_source"] == "row_order"


def test_a_loader_cannot_claim_the_source_for_itself(tmp_path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import numpy as np\n"
        "def load(path, **_):\n"
        "    return np.ones((3, 2)), None, {'name': 'a', 'source': 'a',\n"
        "                                   'sample_ids_source': 'loader'}\n",
        encoding="utf-8",
    )
    _, _, meta = load("anything", adapter=adapter)
    assert meta["sample_ids_source"] == "row_order"


def test_an_npz_may_carry_its_row_names(tmp_path) -> None:
    path = tmp_path / "d.npz"
    np.savez(path, X=np.ones((3, 2)), sample_ids=np.array(["p", "q", "r"]))
    _, _, meta = load(str(path))
    assert meta["sample_ids"] == ["p", "q", "r"]


def test_the_cli_carries_an_id_column_into_the_run(tmp_path, cli) -> None:
    path = tmp_path / "named.csv"
    pd.DataFrame({"sample": [f"c{i}" for i in range(40)],
                  "f0": np.arange(40.0), "f1": np.arange(40.0) ** 0.5}).to_csv(path, index=False)
    result = cli("profile", "--data", path, "--runs-root", tmp_path / "runs",
                 "--run-id", "r1", "--id-column", "sample")
    assert result.code == 0, result.stderr
    from drtools.cache import read_sample_ids
    from drtools.runs import RunDir
    assert read_sample_ids(RunDir(tmp_path / "runs" / "r1"))[:2] == ["c0", "c1"]


def test_an_id_column_keeps_its_text_exactly(tmp_path) -> None:
    """read_csv would turn '001' into 1 and 'NA' into NaN, and '001' and '1' would collide."""
    path = tmp_path / "coded.csv"
    path.write_text("sample,f0,f1\n001,1.0,0.0\n1,2.0,1.0\nNA,3.0,0.5\n", encoding="utf-8")
    X, _, meta = load(str(path), id_column="sample")
    assert meta["sample_ids"] == ["001", "1", "NA"]
    assert X.shape == (3, 2)
