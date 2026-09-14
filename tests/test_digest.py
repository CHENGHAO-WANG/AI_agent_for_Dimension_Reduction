import numpy as np
import scipy.sparse as sp

from drtools.digest import content_hash

BASE = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])


def test_equal_dense_matrices_agree():
    assert content_hash(BASE, None) == content_hash(BASE.copy(), None)


def test_different_data_differs():
    assert content_hash(BASE, None) != content_hash(BASE * 2, None)


def test_unsorted_indices_agree_with_sorted():
    sorted_ = sp.csr_matrix(BASE)
    unsorted = sp.csr_matrix(
        (np.array([2.0, 1.0, 3.0]), np.array([3, 1, 2]), np.array([0, 2, 3])),
        shape=(2, 4),
    )
    assert content_hash(unsorted, None) == content_hash(sorted_, None)


def test_explicit_zero_agrees_with_absent_zero():
    plain = sp.csr_matrix(BASE)
    padded = sp.csr_matrix(
        (np.array([1.0, 0.0, 2.0, 3.0]), np.array([1, 2, 3, 2]), np.array([0, 3, 4])),
        shape=(2, 4),
    )
    assert content_hash(padded, None) == content_hash(plain, None)


def test_index_width_does_not_change_identity():
    narrow = sp.csr_matrix(BASE)
    wide = sp.csr_matrix(BASE)
    wide.indices = wide.indices.astype(np.int64)
    wide.indptr = wide.indptr.astype(np.int64)
    assert content_hash(wide, None) == content_hash(narrow, None)


def test_labels_are_part_of_identity():
    a = np.array([0, 1])
    b = np.array([1, 0])
    assert content_hash(BASE, a) != content_hash(BASE, b)
    assert content_hash(BASE, a) != content_hash(BASE, None)
