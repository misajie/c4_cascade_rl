"""Unit-test densify/mean helpers without real h5ad."""

from __future__ import annotations

import numpy as np

from rl_cross_context.h5ad_means import as_dense_2d, block_mean, means_by_condition


class _FakeAdata:
    def __init__(self, X, conditions):
        import pandas as pd

        self.X = X
        self.obs = pd.DataFrame({"condition": conditions})


def test_dense_block_mean():
    block = np.arange(12, dtype=np.float64).reshape(3, 4)
    m = block_mean(block)
    assert m.shape == (4,)
    np.testing.assert_allclose(m, block.mean(axis=0))


def test_means_by_condition_dense():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    conds = ["a", "a", "b", "ctrl"]
    ad = _FakeAdata(X, conds)
    means = means_by_condition(ad, "condition", ["a", "b", "ctrl"])
    np.testing.assert_allclose(means[0], [2.0, 3.0])
    np.testing.assert_allclose(means[1], [5.0, 6.0])
    np.testing.assert_allclose(means[2], [7.0, 8.0])


def test_as_dense_rejects_object():
    class Bad:
        def __array__(self, dtype=None):
            return np.array([None, None], dtype=object)

    try:
        as_dense_2d(Bad())
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_sparse_block_mean_if_scipy():
    try:
        import scipy.sparse as sp
    except ImportError:
        return
    block = sp.csr_matrix(np.array([[1.0, 0.0], [3.0, 4.0]]))
    m = block_mean(block)
    np.testing.assert_allclose(m, np.array([2.0, 2.0]))
