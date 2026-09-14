"""Backed/sparse-safe per-condition means for Replogle h5ad (no full densify)."""

from __future__ import annotations

import numpy as np


def as_dense_2d(block) -> np.ndarray:
    """Densify a row-slice from ndarray / scipy.sparse / anndata CSRDataset."""
    try:
        import scipy.sparse as sp

        if sp.issparse(block):
            return np.asarray(block.toarray(), dtype=np.float64)
    except Exception:
        pass
    if hasattr(block, "toarray"):
        try:
            return np.asarray(block.toarray(), dtype=np.float64)
        except Exception:
            pass
    if hasattr(block, "to_memory"):
        try:
            return as_dense_2d(block.to_memory())
        except Exception:
            pass
    arr = np.asarray(block)
    if getattr(arr, "dtype", None) == object:
        raise TypeError(
            f"Cannot densify block type={type(block)!r}; need sparse.toarray or ndarray"
        )
    return np.asarray(arr, dtype=np.float64)


def block_mean(block) -> np.ndarray:
    """Mean over cells (axis=0) without casting the full matrix."""
    try:
        import scipy.sparse as sp

        if sp.issparse(block):
            return np.asarray(block.mean(axis=0)).ravel().astype(np.float64)
    except Exception:
        pass
    dense = as_dense_2d(block)
    if dense.ndim == 1:
        return dense.astype(np.float64, copy=False)
    return dense.mean(axis=0).astype(np.float64, copy=False)


def means_by_condition(adata, condition_col: str, conditions: list[str], chunk: int = 4096) -> np.ndarray:
    """Per-condition mean expression; works with backed CSRDataset / sparse / dense.

    Never calls np.asarray on the full `.X` (CSRDataset blows up).
    """
    obs = adata.obs[condition_col].astype(str)
    X = adata.X
    n_genes = int(X.shape[1])
    means = np.zeros((len(conditions), n_genes), dtype=np.float64)
    for i, c in enumerate(conditions):
        idx = np.flatnonzero(obs.values == c)
        if idx.size == 0:
            continue
        if idx.size <= chunk:
            means[i] = block_mean(X[idx])
            continue
        acc = np.zeros(n_genes, dtype=np.float64)
        n = 0
        for start in range(0, idx.size, chunk):
            sub = idx[start : start + chunk]
            block = as_dense_2d(X[sub])
            if block.ndim == 1:
                block = block[None, :]
            acc += block.sum(axis=0)
            n += int(block.shape[0])
        means[i] = acc / max(n, 1)
    return means
