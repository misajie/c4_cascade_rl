"""Ridge / linear source→target delta predictor (numpy / sklearn; no new torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class DeltaPredictor:
    """Predict target delta expression from source delta (or multi-hot condition)."""

    alpha: float = 1.0
    fit_intercept: bool = True
    _model: Ridge | None = None
    n_features_: int = 0
    n_outputs_: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DeltaPredictor":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self._model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        self._model.fit(X, Y)
        self.n_features_ = X.shape[1]
        self.n_outputs_ = Y.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Predictor not fit")
        X = np.asarray(X, dtype=np.float64)
        pred = self._model.predict(X)
        return np.asarray(pred, dtype=np.float64)

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Mean squared error on audit set."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        pred = self.predict(X)
        return float(np.mean((pred - Y) ** 2))


def condition_means(
    X: np.ndarray,
    cond_ids: np.ndarray,
    n_conditions: int,
) -> np.ndarray:
    """Return [n_conditions, n_genes] mean expression."""
    n_genes = X.shape[1]
    means = np.zeros((n_conditions, n_genes), dtype=np.float64)
    for c in range(n_conditions):
        mask = cond_ids == c
        if mask.any():
            means[c] = X[mask].mean(axis=0)
    return means


def deltas_from_control(means: np.ndarray, control_id: int = 0) -> np.ndarray:
    return means - means[control_id]


def multi_hot(indices: Sequence[int], n_conditions: int) -> np.ndarray:
    m = np.zeros((len(indices), n_conditions), dtype=np.float64)
    for row, i in enumerate(indices):
        m[row, int(i)] = 1.0
    return m


def build_xy_from_revealed(
    source_delta: np.ndarray,
    target_delta: np.ndarray,
    revealed_ids: Sequence[int],
    feature: str = "source_delta",
) -> tuple[np.ndarray, np.ndarray]:
    """Build training matrices for revealed conditions.

    feature='source_delta': X = source delta vectors
    feature='multi_hot': X = multi-hot over condition index (n_features = n_conditions)
    """
    ids = list(revealed_ids)
    if not ids:
        raise ValueError("Need at least one revealed condition")
    Y = target_delta[ids]
    if feature == "source_delta":
        X = source_delta[ids]
    elif feature == "multi_hot":
        X = multi_hot(ids, source_delta.shape[0])
    else:
        raise ValueError(feature)
    return X, Y
