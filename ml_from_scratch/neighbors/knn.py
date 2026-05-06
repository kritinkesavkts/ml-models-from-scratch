from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class KNNClassifier:
    """K-nearest neighbors classifier.

    KNN is a lazy learner: training stores examples, and prediction compares
    each query point with the stored training points.
    """

    n_neighbors: int = 5
    X_train_: np.ndarray | None = field(default=None, init=False)
    y_train_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = _validate_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one label per input sample.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed number of training samples.")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_X(X)
        return np.array([self._predict_one(row) for row in X])

    def _predict_one(self, x: np.ndarray) -> object:
        distances = np.linalg.norm(self.X_train_ - x, axis=1)
        neighbor_idx = np.argsort(distances)[: self.n_neighbors]
        neighbor_labels = self.y_train_[neighbor_idx]
        return Counter(neighbor_labels.tolist()).most_common(1)[0][0]

    def _check_fitted(self) -> None:
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Model must be fitted before prediction.")


@dataclass
class KNNRegressor:
    """K-nearest neighbors regressor using the average neighbor target."""

    n_neighbors: int = 5
    X_train_: np.ndarray | None = field(default=None, init=False)
    y_train_: np.ndarray | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        X = _validate_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one target per input sample.")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors cannot exceed number of training samples.")
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_X(X)
        return np.array([self._predict_one(row) for row in X], dtype=float)

    def _predict_one(self, x: np.ndarray) -> float:
        distances = np.linalg.norm(self.X_train_ - x, axis=1)
        neighbor_idx = np.argsort(distances)[: self.n_neighbors]
        return float(np.mean(self.y_train_[neighbor_idx]))

    def _check_fitted(self) -> None:
        if self.X_train_ is None or self.y_train_ is None:
            raise ValueError("Model must be fitted before prediction.")


def _validate_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")
    return X
