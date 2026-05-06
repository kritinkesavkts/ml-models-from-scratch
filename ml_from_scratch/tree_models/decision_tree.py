from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class _Node:
    prediction: float | int
    feature_index: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class DecisionTreeClassifier:
    """Decision tree classifier using Gini impurity."""

    max_depth: int | None = 5
    min_samples_split: int = 2
    max_features: int | None = None
    random_state: int | None = None
    root_: _Node | None = field(default=None, init=False)
    n_features_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = _validate_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one label per input sample.")
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build_tree(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = _validate_X(X)
        return np.array([self._traverse(row, self.root_) for row in X])

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> _Node:
        prediction = Counter(y.tolist()).most_common(1)[0][0]
        node = _Node(prediction=prediction)

        if self._should_stop(y, depth):
            return node

        feature_indices = _sample_features(self.n_features_, self.max_features, rng)
        split = _best_split(X, y, feature_indices, task="classification")
        if split is None:
            return node

        feature_index, threshold, left_mask = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, rng)
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if len(np.unique(y)) == 1:
            return True
        if len(y) < self.min_samples_split:
            return True
        return self.max_depth is not None and depth >= self.max_depth

    def _traverse(self, x: np.ndarray, node: _Node) -> float | int:
        if node.is_leaf:
            return node.prediction
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor using mean squared error reduction."""

    max_depth: int | None = 5
    min_samples_split: int = 2
    max_features: int | None = None
    random_state: int | None = None
    root_: _Node | None = field(default=None, init=False)
    n_features_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = _validate_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one target per input sample.")
        self.n_features_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build_tree(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = _validate_X(X)
        return np.array([self._traverse(row, self.root_) for row in X], dtype=float)

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> _Node:
        node = _Node(prediction=float(np.mean(y)))

        if self._should_stop(y, depth):
            return node

        feature_indices = _sample_features(self.n_features_, self.max_features, rng)
        split = _best_split(X, y, feature_indices, task="regression")
        if split is None:
            return node

        feature_index, threshold, left_mask = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1, rng)
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if len(y) < self.min_samples_split:
            return True
        if np.allclose(y, y[0]):
            return True
        return self.max_depth is not None and depth >= self.max_depth

    def _traverse(self, x: np.ndarray, node: _Node) -> float:
        if node.is_leaf:
            return float(node.prediction)
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    task: Literal["classification", "regression"],
) -> tuple[int, float, np.ndarray] | None:
    best_gain = 0.0
    best_split = None
    parent_impurity = _gini(y) if task == "classification" else _variance(y)

    for feature_index in feature_indices:
        values = np.unique(X[:, feature_index])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left_mask = X[:, feature_index] <= threshold
            if not left_mask.any() or left_mask.all():
                continue
            left_y = y[left_mask]
            right_y = y[~left_mask]
            gain = _impurity_reduction(parent_impurity, left_y, right_y, task)
            if gain > best_gain:
                best_gain = gain
                best_split = (int(feature_index), float(threshold), left_mask)

    return best_split


def _impurity_reduction(
    parent_impurity: float,
    left_y: np.ndarray,
    right_y: np.ndarray,
    task: Literal["classification", "regression"],
) -> float:
    n_total = len(left_y) + len(right_y)
    impurity_fn = _gini if task == "classification" else _variance
    weighted_child_impurity = (
        len(left_y) / n_total * impurity_fn(left_y)
        + len(right_y) / n_total * impurity_fn(right_y)
    )
    return float(parent_impurity - weighted_child_impurity)


def _gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return float(1.0 - np.sum(probabilities**2))


def _variance(y: np.ndarray) -> float:
    return float(np.var(y))


def _sample_features(
    n_features: int, max_features: int | None, rng: np.random.Generator
) -> np.ndarray:
    if max_features is None or max_features >= n_features:
        return np.arange(n_features)
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    return rng.choice(n_features, size=max_features, replace=False)


def _validate_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")
    return X
