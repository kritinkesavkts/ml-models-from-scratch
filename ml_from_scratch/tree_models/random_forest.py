from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.tree_models.decision_tree import DecisionTreeClassifier


@dataclass
class RandomForestClassifier:
    """Random forest classifier built from bootstrapped decision trees."""

    n_estimators: int = 10
    max_depth: int | None = 6
    min_samples_split: int = 2
    max_features: int | None = None
    random_state: int | None = None
    trees_: list[DecisionTreeClassifier] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = self._validate_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one label per input sample.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        max_features = self.max_features or max(1, int(np.sqrt(X.shape[1])))

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_X(X)
        tree_predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.array([self._majority_vote(col) for col in tree_predictions.T])

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> object:
        return Counter(labels.tolist()).most_common(1)[0][0]

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X
