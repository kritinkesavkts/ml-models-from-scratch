from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml_from_scratch.tree_models.decision_tree import DecisionTreeRegressor


@dataclass
class GradientBoostingRegressor:
    """Gradient boosting regressor for squared error loss.

    Each tree learns the residuals left by the current ensemble prediction.
    The final prediction is the initial mean plus the weighted sum of trees.
    """

    n_estimators: int = 50
    learning_rate: float = 0.1
    max_depth: int = 2
    min_samples_split: int = 2
    random_state: int | None = None
    initial_prediction_: float = field(default=0.0, init=False)
    estimators_: list[DecisionTreeRegressor] = field(default_factory=list, init=False)
    losses_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one target per input sample.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        rng = np.random.default_rng(self.random_state)
        self.initial_prediction_ = float(np.mean(y))
        current_prediction = np.full_like(y, self.initial_prediction_, dtype=float)
        self.estimators_ = []
        self.losses_ = []

        for _ in range(self.n_estimators):
            residuals = y - current_prediction
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_prediction += self.learning_rate * update
            self.estimators_.append(tree)
            self.losses_.append(float(np.mean((y - current_prediction) ** 2)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_X(X)
        predictions = np.full(X.shape[0], self.initial_prediction_, dtype=float)
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X
