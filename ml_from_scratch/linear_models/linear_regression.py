from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LinearRegressionGD:
    """Linear regression trained with batch gradient descent.

    The model learns weights `w` and bias `b` for the equation:

        y_hat = X @ w + b

    Training minimizes mean squared error.
    """

    learning_rate: float = 0.01
    n_iterations: int = 1000
    fit_intercept: bool = True
    weights_: np.ndarray | None = field(default=None, init=False)
    bias_: float = field(default=0.0, init=False)
    losses_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0])

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.losses_ = []

        for _ in range(self.n_iterations):
            predictions = X @ self.weights_ + self.bias_
            errors = predictions - y

            grad_w = (2.0 / n_samples) * (X.T @ errors)
            grad_b = float((2.0 / n_samples) * np.sum(errors))

            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

            loss = float(np.mean(errors**2))
            self.losses_.append(loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model must be fitted before calling predict.")
        X = self._validate_X(X)
        return X @ self.weights_ + self.bias_

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError("y must contain one target per input sample.")
        return y
