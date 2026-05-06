from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent.

    The model converts a linear score into a probability using sigmoid:

        p(y = 1 | x) = sigmoid(x @ w + b)

    Training minimizes binary cross-entropy.
    """

    learning_rate: float = 0.1
    n_iterations: int = 1000
    fit_intercept: bool = True
    threshold: float = 0.5
    weights_: np.ndarray | None = field(default=None, init=False)
    bias_: float = field(default=0.0, init=False)
    losses_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0])

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.losses_ = []

        for _ in range(self.n_iterations):
            probabilities = self.predict_proba(X, validate=False)
            errors = probabilities - y

            grad_w = (X.T @ errors) / n_samples
            grad_b = float(np.sum(errors) / n_samples)

            self.weights_ -= self.learning_rate * grad_w
            if self.fit_intercept:
                self.bias_ -= self.learning_rate * grad_b

            self.losses_.append(self._binary_cross_entropy(y, probabilities))

        return self

    def predict_proba(self, X: np.ndarray, validate: bool = True) -> np.ndarray:
        if self.weights_ is None:
            if validate:
                raise ValueError("Model must be fitted before calling predict_proba.")
            raise ValueError("Model weights are not initialized.")
        X = self._validate_X(X) if validate else X
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
        return float(np.mean(loss))

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
        y = np.asarray(y, dtype=int).reshape(-1)
        if y.shape[0] != n_samples:
            raise ValueError("y must contain one target per input sample.")
        unique = set(np.unique(y).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("LogisticRegressionGD supports binary labels 0 and 1.")
        return y
