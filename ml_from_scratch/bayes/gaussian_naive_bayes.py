from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier.

    The model assumes each feature follows a Gaussian distribution inside each
    class and that features are conditionally independent given the class.
    """

    var_smoothing: float = 1e-9
    classes_: np.ndarray | None = field(default=None, init=False)
    means_: dict[object, np.ndarray] = field(default_factory=dict, init=False)
    variances_: dict[object, np.ndarray] = field(default_factory=dict, init=False)
    priors_: dict[object, float] = field(default_factory=dict, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = self._validate_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != X.shape[0]:
            raise ValueError("y must contain one label per input sample.")

        self.classes_ = np.unique(y)
        self.means_.clear()
        self.variances_.clear()
        self.priors_.clear()

        for cls in self.classes_:
            X_cls = X[y == cls]
            self.means_[cls] = X_cls.mean(axis=0)
            self.variances_[cls] = X_cls.var(axis=0) + self.var_smoothing
            self.priors_[cls] = float(X_cls.shape[0] / X.shape[0])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model must be fitted before prediction.")
        X = self._validate_X(X)
        log_posteriors = np.array([self._log_posterior(row) for row in X])
        return self.classes_[np.argmax(log_posteriors, axis=1)]

    def _log_posterior(self, x: np.ndarray) -> list[float]:
        scores = []
        for cls in self.classes_:
            mean = self.means_[cls]
            var = self.variances_[cls]
            log_prior = np.log(self.priors_[cls])
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)
            scores.append(float(log_prior + log_likelihood))
        return scores

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")
        return X
